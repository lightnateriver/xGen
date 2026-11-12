"""Ascend fusion dispatch for MiniMax-H3 blocks and request-local text reuse."""
from contextlib import contextmanager
from functools import partial
import logging
import threading
import weakref

import torch
from torch.nn import functional as F
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

from x_base import get_pad, run_phaa_attention
from x_base.operator.npu_fusion import capabilities as caps
from x_base.operator.npu_fusion.laser import attention_with_preprocess
from x_base.operator.npu_fusion.quantized import QuantizedRows
from x_base.operator.npu_fusion.runtime import Diagnostics, Kernels, no_hooks, quant_linear, quant_matmul, tensors_ok
from x_base.operator.npu_fusion.tensor_signature import signature
from x_base.phaa.fused_layout import make_fused_layout_ops

logger = logging.getLogger(__name__)
_original_h3_block_forward = MiniMaxH3TransformerBlock.forward


def _h3_ffn_layout(ff):
    net = getattr(ff, "net", ())
    return (type(ff) is FeedForward and len(net) == 3 and type(net[0]) is SwiGLU
            and type(net[1]) in (torch.nn.Identity, torch.nn.Dropout)
            and (type(net[1]) is torch.nn.Identity or not net[1].training or net[1].p == 0)
            and all(no_hooks(m) for m in ff.modules()))

def _h3_from_projected(model, ff, u):
    """Consume the actual up-projection width, without fixed B/S/H reshapes."""
    down = ff.net[2]
    if _h3_enabled(model, "swiglu"):
        valid = (tensors_ok(u) and u.ndim == 3 and u.stride(-1) == 1
                 and all(s >= 0 for s in u.stride())
                 and 0 < u.shape[-1] <= 2 * caps.SWIGLU_MAX_INNER and u.shape[-1] % 2 == 0
                 and quant_linear(down, u.shape[-1] // 2, u.device))
        if valid:
            kernel = model._h3_fusion_kernels.get("swiglu", u)
            if kernel is not None:
                q, scale = kernel.swiglu_quantize_reuse(u, tile=4096, blocks=40)
                result = quant_matmul(q, scale, down, u.shape)
                model._h3_fusion_diagnostics.hit("swiglu", f"shape={tuple(u.shape)}")
                return result
        else:
            model._h3_fusion_diagnostics.fallback("swiglu", "requires BF16 [B,S,2F], F<=14336 and native INT8 down projection without hooks")
    hidden, gate = u.chunk(2, dim=-1)
    value = hidden * F.silu(gate)
    for layer in ff.net[1:]:
        value = layer(value)
    return value

def _h3_residual(model, x, branch, gate, indices):
    if _h3_enabled(model, "gate_residual"):
        valid = (tensors_ok(x, branch, gate) and x.ndim == 3 and branch.shape == x.shape
                 and gate.ndim == 2 and gate.shape[0] > 0 and gate.shape[1] == x.shape[-1]
                 and indices.dtype == torch.int64 and indices.device == x.device
                 and indices.shape == (x.shape[1],) and indices.is_contiguous()
                 and all(t.stride(-1) == 1 and all(s >= 0 for s in t.stride()) for t in (x, branch, gate)))
        if valid:
            kernel = model._h3_fusion_kernels.get("gate_residual", x)
            if kernel is not None:
                value = kernel.gate_residual(x, branch, gate.detach(), indices,
                                             tile=5376 if x.shape[-1] == 5376 else 4096)
                model._h3_fusion_diagnostics.hit("gate_residual", f"shape={tuple(x.shape)}")
                return value
        else:
            model._h3_fusion_diagnostics.fallback("gate_residual", "residual/gate dtype, shape, device or strides unsupported")
    return x + gate.index_select(0, indices) * branch

def _h3_feed_forward(model, block, x, scale, shift, indices):
    if _h3_enabled(model, "norm_quant"):
        ff, norm = block.ff, block.norm2
        valid = (_h3_ffn_layout(ff) and type(norm) is torch.nn.RMSNorm and no_hooks(norm)
                 and tensors_ok(x, scale, shift) and x.ndim == 3 and 0 < x.shape[-1] <= caps.NORM_MAX_WIDTH
                 and x.stride(-1) == scale.stride(-1) == shift.stride(-1) == 1
                 and all(s >= 0 for t in (x, scale, shift) for s in t.stride())
                 and scale.ndim == 2 and shift.shape == scale.shape and scale.shape[0] > 0
                 and scale.shape[-1] == x.shape[-1] and indices.shape == (x.shape[1],)
                 and indices.dtype == torch.int64 and indices.device == x.device and indices.is_contiguous()
                 and norm.weight is not None and norm.weight.shape == (x.shape[-1],)
                 and norm.weight.dtype in (torch.bfloat16, torch.float32)
                 and norm.weight.device == x.device and norm.weight.is_contiguous()
                 and norm.eps is not None and 0 < norm.eps < 1
                 and quant_linear(ff.net[0].proj, x.shape[-1], x.device))
        if valid:
            kernel = model._h3_fusion_kernels.get("norm_quant", x)
            if kernel is not None:
                q, token_scale = kernel.norm_affine_quant(x, norm.weight.detach(), scale.detach(),
                                                        shift.detach(), indices, eps=norm.eps, cached=True)
                u = quant_matmul(q, token_scale, ff.net[0].proj, x.shape)
                model._h3_fusion_diagnostics.hit("norm_quant", f"shape={tuple(x.shape)}")
                return _h3_from_projected(model, ff, u)
        else:
            model._h3_fusion_diagnostics.fallback("norm_quant", "requires RMSNorm, H<=6144, compatible BF16 tables/indices and native INT8 FFN without hooks")
    n = block.norm2(x)
    n = n * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    return _h3_ffn(model, block.ff, n)

def _h3_project_qkv(attn, value):
    model = _h3_model(attn)
    if not _h3_enabled(model, "shared_qkv"):
        return None
    layers = [getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")]
    valid = (not attn.training and not getattr(attn, "fused_projections", False)
             and tensors_ok(value) and value.ndim == 3
             and not getattr(model, "peft_config", None)
             and all(quant_linear(layer, value.shape[-1], value.device) for layer in layers))
    if not valid:
        model._h3_fusion_diagnostics.fallback("shared_qkv", "QKV is fused/adapted, or input/projections/hooks require native forward")
        return None
    import torch_npu
    q, scale = torch_npu.npu_dynamic_quant(value.reshape(-1, value.shape[-1]))
    result = tuple(quant_matmul(q, scale, layer, value.shape) for layer in layers)
    model._h3_fusion_diagnostics.hit("shared_qkv", f"shape={tuple(value.shape)}")
    return result

def _h3_rope(model, q, k, c, s, fallback):
    if not _h3_enabled(model, "rope"):
        return fallback(q, k, c, s)
    valid = (tensors_ok(q, k) and q.ndim == 4 and q.shape == k.shape
             and q.shape[-1] == caps.ROPE_HEAD_DIM
             and c.shape == s.shape == (q.shape[1], caps.ROPE_ROTARY_DIM)
             and c.dtype == s.dtype == torch.float32 and c.device == s.device == q.device
             and not c.requires_grad and not s.requires_grad
             and all(t.stride(-1) == 1 and 128 <= t.stride(2) <= 2147483647
                     and all(stride >= 0 for stride in t.stride()) for t in (q, k))
             and all(t.stride(-1) == 1 and t.stride(0) >= 96 for t in (c, s)))
    if valid:
        kernel = model._h3_fusion_kernels.get("rope", q)
        if kernel is not None:
            result = kernel.partial_rope_pair(q, k, c, s, tile_heads=32, blocks=caps.VECTOR_BLOCKS)
            model._h3_fusion_diagnostics.hit("rope", f"shape={tuple(q.shape)}")
            return result
    else:
        model._h3_fusion_diagnostics.fallback("rope", "requires BF16 [...,128] Q/K, FP32 [S,96] frequencies and compatible strides")
    return fallback(q, k, c, s)

def _h3_project_output(layer, value):
    if isinstance(value, QuantizedRows):
        return quant_matmul(value.q, value.scale, layer, value.shape)
    return layer(value)

def _h3_phaa(attn, query, key, value, **kwargs):
    model = _h3_model(attn)
    if (query.device.type not in ("npu", "privateuseone") or kwargs.get("fp8_transport")
            or not any(_h3_enabled(model, feature) for feature in ("phaa_layout", "merge_quant", "laser_preprocess"))):
        return run_phaa_attention(query, key, value, **kwargs)
    pad = get_pad(kwargs.get("pad_key", "pad"))
    supported = (tensors_ok(query, key, value) and query.ndim == 3
                 and query.shape == key.shape == value.shape and query.shape[0] == 1
                 and query.shape[1] > 0 and query.shape[2] == caps.PHAA_HEADS * caps.PHAA_HEAD_DIM
                 and kwargs.get("sp_size") == caps.PHAA_SP and kwargs.get("heads") == caps.PHAA_HEADS
                 and kwargs.get("phaa_split_num") == caps.PHAA_CHUNKS
                 and kwargs.get("head_dim") == caps.PHAA_HEAD_DIM
                 and all(kwargs.get(name) is None for name in ("rope_fn", "extra_attn_fn", "k_ref", "v_ref"))
                 and kwargs.get("num_ref_global", 0) == 0 and kwargs.get("attention_mask") is None)
    layout_kernel = merge_kernel = None
    if _h3_enabled(model, "phaa_layout"):
        valid = supported and pad in (0, 1) and all(
            t.stride(-1) == 1 and 7168 <= t.stride(1) < (1 << 30) for t in (query, key, value))
        if valid:
            layout_kernel = model._h3_fusion_kernels.get("phaa_layout", query)
        else:
            model._h3_fusion_diagnostics.fallback("phaa_layout", "layout kernel requires B1/SP4/H56/PHAA7/D128, padding 0..1 and compatible strides")
    if _h3_enabled(model, "merge_quant"):
        consumer = attn.to_out[0]
        if (supported and query.shape[1] <= caps.MERGE_MAX_SEQUENCE
                and quant_linear(consumer, query.shape[-1], query.device)):
            merge_kernel = model._h3_fusion_kernels.get("merge_quant", query)
        else:
            model._h3_fusion_diagnostics.fallback("merge_quant", "merge kernel geometry or original INT8 output consumer/hooks incompatible")
    for feature, selected in (("phaa_layout", layout_kernel), ("merge_quant", merge_kernel)):
        if selected is not None:
            model._h3_fusion_diagnostics.hit(feature, f"shape={tuple(query.shape)}")
    preprocess = None
    if _h3_enabled(model, "laser_preprocess"):
        if kwargs.get("attention_fn") is None:
            preprocess = partial(attention_with_preprocess, kernel_registry=model._h3_fusion_kernels,
                                 diagnostics=model._h3_fusion_diagnostics)
        else:
            model._h3_fusion_diagnostics.fallback("laser_preprocess", "model-local attention is selected")
    if layout_kernel is None and merge_kernel is None and preprocess is None:
        return run_phaa_attention(query, key, value, **kwargs)
    ops = make_fused_layout_ops(
        query, heads=kwargs["heads"], head_dim=kwargs["head_dim"],
        sp_size=kwargs["sp_size"], splits=kwargs["phaa_split_num"], padding=pad,
        layout_kernel=layout_kernel, merge_kernel=merge_kernel,
        attention=preprocess,
    )
    return run_phaa_attention(query, key, value, local_ops=ops, **kwargs)

def configure_h3_fused_ops(self, features):
    """Set capabilities under the same lock as inference; never replace methods."""
    features = frozenset(features)
    if not hasattr(self, "_h3_request_lock"):
        self._h3_request_lock = threading.Lock()
    if not self._h3_request_lock.acquire(blocking=False):
        raise RuntimeError("Finish the inference request before changing H3 capabilities")
    try:
        if getattr(self, "_h3_request_active", False):
            raise RuntimeError("Finish the inference request before changing H3 capabilities")
        if (getattr(self, "_h3_fused_ops", None) == features
                and hasattr(self, "_h3_fusion_diagnostics")):
            self._h3_text_cache.clear()
            return
        self._h3_fused_ops = features
        self._h3_fusion_diagnostics = Diagnostics(prefix="H3_OPT")
        self._h3_fusion_kernels = Kernels(self._h3_fusion_diagnostics)
        self._h3_text_cache = {}
        self._h3_request_active = False
        parent = weakref.ref(self)
        for block in self.transformer_blocks:
            block._h3_transformer_ref = parent
            if hasattr(block, "attn"):
                block.attn._h3_transformer_ref = parent
        logger.info("[H3_OPT] requested=%s", ",".join(sorted(features)) or "none")
    finally:
        self._h3_request_lock.release()


def _h3_model(module):
    reference = getattr(module, "_h3_transformer_ref", None)
    return reference() if reference is not None else None


def _h3_enabled(model, feature):
    return (model is not None and getattr(model, "_h3_request_active", False)
            and feature in getattr(model, "_h3_fused_ops", ()))


@contextmanager
def h3_inference_context(self):
    """Bound text reuse and diagnostics to one complete inference request."""
    lock = getattr(self, "_h3_request_lock", None)
    if lock is None:
        yield
        return
    if not lock.acquire(blocking=False):
        raise RuntimeError("The H3 model is already serving an inference request")
    features = self._h3_fused_ops
    if not features:
        lock.release()
        yield
        return
    diagnostics = self._h3_fusion_diagnostics
    failed = False
    try:
        diagnostics.reset()
        diagnostics.in_request = True
        self._h3_text_cache.clear()
        self._h3_request_active = True
        try:
            with torch.no_grad():
                yield
        except BaseException as error:
            failed = True
            logger.error("[H3_OPT] request_failed=%s detail=%s", type(error).__name__, error)
            raise
        finally:
            self._h3_text_cache.clear()
            self._h3_request_active = False
            for feature in features:
                if not failed and not diagnostics.hits[feature] and not any(f == feature for f, _ in diagnostics.fallbacks):
                    diagnostics.fallback(feature, "not reached or no reusable input in this request")
            diagnostics.summary("request_error" if failed else "request_end")
    finally:
        diagnostics.in_request = False
        self._h3_request_active = False
        self._h3_request_lock.release()


def _h3_cached_text_call(self, name, value, *, dtype=None):
    """Cache only the explicitly named H3 projection/refiner calls, not forwards."""
    layer = getattr(self, name)
    def compute():
        return layer(value if dtype is None else value.to(dtype))

    if not _h3_enabled(self, "text_cache"):
        return compute()
    roots = [self.context_embedder, self.token_refiner]
    modules = [module for root in roots for module in root.modules()]
    parameters = [p for root in roots for p in (*root.parameters(), *root.buffers())]
    slot = self._h3_text_cache.setdefault(name, {})
    diagnostics = self._h3_fusion_diagnostics
    eligible = (not torch.is_grad_enabled() and isinstance(value, torch.Tensor)
                and not value.requires_grad and not getattr(self, "peft_config", None)
                and all(not module.training and no_hooks(module) for module in modules))
    if not eligible:
        slot.clear()
        diagnostics.fallback("text_cache", "training, gradients, hooks or adapters disable request caching")
        return compute()
    input_key = signature(value)
    versions = tuple(signature(p) for p in parameters)
    if input_key is None or any(item is None for item in versions):
        slot.clear()
        diagnostics.fallback("text_cache", "input/parameter mutation version unavailable")
        return compute()
    try:
        autocast = (torch.is_autocast_enabled(value.device.type), torch.get_autocast_dtype(value.device.type))
    except (RuntimeError, TypeError):
        slot.clear()
        diagnostics.fallback("text_cache", "autocast state unavailable")
        return compute()
    key = (input_key, dtype, versions, autocast, tuple((id(module), module.training) for module in modules))
    if slot.get("key") == key and signature(slot["value"]) == slot["output_key"]:
        diagnostics.hit("text_cache", f"module={name} shape={tuple(value.shape)}")
        return slot["value"]
    output = compute()
    slot.clear()
    if isinstance(output, torch.Tensor) and signature(output) is not None:
        slot.update(key=key, input=value, value=output, output_key=signature(output))
    return output


def project_text(model, text):
    """Share the text projection path between full-sequence and local input."""
    if getattr(model, '_h3_precompute_conditioning', False):
        from .h3_conditioning import project_conditioning

        return project_conditioning(model, text)
    embeds = _h3_cached_text_call(model, 'context_embedder', text, dtype=model.context_embedder.weight.dtype)
    return _h3_cached_text_call(model, 'token_refiner', embeds)


def _h3_ffn(model, ff, hidden_states):
    if not _h3_enabled(model, "swiglu"):
        return ff(hidden_states)
    if (ff.training or not tensors_ok(hidden_states) or hidden_states.ndim != 3
            or not _h3_ffn_layout(ff)):
        model._h3_fusion_diagnostics.fallback("swiglu", "FFN layout, call, dtype, gradients or hooks require native forward")
        return ff(hidden_states)
    return _h3_from_projected(model, ff, ff.net[0].proj(hidden_states))


def h3_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    model = _h3_model(self)
    features = ("gate_residual", "norm_quant", "swiglu")
    if not any(_h3_enabled(model, feature) for feature in features):
        return _original_h3_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    if (type(self) is not MiniMaxH3TransformerBlock or self.training or not no_hooks(self)
            or not tensors_ok(hidden_states) or hidden_states.ndim != 3 or getattr(model, "peft_config", None)):
        for feature in features:
            if _h3_enabled(model, feature):
                model._h3_fusion_diagnostics.fallback(feature, "block dtype/device/grad/training/hooks/adapter requires native forward")
        return _original_h3_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    shift, scale, gate, shiftf, scalef, gatef = self.adaln_proj(temb)
    n = self.norm1(hidden_states)
    n = n * (1.0 + scale.index_select(0, adaln_indices)) + shift.index_select(0, adaln_indices)
    branch = self.attn(n, rotary_emb, attention_mask)
    residual = _h3_residual(model, hidden_states, branch, gate, adaln_indices)
    branch = _h3_feed_forward(model, self, residual, scalef, shiftf, adaln_indices)
    return _h3_residual(model, residual, branch, gatef, adaln_indices)
