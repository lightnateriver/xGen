"""H3 model-specific Transformer adaptation."""


# ------------------------------------------------------------------------
# transformer.h3_npu_fusion
# ------------------------------------------------------------------------

# Ascend fusion dispatch for MiniMax-H3 blocks and request-local text reuse.
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


# ------------------------------------------------------------------------
# transformer.h3_adaln
# ------------------------------------------------------------------------

# H3 model-state invariants for table-only AdaLN execution.

import fnmatch


def assert_weights_absent(model):
    from x_base.adaln_cache import ADALN_DELETE_PATTERNS

    if any(
        any(fnmatch.fnmatch(name, pattern) for pattern in ADALN_DELETE_PATTERNS)
        for name, _ in model.named_parameters()
    ):
        raise ValueError(
            "AdaLN projection/time-embedding weights remain in table-only mode"
        )


# ------------------------------------------------------------------------
# transformer.h3_attention
# ------------------------------------------------------------------------

# Model-local optional operators; original XGEN SP/PHAA owns communication.
import inspect
import json

from pathlib import Path
import time


import torch.distributed as dist

from x_base.infinite_vram.immutable_weights import CpuWeights
from x_base.utils.stage_metrics import Telemetry
from x_diffusers.framework.h3_config import H3AttentionConfig
from x_diffusers.framework.h3_config import H3AttentionRequest




class BaseAttentionSession:
    def __init__(self, model, config: H3AttentionConfig):
        from h3_ascend.context import VideoLayout, HeadShard
        self.layout_type, self.head_type = VideoLayout, HeadShard
        self.model, self.backend_key = model, config.key
        self.backend = config.backend
        self.sparsity = config.sparsity
        self.metrics = Telemetry()
        self.programs = {}
        self.bundle, self.weights = None, None
        self.calls, self.failed = 0, False
        self.permutation = None
        self.signature = inspect.signature(model.forward)
        self.layout_key = None
        self.request = None
        self.predictor_loads = 0
        started = time.perf_counter()
        if self.backend == 'veda':
            from h3_ascend._vendor.miowtion.veda import bundle
            self.bundle = bundle.load(config.predictor, device='cpu')
            self.bundle.predictor.requires_grad_(False).eval()
            self.weights = CpuWeights(self.bundle.predictor, 'veda.predictor', self.metrics)
            self.predictor_loads = 1
        self.initialization_seconds = time.perf_counter()-started
        self.predictor_identity = self.weights.fingerprint() if self.weights is not None else None
        model.register_forward_pre_hook(self.prepare, with_kwargs=True)
        model.register_forward_hook(self.finished, with_kwargs=True, always_call=True)
        for layer, block in enumerate(model.transformer_blocks):
            if block.attn.heads != 56:
                raise ValueError('H3 sparse operators require the original 56-head model')
            block.attn._h3_attention_backend = self.operator(block.attn, layer)
        logger.warning('H3_ATTENTION_READY experimental=true backend=%s predictor_loads=%s',
                       self.backend, self.predictor_loads)

    def begin(self, request: H3AttentionRequest):
        if request.config.key != self.backend_key:
            raise ValueError('Restart the worker to change H3 attention settings')
        if self.failed:
            raise RuntimeError('H3 attention worker failed; restart required')
        if self.calls:
            raise RuntimeError('Previous H3 request did not complete the configured steps')
        if type(request.num_inference_steps) is not int or request.num_inference_steps < 1:
            raise ValueError("H3 attention requires a positive evaluation count")
        self.request = request
        self.expected_steps = request.num_inference_steps
        self.metrics.reset()
        self.started = time.perf_counter()
        self.operator_calls = 0
        self.routed_heads = set()

    def prepare(self, module, positional, kwargs):
        if self.request is None or self.failed:
            raise RuntimeError('Begin a validated H3 attention request before forward')
        values = self.signature.bind(*positional, **kwargs).arguments
        if self.calls == 0:
            text, audio, video = (values[k] for k in ('text_indices','audio_indices','video_indices'))
            used = values['position_ids'].shape[0]
            i2va = self.request.task_type == 'fl2va'
            grid = (72 if i2va else 37, 24, 42)
            condition_rows = 24 * 42 if i2va else 0
            if len(video) != grid[0] * grid[1] * grid[2] + condition_rows:
                raise ValueError('Unexpected 768P Base latent grid or first-frame token count')
            permutation = torch.cat((text, audio, video)).long()
            if len(permutation) != used or not torch.equal(permutation.sort().values,
                                                          torch.arange(used, device=permutation.device)):
                raise ValueError('H3 modalities must cover every real token exactly once')
            shape = (len(text), len(audio), len(video))
            if self.layout_key != shape:
                self.programs.clear()
                self.layout_key = shape
            self.permutation = permutation
            # FL2VA puts the first frame before the generated rows in video_indices.
            # Keep it dense. The upstream 345-frame profile supplies tuning only.
            segments = (len(text), len(audio), condition_rows) if i2va else (len(text), len(audio))
            self.layout = self.layout_type(grid, sum(segments), len(text), segments, frames=345 if i2va else 124)
            if self.weights is not None:
                self.weights.onload()
        self.calls += 1

    def operator(self, attention, layer):
        def run(q, k, v, head_offset=0, attn_mask=None):
            if attn_mask is not None:
                raise ValueError('Sparse H3 operators reject arbitrary masks; SP tail padding is separate')
            if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
                raise ValueError('H3 attention requires equal batch-one BNSD Q/K/V with D=128')
            if any(x.dtype != torch.bfloat16 for x in (q,k,v)):
                raise ValueError('Experimental H3 sparse operators require BF16 Q/K/V')
            parallel = getattr(attention, 'parallel_manager', None)
            rank = dist.get_rank(parallel.sp_group) if parallel is not None else 0
            size = parallel.sp_size if parallel is not None else 1
            start = rank*(56//size)+head_offset
            heads = self.head_type(start, q.shape[1])
            program_key = (start, q.shape[1])
            first_use = program_key not in self.programs
            if first_use:
                with self.metrics.host_span('h3_attention.program_initialize'):
                    if self.backend == 'spark':
                        from h3_ascend.backends.spark import SparkAttention
                        program = SparkAttention(self.layout)
                    else:
                        from x_base.operator.h3_attention.veda import VedaAttention
                        program = VedaAttention(self.layout,heads,
                            predictor_path=self.request.config.predictor, sparsity=self.sparsity,
                            shared_bundle=self.bundle,device=str(q.device))
                    self.programs[program_key] = program
            self.routed_heads.update(range(start, start+q.shape[1]))
            with self.metrics.span('h3_attention.pack',sync=False):
                packed = [x.transpose(1,2)[0].index_select(0,self.permutation).contiguous() for x in (q,k,v)]
            with self.metrics.span('h3_attention.'+self.backend,sync=False,layer=layer,head_start=start,
                                   heads=q.shape[1],step=self.calls-1,first_use=first_use):
                result = self.programs[program_key](*packed,layer=layer)
            if result.shape != packed[0].shape or result.dtype != q.dtype:
                raise ValueError('Sparse H3 operator changed output shape or dtype')
            with self.metrics.span('h3_attention.unpack',sync=False):
                output = torch.zeros_like(q)
                output.index_copy_(2,self.permutation,result.transpose(0,1)[None])
            self.operator_calls += 1
            return output
        return run

    def finished(self, module, positional, kwargs, output):
        if output is None:
            self.failed = True
            if self.weights is not None:
                self.weights.offload()
            return
        if self.calls == self.expected_steps:
            parallel = getattr(self.model,'parallel_manager',None)
            rank_in_group = dist.get_rank(parallel.sp_group) if parallel is not None else 0
            group_size = parallel.sp_size if parallel is not None else 1
            expected_heads = set(range(rank_in_group*(56//group_size),(rank_in_group+1)*(56//group_size)))
            expected_calls = self.expected_steps*len(self.model.transformer_blocks)*max(1,self.request.phaa_num)
            if self.routed_heads != expected_heads or self.operator_calls != expected_calls:
                self.failed = True
                raise RuntimeError('Unexpected sparse attention head coverage or operator call count')
            if self.weights is not None:
                self.weights.offload()
                if self.weights.fingerprint() != self.predictor_identity:
                    self.failed = True
                    raise RuntimeError('Veda predictor CPU master identity changed')
            record = dict(experimental=True,backend=self.backend,actual_base_dit=self.expected_steps,
                          sparsity=self.sparsity if self.backend == 'veda' else None,
                          operator_calls=self.operator_calls,global_heads=sorted(self.routed_heads),
                          predictor_loads=self.predictor_loads,initialization_seconds=self.initialization_seconds,
                          predictor_identity=self.predictor_identity,
                          base_seconds=time.perf_counter()-self.started,phases=self.metrics.resolve())
            rank = dist.get_rank() if dist.is_initialized() else 0
            path = Path(self.request.save_path).with_suffix(f'.attention.rank{rank}.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(record),encoding='utf-8')
            self.metrics.active = False
            self.calls = 0
            self.request = None
            logger.info('H3_ATTENTION_COMPLETE backend=%s rank=%s operator_calls=%s predictor_loads=%s',
                        self.backend,rank,self.operator_calls,self.predictor_loads)


def install_h3_attention(model, config: H3AttentionConfig):
    if getattr(model,'_h3_attention_session',None) is not None:
        if model._h3_attention_session.backend_key != config.key:
            raise ValueError('Restart to change H3 attention backend')
        return
    model._h3_attention_session = BaseAttentionSession(model,config)


# ------------------------------------------------------------------------
# transformer.h3_conditioning
# ------------------------------------------------------------------------

# Request-scoped H3 condition projection; no cross-request tensors on models.

from contextvars import ContextVar





_CURRENT = ContextVar('h3_request_conditioning', default=None)
_ROPE_TABLE = ContextVar("h3_request_rope_table", default=None)


def _version(value):
    try:
        return value._version
    except RuntimeError:  # Inference-mode tensors have no version counter.
        return None


@contextmanager
def request_conditioning(model):
    if _CURRENT.get() is not None:
        raise RuntimeError('Nested H3 condition precomputation is not supported')
    metrics = dict(projection_calls=0, computations=0, reuses=0,
                   peak_output_bytes=0, released_before_vae=False, cleared=False)
    state = dict(model=model, entries=[], metrics=metrics)
    token = _CURRENT.set(state)
    try:
        yield
    finally:
        state['entries'].clear()
        metrics['cleared'] = True
        model._h3_conditioning_metrics = dict(metrics)  # Scalars only.
        _CURRENT.reset(token)
        logger.info('H3 request condition precompute: %s', metrics)


def project_conditioning(model, source):
    state = _CURRENT.get()
    if state is None or state['model'] is not model:
        raise RuntimeError('H3 condition precompute requires its managed request scope')
    if torch.is_grad_enabled():
        raise ValueError('H3 condition precompute is inference-only')
    metrics = state['metrics']
    metrics['projection_calls'] += 1
    version = _version(source)
    for index, entry in enumerate(state['entries']):
        if entry['source'] is not source:
            continue
        unchanged = (version == entry['version'] if version is not None
                     else torch.equal(source, entry['snapshot']))
        if unchanged:
            metrics['reuses'] += 1
            return entry['output']
        state['entries'].pop(index)
        break
    output = model.token_refiner(model.context_embedder(source.to(model.context_embedder.weight.dtype)))
    # Retain at most positive/negative condition branches. Changed conditions
    # are recomputed; holding the source prevents pointer/id reuse collisions.
    if len(state['entries']) == 2:
        state['entries'].pop(0)
    state['entries'].append(dict(source=source, version=version,
        snapshot=source.clone() if version is None else None, output=output))
    metrics['computations'] += 1
    metrics['peak_output_bytes'] = max(metrics['peak_output_bytes'],
        sum(e['output'].numel()*e['output'].element_size() for e in state['entries']))
    return output


def release_conditioning(model):
    """Release finished DiT conditions before the existing VAE memory trim."""
    state = _CURRENT.get()
    if state is not None and state['model'] is model:
        state['entries'].clear()
        state['metrics']['released_before_vae'] = True


# ------------------------------------------------------------------------
# transformer.h3_local_io
# ------------------------------------------------------------------------

# Pure-Ulysses local packed input/output, sharing the existing H3 block loop.
#
# Text refiner keeps its full text context. Only raw video/audio rows owned by
# this CP rank are projected. Gather final predictions in the original head dtype.
# No per-request tensors are cached on the model.
from dataclasses import dataclass



from x_base import gather_sequence, set_pad


@dataclass(frozen=True)
class LocalSpan:
    start: int
    length: int
    valid: int

    @classmethod
    def for_rank(cls, total, peers, rank):
        if total < 1 or peers < 1 or not 0 <= rank < peers:
            raise ValueError('Invalid H3 local sequence layout')
        length = (total + peers - 1) // peers
        start = rank * length
        return cls(start, length, max(0, min(length, total-start)))

    def slice_padded(self, tensor, dim=0, value=0):
        local = tensor.narrow(dim, min(self.start, tensor.shape[dim]), self.valid)
        if self.valid == self.length:
            return local.contiguous()
        shape = list(local.shape)
        shape[dim] = self.length-self.valid
        return torch.cat((local, tensor.new_full(shape, value)), dim=dim)

    def indices(self, packed_indices):
        source = ((packed_indices >= self.start) & (packed_indices < self.start+self.valid)).nonzero().flatten()
        return source, packed_indices.index_select(0, source)-self.start


def prepare_local(model, video, audio, text, positions, text_indices, video_indices, audio_indices):
    manager = model.parallel_manager
    peers = manager.sp_size
    rank = manager.sp_rank if peers > 1 else 0
    span = LocalSpan.for_rank(len(positions), peers, rank)
    for inputs, indices in ((video,video_indices),(audio,audio_indices),(text,text_indices)):
        if indices.ndim != 1 or len(indices) != inputs.shape[1]:
            raise ValueError('H3 packed modality indices must match the input rows')
        # Filtering to this CP span must not silently discard invalid global
        # indices. On CUDA the assertion does not synchronize valid requests.
        torch._assert_async(((indices >= 0) & (indices < len(positions))).all(),
                            'H3 packed modality index is outside the sequence')
    if peers > 1:
        set_pad('mmh3_seq', len(positions), manager.sp_group)
    # The text refiner is a separate full-context transformer, not token-wise.

    embeds = project_text(model, text)
    hidden = embeds.new_zeros((embeds.shape[0], span.length, embeds.shape[-1]))
    source, target = span.indices(text_indices)
    hidden.index_copy_(1, target, embeds.index_select(1, source))
    del embeds
    for inputs, indices, projection in ((video, video_indices, model.proj_in),
                                        (audio, audio_indices, model.audio_proj_in)):
        source, target = span.indices(indices)
        local = inputs.index_select(1, source).to(projection.weight.dtype)
        hidden.index_copy_(1, target, projection(local).to(hidden.dtype))
    # SP keeps local RoPE; shared TP columns consume the full real sequence.
    shared = getattr(model, '_h3_shared_sequence_exchange', None)
    if shared is not None:
        shared.set_tokens(len(positions))
        rotary = _compute_rope(model, positions)
    else:
        rotary = _compute_rope(model, span.slice_padded(positions))
    return hidden, rotary, span


def gather_local_output(model, hidden, temb, span, timestep_indices, video_indices, audio_indices):
    local_indices = span.slice_padded(timestep_indices)
    hidden = model.norm_out(hidden, temb, local_indices).to(model.proj_out.weight.dtype)
    video = model.proj_out(hidden)
    audio = model.audio_proj_out(hidden.to(model.audio_proj_out.weight.dtype))
    if getattr(model, '_h3_cuda_precision', 'none') in ('fp8-bf16', 'fp8-sage') and (video.dtype != torch.float32 or audio.dtype != torch.float32):
        raise ValueError('H3 TP/CP final prediction heads must remain FP32')
    width = video.shape[-1]
    prediction = torch.cat((video, audio), dim=-1)
    manager = model.parallel_manager
    if manager.sp_size > 1:
        pad = span.length*manager.sp_size - len(timestep_indices)
        prediction = gather_sequence(prediction, manager.sp_group, dim=1, pad=pad)
    return (prediction[..., :width].index_select(1, video_indices),
            prediction[..., width:].index_select(1, audio_indices))


# ------------------------------------------------------------------------
# transformer.h3.parallel
# ------------------------------------------------------------------------

# H3-only TP adapter; reuse xGen mesh, Linear, FP8 GEMM and PHAA.
#
# Weights are sliced on CPU before the existing resident-device transfer.
# Fused QKV and SwiGLU halves are sliced separately, never as one flat matrix.


from functools import wraps



from torch import nn





def initialize(model, config):
    from x_base.sequence_parallelism.mesh import ParallelManager

    if config.mode not in ("tp_cp", "tp_cp_shared", "tp_sp"):
        return None
    if hasattr(model, "_h3_parallel_config"):
        raise ValueError(
            "H3 TP/CP layout is immutable; restart the worker to change it"
        )
    heads = model.config.num_attention_heads
    if heads % (
        max(config.tp, config.cp)
        if config.mode in ("tp_cp_shared", "tp_sp")
        else config.tp * config.cp
    ):
        raise ValueError("H3 attention heads must divide TP * CP")
    if config.mode == "tp_cp_shared":
        from x_base.tensor_parallelism.shared_sequence import SharedSequenceExchange

        manager = ParallelManager(1, 1, config.cp, 1, register_tp_global=False)
        tp_manager = ParallelManager(1, 1, 1, config.tp, register_tp_global=False)
        model._h3_shared_tp_manager = tp_manager
        model._h3_shared_sequence_exchange = SharedSequenceExchange(
            manager.sp_group,
            tp_manager.tp_group,
            config.tp_comm,
            config.tp_chunk_elements,
        )
    elif config.mode == "tp_sp":
        from x_base.tensor_parallelism.pair_sequence import PairSequenceExchange

        manager = ParallelManager(1, 1, config.cp, 1, register_tp_global=False)
        model._h3_pair_manager = ParallelManager(1, 1, 4, 2, register_tp_global=False)
        model._h3_pair_exchange = PairSequenceExchange(
            model._h3_pair_manager.tp_group, config.tp_comm, config.tp_chunk_elements
        )
    else:
        manager = ParallelManager(1, 1, config.cp, config.tp, register_tp_global=False)
    model.set_parallel_manager(manager)
    model._h3_parallel_config = config
    if config.warmup_mib:
        from x_base.tensor_parallelism.warmup import warmup_groups

        model._h3_parallel_warmup = warmup_groups(
            manager,
            buffer_mib=config.warmup_mib,
            tp_wire=config.tp_comm,
            cp_wire=config.cp_comm,
            shared_layout=config.mode in ("tp_sp", "tp_cp_shared"),
        )
        if config.mode == "tp_cp_shared":
            model._h3_shared_tp_warmup = warmup_groups(
                tp_manager,
                buffer_mib=config.warmup_mib,
                tp_wire=config.tp_comm,
                cp_wire=config.cp_comm,
                shared_layout=config.mode in ("tp_sp", "tp_cp_shared"),
            )
        elif config.mode == "tp_sp":
            model._h3_pair_warmup = warmup_groups(
                model._h3_pair_manager,
                buffer_mib=config.warmup_mib,
                tp_wire=config.tp_comm,
                cp_wire=config.cp_comm,
                shared_layout=config.mode in ("tp_sp", "tp_cp_shared"),
            )
    for block in model.transformer_blocks:
        block.attn._h3_tp_cp = True
    projection_manager = getattr(
        model, "_h3_pair_manager", getattr(model, "_h3_shared_tp_manager", manager)
    )
    logger.info(
        "Experimental H3 parallelism=%s TP=%d CP(Ulysses)=%d TP_wire=%s CP_wire=%s TP_packet_elements=%d ranks_tp=%s ranks_cp=%s",
        config.mode,
        config.tp,
        config.cp,
        config.tp_comm,
        config.cp_comm,
        config.tp_chunk_elements,
        dist.get_process_group_ranks(projection_manager.tp_group)
        if config.tp > 1
        else [],
        dist.get_process_group_ranks(manager.sp_group) if config.cp > 1 else [],
    )
    if config.mode == "tp_cp_shared":
        logger.info(
            "H3 shared-rank TP/CP: CP residual rows, TP8 QKV/O/FFN; QKV all-to-all fused into TP head layout; row sums reduce-scatter to CP rows"
        )
    return config


def shard_projection(original, manager, *, column, parts=1, fp8=True, reduction=None):
    from x_base.tensor_parallelism.projections import shard_linear
    from x_base.operator.h3_fp8 import H3FP8TPLinear

    layer = shard_linear(
        original, manager, column=column, parts=parts, reduction=reduction
    )
    if not fp8:
        return layer
    # Preserve the scale of the full source tensor across all projection shards.
    scale = (original.weight.detach().float().abs().amax() / 448).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    weight = layer.weight.detach()
    layer = H3FP8TPLinear(layer)
    layer.weight.copy_(
        (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    layer.weight_scale.copy_(scale)
    if not column:
        layer.activation_group = manager.tp_group
    return layer


def prepare_block(block, manager, config, *, fp8=True):
    from x_base.tensor_parallelism.fp8_reduce import BF16Reduction, FP8Reduction

    if manager.tp_size == 1:
        raise ValueError("TP1 uses the original projection preparation")
    attn = block.attn
    attn._h3_tp_cp = True
    if attn.fused_projections:
        raise ValueError("H3 TP requires original unfused Q/K/V weights")
    projections = [attn.to_q, attn.to_k, attn.to_v]
    if any(p.bias is None for p in projections) and not all(
        p.bias is None for p in projections
    ):
        raise ValueError("H3 QKV biases must all exist or all be absent")
    merged = nn.Linear(
        projections[0].in_features,
        sum(p.out_features for p in projections),
        bias=projections[0].bias is not None,
        dtype=projections[0].weight.dtype,
        device="meta",
    )
    merged.weight = nn.Parameter(
        torch.cat([p.weight.detach() for p in projections]), requires_grad=False
    )
    if projections[0].bias is not None:
        merged.bias = nn.Parameter(
            torch.cat([p.bias.detach() for p in projections]), requires_grad=False
        )
    reduction = (
        FP8Reduction(chunk_elements=config.tp_chunk_elements)
        if config.tp_comm == "fp8"
        else BF16Reduction()
    )
    attn.to_qkv = shard_projection(merged, manager, column=True, parts=3, fp8=fp8)
    attn.to_q = attn.to_k = attn.to_v = None
    attn.fused_projections = True
    attn.heads //= manager.tp_size
    attn.inner_dim //= manager.tp_size
    attn.to_out[0] = shard_projection(
        attn.to_out[0], manager, column=False, fp8=fp8, reduction=reduction
    )
    block.ff.net[0].proj = shard_projection(
        block.ff.net[0].proj, manager, column=True, parts=2, fp8=fp8
    )
    block.ff.net[2] = shard_projection(
        block.ff.net[2], manager, column=False, fp8=fp8, reduction=reduction
    )
    return reduction


def attach_pipeline(pipe, *, vae_parallel=False):
    """Called instead of enable_sp only for the opt-in H3 path."""
    from x_base.sequence_parallelism.mesh import ParallelManager

    model = pipe.transformer
    config = model._h3_parallel_config
    if vae_parallel:
        # VAE still spans all devices; it does not inherit DiT's smaller CP group.
        pipe._h3_vae_parallel_manager = (
            model.parallel_manager
            if config.mode == "tp_cp_shared"
            else ParallelManager(
                1, 1, dist.get_world_size(), 1, register_tp_global=False
            )
        )
        model._h3_condition_parallel_manager = pipe._h3_vae_parallel_manager
    logger.info(
        "H3 local IO enabled; FP32 prediction gather, text TP unchanged, VAE SP=%d",
        dist.get_world_size() if vae_parallel else 1,
    )
    return config


def stage_reference_audio(pipe, device):
    """Reference audio follows the same whole-module staging contract as decode."""
    from x_diffusers.framework.vae.minimax_h3 import offload_audio

    audio = pipe.audio_vae
    original = audio.encode

    @wraps(original)
    def encode(*values, **kwargs):
        audio.to(device)
        try:
            return original(*values, **kwargs)
        finally:
            offload_audio(audio)

    audio.encode = encode


# ------------------------------------------------------------------------
# transformer.h3.pair_parallel
# ------------------------------------------------------------------------

# Optional TP2 QKV/O/FFN projections; retain the existing SP8/PHAA processor.




from x_base.tensor_parallelism.projections import (
    shard_fused_tensor,
    SequenceColumn,
    SequenceRow,
)





def slice_projection(original, exchange, *, column, parts=1):
    from x_base.operator.h3_fp8 import H3FP8Linear

    if (
        not isinstance(original, (nn.Linear, H3FP8Linear))
        or original.weight.device.type != "cpu"
    ):
        raise ValueError(
            "TP2 must slice CPU projections before the resident-device transfer"
        )
    axis = 0 if column else 1
    if original.weight.shape[axis] % (parts * 2):
        raise ValueError(
            "TP2 projection dimensions must divide fused parts and two ranks"
        )
    weight = shard_fused_tensor(
        original.weight, axis=axis, rank=exchange.rank, world_size=2, parts=parts
    )
    bias = original.bias
    if column and bias is not None:
        bias = shard_fused_tensor(
            bias, axis=0, rank=exchange.rank, world_size=2, parts=parts
        )
    if isinstance(original, H3FP8Linear):
        # Preserve the original FP8 bytes and global weight scale. No requantization.
        result = H3FP8Linear.from_packed(
            weight, original.weight_scale, bias if column else None
        )
        if not column:
            result.activation_group = exchange.group
    else:
        result = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=column and bias is not None,
            dtype=weight.dtype,
            device="meta",
        )
        result.weight = nn.Parameter(weight, requires_grad=False)
        if column and bias is not None:
            result.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
    return result


class PairColumn(SequenceColumn):
    def __init__(self, original, exchange, parts, restore_sp):
        projection = slice_projection(original, exchange, column=True, parts=parts)
        super().__init__(projection, exchange, parts=parts, restore_sequence=restore_sp)


class PairRow(SequenceRow):
    def __init__(self, original, exchange, gather_sp):
        projection = slice_projection(original, exchange, column=False)
        super().__init__(
            projection,
            exchange,
            bias=original.bias,
            gather_input=gather_sp,
            partitions=2,
        )


def prepare_pair_block(block, exchange):
    attn = block.attn
    if not attn.fused_projections:
        raise ValueError("TP2/SP8 expects the prepared fused QKV projection")
    attn.to_qkv = PairColumn(attn.to_qkv, exchange, parts=3, restore_sp=True)
    attn.to_out[0] = PairRow(attn.to_out[0], exchange, gather_sp=True)
    block.ff.net[0].proj = PairColumn(
        block.ff.net[0].proj, exchange, parts=2, restore_sp=False
    )
    block.ff.net[2] = PairRow(block.ff.net[2], exchange, gather_sp=False)


def install_pair_parallel(model):
    from x_base.sequence_parallelism.mesh import ParallelManager
    from x_base.tensor_parallelism.pair_sequence import PairSequenceExchange

    if getattr(model, "_h3_pair_installed", False):
        raise ValueError("H3 TP2 projections have already been installed")
    manager = getattr(model, "_h3_pair_manager", None)
    if manager is None:
        manager = ParallelManager(1, 1, 4, 2, register_tp_global=False)
    exchange = getattr(model, "_h3_pair_exchange", None)
    if exchange is None:
        exchange = PairSequenceExchange(manager.tp_group)
    for block in model.transformer_blocks:
        prepare_pair_block(block, exchange)
    model._h3_pair_manager, model._h3_pair_exchange = manager, exchange
    model._h3_pair_installed = True
    logger.info(
        "Experimental H3 linear TP2, SP8 attention unchanged; BF16 pair transport, ranks=%s",
        exchange.metrics["ranks"],
    )


# ------------------------------------------------------------------------
# transformer.h3.shared_parallel
# ------------------------------------------------------------------------

# H3 layerwise TP/CP on the same ranks, using existing xGen components.
#
# Between sublayers the residual stream is CP sequence-sharded. A column TP
# projection consumes gathered rows and produces the exact head-sharded layout
# that Ulysses normally obtains through its QKV all-to-all. Fusing these inverse
# layout changes avoids sending QKV out and immediately sending it back.
# Row projections reduce-scatter their sums back to CP sequence ownership.





from x_base.tensor_parallelism.projections import GatherColumnInput


class SharedTPAttention:
    """Existing dense backend on the CP8/Ulysses internal head partition."""

    def __init__(self, exchange, local_heads, chunks):
        self.exchange, self.local_heads, self.chunks = exchange, local_heads, chunks

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):

        full = self.exchange.gather(hidden_states)
        query, key, value = attn.to_qkv(full).chunk(3, dim=-1)
        del full
        query, key, value = (
            v.unflatten(-1, (self.local_heads, -1)) for v in (query, key, value)
        )
        normalized = None
        fused_rope = getattr(attn, "_h3_fused_partial_rope", False)
        if getattr(attn, "_h3_fused_qk_rms", False):
            from x_base.operator.h3_qk_rms import try_h3_qk_rms

            normalized = try_h3_qk_rms(
                query, key, attn.norm_q, attn.norm_k, rotary_emb if fused_rope else None
            )
        query, key = (
            (attn.norm_q(query), attn.norm_k(key)) if normalized is None else normalized
        )
        if rotary_emb is not None and not (fused_rope and normalized is not None):
            rotated = None
            if fused_rope:
                from x_base.operator.h3_partial_rope import try_h3_partial_rope

                rotated = try_h3_partial_rope(query, key, *rotary_emb)
            query, key = (
                rotated
                if rotated is not None
                else _rope_minimax_h3(query, key, *rotary_emb)
            )
        # Keep the existing PHAA head-chunk cadence (e.g. seven heads -> 4/3),
        # while the TP projections already supply its gathered QKV layout.
        outputs = []
        for q, k, v in zip(
            torch.tensor_split(query, self.chunks, dim=2),
            torch.tensor_split(key, self.chunks, dim=2),
            torch.tensor_split(value, self.chunks, dim=2),
        ):
            outputs.append(
                attn._h3_attention_backend(
                    q.transpose(1, 2).contiguous(),
                    k.transpose(1, 2).contiguous(),
                    v.transpose(1, 2).contiguous(),
                    attn_mask=attention_mask,
                )
            )
        result = torch.cat(outputs, dim=1).transpose(1, 2).flatten(2, 3)
        self.exchange.metrics["attention_calls"] += 1
        return attn.to_out[1](attn.to_out[0](result))


def prepare_shared_block(block, manager, exchange, *, fp8=True, chunks=2):

    attn = block.attn
    if attn.fused_projections or attn.heads % manager.tp_size:
        raise ValueError(
            "Shared TP/CP requires original QKV and divisible attention heads"
        )
    local_heads = attn.heads // manager.tp_size
    if type(chunks) is not int or not 1 <= chunks <= local_heads:
        raise ValueError("Shared TP/CP attention chunk count exceeds local heads")
    originals = (attn.to_q, attn.to_k, attn.to_v)
    biases = [v.bias for v in originals]
    if any(v is None for v in biases) and not all(v is None for v in biases):
        raise ValueError("Shared TP/CP QKV biases must all exist or all be absent")
    merged = nn.Linear(
        originals[0].in_features,
        sum(v.out_features for v in originals),
        bias=biases[0] is not None,
        device="meta",
        dtype=originals[0].weight.dtype,
    )
    merged.weight = nn.Parameter(
        torch.cat([v.weight.detach() for v in originals]), requires_grad=False
    )
    if biases[0] is not None:
        merged.bias = nn.Parameter(torch.cat(biases), requires_grad=False)
    attn.to_qkv = shard_projection(merged, manager, column=True, parts=3, fp8=fp8)
    attn.to_q = attn.to_k = attn.to_v = None
    attn.fused_projections = True
    attn.to_out[0] = shard_projection(
        attn.to_out[0],
        manager,
        column=False,
        fp8=fp8,
        reduction=exchange.reduce_scatter,
    )
    column = shard_projection(
        block.ff.net[0].proj, manager, column=True, parts=2, fp8=fp8
    )
    block.ff.net[0].proj = GatherColumnInput(column, exchange)
    block.ff.net[2] = shard_projection(
        block.ff.net[2],
        manager,
        column=False,
        fp8=fp8,
        reduction=exchange.reduce_scatter,
    )
    attn.set_processor(SharedTPAttention(exchange, local_heads, chunks))
    attn._h3_shared_tp_cp = True
    # The public attention shape remains full-head CP input/output. Only the
    # internal TP GEMM/backend sees the rank's head partition.
    return exchange


# ------------------------------------------------------------------------
# transformer.h3.precision
# ------------------------------------------------------------------------

# H3 FP8 layer mapping and model-state preparation.

from x_diffusers.framework.h3_config import H3ExecutionOptions







def prepare_transformer(pipe, options: H3ExecutionOptions, device):
    from x_base.operator.h3_adaln_owner import H3OwnedAdaLN, check_shared_timestep
    from x_base.operator.h3_dense_cuda import H3DenseCUDA
    from x_base.operator.h3_fp8 import H3FP8Linear

    model = pipe.transformer
    table_mode = bool(getattr(options, "adaln_cache_file", None))
    if table_mode:
        from pathlib import Path

        assert_weights_absent(model)
        model._h3_cuda_adaln_table = True
        model._h3_cuda_adaln_checkpoint = str(Path(options.transformer_path).resolve())
        model._h3_cuda_adaln_load_count = 0
    config = None
    if getattr(options, "h3_parallel_mode", "sp") in ("tp_cp", "tp_cp_shared", "tp_sp"):

        config = initialize(model, options.parallel)
    if options.h3_cuda_precision == "fp8-sage":
        from x_base.operator.h3_sage_cuda import H3SageCUDA

        attention_backend = H3SageCUDA()
    else:
        attention_backend = H3DenseCUDA()
    converted = 0
    for index, block in enumerate(model.transformer_blocks):
        attn = block.attn
        if attn.fused_projections or not all(
            isinstance(getattr(attn, name), nn.Linear)
            for name in ("to_q", "to_k", "to_v")
        ):
            raise ValueError("FP8 recipe expects original unfused BF16 H3 projections")
        projections = (attn.to_q, attn.to_k, attn.to_v)
        biases = [layer.bias for layer in projections]
        if any(b is None for b in biases) and not all(b is None for b in biases):
            raise ValueError("H3 QKV biases must either all exist or all be absent")
        if config is not None and config.mode == "tp_cp_shared":

            prepare_shared_block(
                block,
                model._h3_shared_tp_manager,
                model._h3_shared_sequence_exchange,
                chunks=options.phaa_num,
            )
        elif config is not None and config.mode != "tp_sp" and config.tp > 1:

            prepare_block(block, model.parallel_manager, config)
        else:
            attn.to_qkv = H3FP8Linear(
                torch.cat([layer.weight.detach() for layer in projections]),
                torch.cat(biases) if biases[0] is not None else None,
            )
            attn.to_q = attn.to_k = attn.to_v = None
            attn.fused_projections = True
            targets = (
                (attn.to_out, "0"),
                (block.ff.net[0], "proj"),
                (block.ff.net, "2"),
            )
            for parent, name in targets:
                original = getattr(parent, name)
                setattr(parent, name, H3FP8Linear(original.weight, original.bias))
        converted += 4
        if not table_mode:
            block.adaln_proj = H3OwnedAdaLN(block.adaln_proj, index, exact_cache=True)
        attn._h3_attention_backend = attention_backend
    if not table_mode:
        model.time_embedder.register_forward_hook(check_shared_timestep)
    # These fusions preserve the ordinary BF16/FP8 Linear interfaces. Packed FP4 FFN fusions do not.

    install_dit_fusions(model, options.dit_fusions)
    if options.h3_cuda_dit_fusions:
        _install_extended_dit_fusions(model, device)
    resident = len(model.transformer_blocks)
    expected_blocks = model.config.num_layers if config is not None else 50
    if resident != expected_blocks or resident < 1:
        raise ValueError(
            f"{options.h3_cuda_precision} expects the complete 50-block H3 Base checkpoint"
        )
    input_refiner = getattr(options, "h3_cuda_input_refiner", "none")
    if input_refiner != "none":

        install_input_refiner(
            model,
            input_refiner,
            sp=getattr(options, "h3_cuda_input_refiner_sp", None),
            attention=getattr(options, "h3_cuda_input_refiner_attention", None),
            ffn=getattr(options, "h3_cuda_input_refiner_ffn", None),
        )
    if getattr(options, "h3_cuda_precompute_conditioning", False):
        model._h3_precompute_conditioning = True
    if config is not None and config.mode == "tp_sp":

        install_pair_parallel(model)
    for name in ("h3_cuda_local_input", "h3_cuda_local_output"):
        value = getattr(options, name, None)
        setattr(model, "_" + name, options.local_io_default if value is None else value)
    if model._h3_cuda_local_input:
        for block in model.transformer_blocks:
            block.attn._h3_tp_cp = True
    model.to(device)
    if getattr(options, "h3_cuda_refiner_ff_chunk_size", 0):

        install_ff_chunking(model, options.h3_cuda_refiner_ff_chunk_size)
    model._h3_cuda_precision = options.h3_cuda_precision
    if options.h3_cuda_precision == "fp8-sage":
        model._h3_sage_cuda = attention_backend
    else:
        model._h3_dense_cuda = attention_backend
    model._h3_cuda_resident_blocks = resident
    if getattr(options, "h3_cuda_local_output", False):
        model._h3_cuda_local_output = True
        logger.info(
            "Experimental H3 SP local output enabled: FP32 prediction gather, original block loop unchanged"
        )
    logger.info(
        "H3 CUDA precision=%s FP8_main_projections=%d resident_blocks=%d AdaLN=%s GEMM=torch._scaled_mm",
        options.h3_cuda_precision,
        converted,
        resident,
        "BF16 table, weights absent" if table_mode else "rank-owner exact_cache=True",
    )


def prepare_text_tp(pipe):
    from x_base.tensor_parallelism.layers import ColumnParallelLinear, RowParallelLinear
    from x_base.operator.h3_fp8 import H3FP8TPLinear

    targets = [
        (parent, name, child)
        for parent in pipe.text_encoder.modules()
        for name, child in parent.named_children()
        if isinstance(child, (ColumnParallelLinear, RowParallelLinear))
    ]
    for parent, name, child in targets:
        setattr(parent, name, H3FP8TPLinear(child))
    if not targets:
        raise ValueError("No existing text TP projections found for the H3 FP8 profile")
    logger.info("H3 CUDA precision FP8 text TP projections=%d", len(targets))


# ------------------------------------------------------------------------
# transformer.h3.input_refiner
# ------------------------------------------------------------------------

# Opt-in H3 input Token Refiner Ulysses/cuDNN, reusing xGen SP and FP8 ops.








PAD_KEY = "h3_input_refiner"
PROFILES = ("none", "cudnn-sp", "cudnn-sp-fp8")


class RefinerSPAttention:
    """BF16 full-context attention with heads distributed over the SP group."""

    def __init__(self, group, backend, metrics):
        self.group, self.backend, self.metrics = group, backend, metrics

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from x_base import all_to_all_before_attn, all_to_all_after_attn

        if (
            rotary_emb is not None
            or attention_mask is not None
            or attn.fused_projections
        ):
            raise ValueError(
                "H3 input refiner SP expects unfused QKV, no RoPE and no mask"
            )
        query, key, value = (
            layer(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
            for layer in (attn.to_q, attn.to_k, attn.to_v)
        )
        query, key = attn.norm_q(query), attn.norm_k(key)
        if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
            raise ValueError("H3 input refiner attention requires BF16 QKV")
        self.metrics["attention_calls"] += 1
        if self.group is not None:
            self.metrics["logical_sp_payload_bytes"] += (
                4 * query.numel() * query.element_size()
            )
        # Native helper removes tail padding BEFORE attention, so padded tokens
        # never enter softmax. Its inverse restores padding before the exchange.
        if self.group is not None:
            query, key, value = (
                all_to_all_before_attn(tensor, self.group, pad_key=PAD_KEY)
                for tensor in (query, key, value)
            )
        query, key, value = (
            tensor.transpose(1, 2).contiguous() for tensor in (query, key, value)
        )
        if self.backend is not None:
            output = self.backend(query, key, value)
        else:
            # Preserve the original runtime's attention selection (for example
            # NPU Laser); SP changes the layout, not the selected operator.
            from x_base import attention_manager

            output = attention_manager.attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        output = output.transpose(1, 2).contiguous()
        if self.group is not None:
            output = all_to_all_after_attn(output, self.group, pad_key=PAD_KEY)
        output = output.flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](output))


def convert_ffn(refiner, group=None):
    """Keep BF16 module interfaces; use global activation amax across SP ranks."""
    from x_base.operator.h3_fp8 import H3FP8Linear

    converted = 0
    for block in refiner.refiner_blocks:
        for parent, name in ((block.ff.net[0], "proj"), (block.ff.net, "2")):
            original = getattr(parent, name)
            if not isinstance(original, torch.nn.Linear):
                raise ValueError(
                    "H3 input refiner FP8 expects original FFN Linear modules"
                )
            packed = H3FP8Linear(original.weight, original.bias)
            packed.activation_group = group
            setattr(parent, name, packed)
            converted += 1
    return converted


def install_input_refiner(model, profile, *, sp=None, attention=None, ffn=None):
    from x_base import gather_sequence, set_pad, split_sequence

    if profile not in (*PROFILES[1:], "custom"):
        raise ValueError("Unknown H3 input refiner profile")
    sp = True if sp is None else sp
    attention = "cudnn" if attention is None else attention
    ffn = ("fp8" if profile == "cudnn-sp-fp8" else "bf16") if ffn is None else ffn
    refiner = model.token_refiner
    if hasattr(refiner, "_h3_sp_metrics"):
        raise ValueError("H3 input refiner SP is already installed")
    # CPU weight preparation precedes enable_parallel() in the normal loader.
    # Bind the existing SP group on the first forward, after xGen creates it.
    world = dist.get_world_size()
    if world not in (2, 4, 8) or (
        sp and any(block.attn.heads % world for block in refiner.refiner_blocks)
    ):
        raise ValueError(
            "H3 input refiner SP requires 2/4/8 ranks and divisible attention heads"
        )
    if any(p.dtype != torch.bfloat16 for p in refiner.parameters()):
        raise ValueError("H3 input refiner SP expects original BF16 parameters")
    metrics = dict(
        profile=profile,
        sp_size=world if sp else 1,
        communication_dtype="bfloat16" if sp else "none",
        attention_backend=attention,
        ffn_precision=ffn,
        calls=0,
        completed=0,
        attention_calls=0,
        max_input_tokens=0,
        local_tokens=0,
        last_input_tokens=0,
        last_padding=0,
        fp8_ffn_projections=0,
        logical_sp_payload_bytes=0,
    )
    if ffn == "fp8":
        metrics["fp8_ffn_projections"] = convert_ffn(refiner)
    backend = None
    if attention == "cudnn":
        from x_base.operator.h3_dense_cuda import H3DenseCUDA

        backend = H3DenseCUDA()
    processors = []
    for block in refiner.refiner_blocks:
        processor = RefinerSPAttention(None, backend, metrics)
        block.attn.set_processor(processor)
        processors.append(processor)
    original = refiner.forward
    group = None
    bound = False

    @wraps(original)
    def forward(hidden_states):
        nonlocal group, bound
        if torch.is_grad_enabled():
            raise ValueError("H3 input refiner SP is inference-only")
        if (
            hidden_states.device.type not in ("cuda", "npu")
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 3
        ):
            raise ValueError(
                "H3 input refiner SP expects CUDA/NPU BF16 batch/token/channel inputs"
            )
        if hidden_states.device.type != "cuda" and (
            attention == "cudnn" or ffn == "fp8"
        ):
            raise ValueError("cuDNN/FP8 refiner operators require CUDA")
        tokens = hidden_states.shape[1]
        if tokens < 1:
            raise ValueError("H3 input refiner cannot have an empty context")
        if sp and not bound:
            manager = getattr(
                model,
                "_h3_condition_parallel_manager",
                getattr(model, "parallel_manager", None),
            )
            if (
                manager is None
                or manager.sp_group is None
                or dist.get_world_size(manager.sp_group) != world
                or getattr(manager, "tp_size", 1) != 1
            ):
                raise ValueError(
                    f"H3 input refiner requires an initialized SP{world} group"
                )
            group = manager.sp_group
            for processor in processors:
                processor.group = group
            if ffn == "fp8":
                for block in refiner.refiner_blocks:
                    block.ff.net[0].proj.activation_group = group
                    block.ff.net[2].activation_group = group
            bound = True
        pad = set_pad(PAD_KEY, tokens, group) if sp else 0
        metrics["calls"] += 1
        metrics["last_input_tokens"] = tokens
        metrics["last_padding"] = pad
        metrics["max_input_tokens"] = max(tokens, metrics["max_input_tokens"])
        local = (
            split_sequence(hidden_states, group, dim=1, pad=pad)
            if sp
            else hidden_states
        )
        metrics["local_tokens"] = local.shape[1]
        local = original(local)
        result = gather_sequence(local, group, dim=1, pad=pad) if sp else local
        metrics["completed"] += 1
        return result

    refiner.forward = forward
    refiner._h3_sp_metrics = metrics
    refiner._h3_cudnn_backend = backend
    model._h3_input_refiner_profile = profile
    logger.info(
        "Experimental H3 input Token Refiner: profile=%s SP=%d wire=%s attention=%s FFN=%s FP8_projections=%d; main DiT unchanged",
        profile,
        metrics["sp_size"],
        metrics["communication_dtype"],
        attention,
        ffn,
        metrics["fp8_ffn_projections"],
    )


# ------------------------------------------------------------------------
# transformer.h3.ff_chunking
# ------------------------------------------------------------------------

# Optional native FeedForward chunking for H3's replicated input refiner.









def install_ff_chunking(model, chunk_size):
    """Bound token-wise FFN temporaries without changing attention or weights."""
    from diffusers.models.attention import _chunked_feed_forward

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("H3 input refiner FFN chunk size must be a positive integer")
    if hasattr(model, "_h3_refiner_ff_chunking"):
        raise ValueError("H3 input refiner FFN chunking is already installed")
    metrics = dict(
        chunk_size=chunk_size,
        modules=len(model.token_refiner.refiner_blocks),
        calls=0,
        completed=0,
        chunks_completed=0,
        max_input_tokens=0,
    )

    def wrap(original):
        @wraps(original)
        def forward(hidden_states, *args, **kwargs):
            if torch.is_grad_enabled():
                raise ValueError(
                    "Experimental H3 refiner FFN chunking is inference-only"
                )
            if hidden_states.ndim != 3:
                raise ValueError(
                    "H3 input refiner FFN expects batch/token/channel input"
                )
            tokens = hidden_states.shape[1]
            metrics["calls"] += 1
            metrics["max_input_tokens"] = max(tokens, metrics["max_input_tokens"])
            if tokens <= chunk_size:
                output = original(hidden_states, *args, **kwargs)
                metrics["chunks_completed"] += 1
            else:
                prefix = tokens // chunk_size * chunk_size

                def ff(value):
                    result = original(value, *args, **kwargs)
                    metrics["chunks_completed"] += 1
                    return result

                output = _chunked_feed_forward(
                    ff, hidden_states[:, :prefix], 1, chunk_size
                )
                if prefix != tokens:
                    tail = ff(hidden_states[:, prefix:])
                    output = torch.cat((output, tail), dim=1)
            metrics["completed"] += 1
            return output

        return forward

    for block in model.token_refiner.refiner_blocks:
        block.ff.forward = wrap(block.ff.forward)
    model._h3_refiner_ff_chunking = metrics
    logger.info(
        "Experimental H3 input refiner FFN chunking=%d tokens, blocks=%d; attention/weights unchanged",
        chunk_size,
        metrics["modules"],
    )


# ------------------------------------------------------------------------
# transformer.h3.fusion
# ------------------------------------------------------------------------

# Model-specific CUDA fusion binding for H3 blocks.


from types import MethodType





def install_dit_fusions(transformer, enabled):
    if "XGEN_H3_FUSED_OUTPUT_ADALN" in enabled:
        transformer.norm_out._h3_fused_indexed_affine = True
        logger.info("Experimental CUDA H3 output AdaLN indexed affine fusion enabled")
    if "XGEN_H3_FUSED_INDEXED_AFFINE" in enabled:
        from x_base.operator.h3_indexed_affine import fused_h3_block_forward

        for block in transformer.transformer_blocks:
            block._h3_eager_forward = block.forward
            block._h3_fused_indexed_affine = True
            block.forward = MethodType(fused_h3_block_forward, block)
        logger.info("Experimental CUDA H3 indexed affine fusion enabled")
    if "XGEN_H3_FUSED_PARTIAL_ROPE" in enabled:
        for block in transformer.transformer_blocks:
            block.attn._h3_fused_partial_rope = True
        logger.info("Experimental CUDA H3 partial RoPE fusion enabled")
    if "XGEN_H3_FUSED_QK_RMS" in enabled:
        for block in transformer.transformer_blocks:
            block.attn._h3_fused_qk_rms = True
        logger.info("Experimental CUDA H3 Q/K RMS fusion enabled")
    if "XGEN_H3_NVFP4_FUSED_AMAX" in enabled:
        from x_base.operator.h3_nvfp4 import H3NVFP4Linear

        for module in transformer.modules():
            if isinstance(module, H3NVFP4Linear):
                module.fused_amax = True
        logger.info("Experimental CUDA H3 NVFP4 fused amax enabled")
    if "XGEN_H3_FUSED_SWIGLU" in enabled:
        from x_base.operator.h3_nvfp4 import H3GateFirstSwiGLU

        for module in transformer.modules():
            if isinstance(module, H3GateFirstSwiGLU):
                module.fused_swiglu = True
        logger.info("Experimental CUDA H3 SwiGLU fusion enabled")
    if "XGEN_H3_FUSED_FF_AMAX" in enabled:
        from x_base.operator.h3_nvfp4 import (
            H3GateFirstSwiGLU,
            H3NVFP4Linear,
            fused_h3_ff_forward,
        )

        for block in transformer.transformer_blocks:
            ff = block.ff
            if (
                len(ff.net) != 3
                or not isinstance(ff.net[0], H3GateFirstSwiGLU)
                or not isinstance(ff.net[1], torch.nn.Dropout)
                or not isinstance(ff.net[2], H3NVFP4Linear)
            ):
                raise ValueError(
                    "H3 fused FFN amax requires gate-first NVFP4 FeedForward"
                )
            ff._h3_eager_ff_forward = ff.forward
            ff._h3_fused_ff_amax = True
            ff.forward = MethodType(fused_h3_ff_forward, ff)
        logger.info("Experimental CUDA H3 SwiGLU with activation amax enabled")


def _extended_dit_block(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    from x_base.operator.h3_indexed_affine import _apply
    from x_base.operator.h3_norm_adaln import rms_adaln

    if torch.is_grad_enabled():
        return self._h3_previous_affine(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = self.adaln_proj(temb)
    first = rms_adaln(hidden_states, self.norm1, shift_attn, scale_attn, adaln_indices)
    if first is None:
        return self._h3_previous_affine(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    attention = self.attn(first[1], rotary_emb, attention_mask)
    second = rms_adaln(
        hidden_states, self.norm2, shift_ff, scale_ff, adaln_indices, branch=attention, gate=gate_attn,
    )
    if second is None:
        residual = _apply(attention, gate_attn, hidden_states, adaln_indices, modulation=False)
        normed = _apply(self.norm2(residual), scale_ff, shift_ff, adaln_indices, modulation=True)
    else:
        residual, normed = second
    return _apply(self.ff(normed), gate_ff, residual, adaln_indices, modulation=False)


def _value_first_swiglu(self, hidden_states):
    from x_base.operator.h3_swiglu import value_first_swiglu

    if torch.is_grad_enabled():
        return self._h3_eager_swiglu_forward(hidden_states)
    projected = self.proj(hidden_states)
    if projected.is_cuda and projected.dtype == torch.bfloat16 and projected.is_contiguous():
        return value_first_swiglu(projected)
    values, gate = projected.chunk(2, dim=-1)
    return values * self.activation(gate)


def _install_extended_dit_fusions(model, device):
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 3), (12, 0)):
        logger.info("H3 extended CUDA DiT fusions unavailable on compute capability %s", capability)
        return
    for block in model.transformer_blocks:
        block._h3_previous_affine = block.forward
        block.forward = MethodType(_extended_dit_block, block)
        activation = block.ff.net[0]
        if type(activation) is SwiGLU:
            activation._h3_eager_swiglu_forward = activation.forward
            activation.forward = MethodType(_value_first_swiglu, activation)
    model._h3_rope_table_enabled = True
    logger.info(
        "H3 extended CUDA DiT fusions enabled: RMSNorm/AdaLN/residual, BF16 SwiGLU, request RoPE table"
    )


def configure_phaa_transport(
    model,
    *,
    gather_priority,
    chunk_order,
    device_join,
    async_options,
    scatter_group,
    scatter_stream,
):
    for block in model.transformer_blocks:
        block.attn._h3_phaa_fp8 = True
        block.attn._h3_phaa_gather_priority = gather_priority
        block.attn._h3_phaa_tail_first = chunk_order == "tail-first"
        block.attn._h3_phaa_device_join = device_join
        block.attn._h3_phaa_async_options = async_options
        block.attn._h3_phaa_scatter_group = scatter_group
        block.attn._h3_phaa_scatter_stream = scatter_stream


# ------------------------------------------------------------------------
# transformer.h3.nvfp4
# ------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0
# Load the FastVideo pruned NVFP4 H3 checkpoint into the XGEN DiT.






from safetensors import safe_open

from x_base.operator.h3_nvfp4 import H3GateFirstSwiGLU, read_packed_h3_projections


_ALIASES = {
    "video_patch_proj": "proj_in",
    "audio_patch_proj": "audio_proj_in",
    "condition_proj": "context_embedder",
    "blocks": "transformer_blocks",
    "token_refiner.blocks": "token_refiner.refiner_blocks",
    "final_layer.norm": "norm_out.norm",
    "final_layer.adaln_proj.linear": "norm_out.linear",
    "final_layer.video_out": "proj_out",
    "final_layer.audio_out": "audio_proj_out",
}
_BLOCK_ALIASES = {
    ".attn.qkv_proj.": ".attn.to_qkv.",
    ".attn.out_proj.": ".attn.to_out.0.",
    ".attn.q_norm.": ".attn.norm_q.",
    ".attn.k_norm.": ".attn.norm_k.",
    ".mlp.fc1.": ".ff.net.0.proj.",
    ".mlp.fc2.": ".ff.net.2.",
}


@torch.no_grad()
def load_packed_h3(model_cls, config, checkpoint):
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        table = source.get_tensor("adaln_t_table")
        if table.shape != (1025, 8) or table.dtype != torch.float32:
            raise ValueError("Expected a 1025 by 8 FP32 H3 AdaLN curve")

        with torch.device("meta"):
            model = model_cls.from_config(config, time_embed_dim=table.shape[1])
            del model.time_embedder
            for block in [
                *model.transformer_blocks,
                *model.token_refiner.refiner_blocks,
            ]:
                block.attn.fuse_projections()
                del block.attn.to_q, block.attn.to_k, block.attn.to_v
                block.ff.net[0] = H3GateFirstSwiGLU(block.ff.net[0].proj)

        packed = read_packed_h3_projections(checkpoint, len(model.transformer_blocks))
        packed_buffers = set()
        for name, module in packed.items():
            original = model.get_submodule(name)
            if (original.in_features, original.out_features) != (
                module.in_features,
                module.out_features,
            ):
                raise ValueError(f"Packed H3 projection shape mismatch for {name}")
            model.set_submodule(name, module)
            packed_buffers.update(f"{name}.{key}" for key in module.state_dict())

        for block in model.transformer_blocks:
            block.adaln_proj._h3_packed_adaln = True
        model.norm_out._h3_packed_adaln = True
        model.register_buffer("_h3_packed_adaln_table", table, persistent=False)
        model.rope.inv_freq = source.get_tensor("rope.inv_freq")

        expected = model.state_dict()
        dense = {}
        for key in source.keys():
            if key in {"adaln_t_table", "rope.inv_freq"}:
                continue
            name = key
            for original, target in _ALIASES.items():
                if name.startswith(original + "."):
                    name = target + name[len(original) :]
                    break
            for original, target in _BLOCK_ALIASES.items():
                if original in name:
                    name = name.replace(original, target, 1)
                    break
            module_name, _, suffix = name.rpartition(".")
            if module_name in packed and suffix in {
                "weight",
                "bias",
                "weight_scale",
                "weight_scale_2",
                "comfy_quant",
            }:
                continue
            if name not in expected or name in dense:
                raise ValueError(
                    f"Unexpected H3 checkpoint tensor {key} mapped to {name}"
                )
            value = source.get_tensor(key)
            dtype = (
                torch.float32
                if ".adaln_proj.linear." in name or name.startswith("norm_out.linear.")
                else torch.bfloat16
            )
            dense[name] = value.to(dtype)
        missing = expected.keys() - dense.keys() - packed_buffers
        if missing:
            raise ValueError(f"Missing H3 checkpoint tensors: {sorted(missing)[:8]}")
        model.load_state_dict(dense, strict=False, assign=True)
    model.requires_grad_(False)
    return model.eval()


def compile_resident_blocks(transformer):
    from x_base.operator.h3_nvfp4 import H3NVFP4Linear

    for block in transformer.transformer_blocks:
        # Keep stream choreography and packed custom kernels outside Dynamo.
        block.attn.forward = torch.compiler.disable(block.attn.forward)
        for module in block.ff.modules():
            if isinstance(module, H3NVFP4Linear):
                module.forward = torch.compiler.disable(module.forward)
        block.compile(dynamic=False, options={"triton.cudagraphs": False})
    logger.info(
        "Regional compilation enabled for %d H3 DiT blocks",
        len(transformer.transformer_blocks),
    )


# ------------------------------------------------------------------------
# transformer.minimax_h3
# ------------------------------------------------------------------------

# NPU adaptation for the MiniMax-H3 transformer.
#
# This module replaces the upstream ``MiniMaxH3AttnProcessor`` with
# ``AscendMiniMaxH3AttnProcessor``, which calls ``attention_manager.attention``
# directly — the same x_base interface used by the Wan and HunyuanVideo
# adaptations.  This gives MiniMax-H3 immediate access to all x_base attention
# backends (sage, laser, rainfusion, ada_sparse) without going through
# ``dispatch_attention_fn``.
#
# Sequence-parallel (SP) support is added via the same
# ``all_to_all_before_attn`` / ``all_to_all_after_attn`` pair that the Wan
# adaptation uses.  When ``sp_size == 1`` the processor is a straight pass-through
# to ``attention_manager.attention``.


import os

from typing import Any, Optional



from x_base.device import device_type

if device_type() == "npu":
    import torch_npu  # noqa: F401  – registers NPU kernels
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


from diffusers.models.transformers import transformer_minimax_h3
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3AdaLayerNormModulation, MiniMaxH3AdaLayerNormOut, MiniMaxH3Attention, MiniMaxH3AttnProcessor, MiniMaxH3TransformerOutput, _apply_rotary_emb
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.utils import apply_lora_scale


from x_base import ParallelManager, all_to_all_before_attn, all_to_all_after_attn, attention_manager, rope_manager, split_sequence, is_phaa_enabled, get_phaa_split_num


def _rope_minimax_h3(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for MiniMax-H3: partial rotation (96/128 channels) with rotate_half convention.

    MiniMax-H3 rotates only the first ``rotary_dim`` channels (96 of 128) using
    the rotate_half convention.  The remaining 32 channels pass through unchanged.

    Routes through ``rope_manager`` for the fused NPU path when available,
    falling back to the reference ``_apply_rotary_emb`` implementation otherwise.
    """
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts.
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    if rope_manager.is_rope_fused:
        # Try fused mindie backend with rotate_half mode.
        try:
            from x_base.operator.rope import AVAILABLE_MINDIESD, rotary_position_embedding

            if AVAILABLE_MINDIESD:
                q_rot = rotary_position_embedding(
                    q_rot.contiguous(), cos, sin, rotated_mode="rotated_half", fused=True
                )
                k_rot = rotary_position_embedding(
                    k_rot.contiguous(), cos, sin, rotated_mode="rotated_half", fused=True
                )
                return (
                    torch.cat([q_rot, q_pass], dim=-1),
                    torch.cat([k_rot, k_pass], dim=-1),
                )
        except Exception:
            pass  # Fall through to base implementation.

    # Base implementation: rotate_half convention (reference).
    query_out = _apply_rotary_emb(query, cos, sin)
    key_out = _apply_rotary_emb(key, cos, sin)
    return query_out, key_out


class AscendMiniMaxH3AttnProcessor:
    """NPU attention processor for MiniMax-H3 with optional SP support.

    When ``attn.parallel_manager`` is set and ``sp_size > 1``, the heads are
    scattered across the SP group (Ulysses pattern) before attention and
    gathered back afterwards.  Otherwise the call is forwarded directly to
    ``attention_manager.attention`` — the same x_base interface that the Wan
    and HunyuanVideo adaptations use — so that sage / laser / rainfusion /
    ada_sparse attention are all available through the global
    ``attention_manager`` configuration.
    """

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- projections (identical to the upstream processor) ---
        model = _h3_model(attn)
        projected = _h3_project_qkv(attn, hidden_states)
        if projected is not None:
            query, key, value = projected
        elif attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        normalized = None
        fused_rope = getattr(attn, "_h3_fused_partial_rope", False)
        if getattr(attn, "_h3_fused_qk_rms", False):
            from x_base.operator.h3_qk_rms import try_h3_qk_rms

            normalized = try_h3_qk_rms(
                query, key, attn.norm_q, attn.norm_k, rotary_emb if fused_rope else None,
            )
        if normalized is None:
            query, key = attn.norm_q(query), attn.norm_k(key)
        else:
            query, key = normalized

        # RoPE applied before SP branch (original position).
        # all_to_all doesn't change (position, head) mapping, so RoPE stays valid.
        if rotary_emb is not None and not (fused_rope and normalized is not None):
            rotated = None
            if fused_rope:
                from x_base.operator.h3_partial_rope import try_h3_partial_rope

                rotated = try_h3_partial_rope(query, key, *rotary_emb)
            query, key = rotated if rotated is not None else _h3_rope(
                model, query, key, *rotary_emb, fallback=_rope_minimax_h3,
            )
        parallel_manager = getattr(attn, "parallel_manager", None)
        sp_size = parallel_manager.sp_size if parallel_manager is not None else 1
        if getattr(attn, '_h3_tp_cp', False) and sp_size > 1:
            stats = getattr(attn, '_h3_cp_metrics', None)
            if stats is None:
                stats = attn._h3_cp_metrics = dict(calls=0, logical_payload_bytes=0)
            stats['calls'] += 1
            # Includes self exchange, Q/K/V plus returning attention output;
            # excludes topology/protocol overhead and is not NIC traffic.
            fp8_wire = getattr(attn, '_h3_phaa_fp8', False)
            stats['logical_payload_bytes'] += 4 * (query.numel() // query.shape[-1]) * (
                query.shape[-1]+4 if fp8_wire else query.shape[-1]*query.element_size())

        if sp_size > 1:
            if is_phaa_enabled():
                # PHAA path: RoPE applied inside run_phaa_attention via rope_fn
                return self._phaa_parallel(
                    attn=attn,
                    query=query,
                    key=key,
                    value=value,
                    rotary_emb=rotary_emb,
                    attention_mask=attention_mask,
                    sp_group=parallel_manager.sp_group,
                    sp_size=sp_size,
                )
            # Ulysses SP: scatter heads (dim=2), gather sequence (dim=1).
            attn_heads = attn.heads // sp_size
            query, key, value = map(
                lambda x: all_to_all_before_attn(
                    x, parallel_manager.sp_group, scatter_dim=2, gather_dim=1,
                    pad_key='mmh3_seq' if getattr(attn, '_h3_tp_cp', False) else 'pad',
                ),
                [query, key, value],
            )
        else:
            attn_heads = attn.heads

        # Transpose to (B, H, S, D) for attention dispatch.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)


        if getattr(attn, '_h3_attention_backend', None) is not None:
            hidden_states = attn._h3_attention_backend(
                query, key, value, attn_mask=attention_mask)
        elif getattr(attn, "_h3_turbo_sla", False):
            from x_base.operator.h3_sla import sparse_attention
            # A full real sequence is published by the transformer before SP
            # adds tail padding. Text-refiner attention is never SP-sharded.
            used_len = getattr(attn, "_h3_sla_used_len", query.shape[-2])
            if attention_mask is not None:
                raise ValueError(
                    "Turbo-SLA reference does not accept arbitrary attention masks. "
                    "Use a real-token H3 packed sequence; SP tail padding is stripped separately."
                )
            hidden_states = sparse_attention(query, key, value, used_len=used_len)
        else:
            hidden_states = attention_manager.attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if sp_size > 1:
            hidden_states = all_to_all_after_attn(
                hidden_states,
                parallel_manager.sp_group,
                scatter_dim=1,
                gather_dim=2,
                pad_key='mmh3_seq' if getattr(attn, '_h3_tp_cp', False) else 'pad',
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _phaa_parallel(
        self,
        attn: MiniMaxH3Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sp_group=None,
        sp_size: int = 1,
    ) -> torch.Tensor:
        """PHAA for MiniMax-H3: overlap all2all with flash attention.

        Thin wrapper around ``run_phaa_attention``. MiniMax-H3 has no img
        attention (extra_attn_fn=None) and uses partial-rotation RoPE.

        ``run_phaa_attention`` handles unequal chunk sizes internally:
        56 heads, 8 NPUs, phaa_num=2 → 7 heads/dev → chunks [4, 3].
        """
        head_dim = query.shape[-1]

        # Flatten to (B, S, heads * head_dim) for run_phaa_attention
        query = query.flatten(-2)
        key = key.flatten(-2)
        value = value.flatten(-2)

        hidden_states = _h3_phaa(
            attn, query, key, value,
            comm_stream=self.comm_stream,
            sp_group=sp_group,
            sp_size=sp_size,
            heads=attn.heads,
            phaa_split_num=get_phaa_split_num() or attn.heads // sp_size,
            head_dim=head_dim,
            rope_fn=None,  # RoPE already applied before SP branch
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            pad_key="mmh3_seq",
            attention_fn=getattr(attn, '_h3_attention_backend', None),
            fp8_transport=getattr(attn, '_h3_phaa_fp8', False),
            gather_priority=getattr(attn, '_h3_phaa_gather_priority', False),
            tail_first=getattr(attn, '_h3_phaa_tail_first', False),
            device_join=getattr(attn, '_h3_phaa_device_join', False),
            async_options=getattr(attn, '_h3_phaa_async_options', None),
            scatter_group=getattr(attn, '_h3_phaa_scatter_group', None),
            scatter_stream=getattr(attn, '_h3_phaa_scatter_stream', None),
        )

        hidden_states = _h3_project_output(attn.to_out[0], hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# Monkey-patch: replace default processor with AscendMiniMaxH3AttnProcessor.
# Attention backend selection happens through global attention_manager, not set_attention_backend.
MiniMaxH3Attention._default_processor_cls = AscendMiniMaxH3AttnProcessor
MiniMaxH3Attention._available_processors = [
    MiniMaxH3AttnProcessor,
    AscendMiniMaxH3AttnProcessor,
]




def _set_parallel_manager(self, parallel_manager):
    """Attach a ``ParallelManager`` so the processor can read ``sp_size``."""
    self.parallel_manager = parallel_manager
    for block in getattr(self, "transformer_blocks", []):
        if hasattr(block, "attn"):
            block.attn.parallel_manager = parallel_manager
    # The default input refiner runs before packed-sequence splitting. Its
    # optional CUDA adapter installs a separate SP processor and padding key.


MiniMaxH3Transformer3DModel = transformer_minimax_h3.MiniMaxH3Transformer3DModel
MiniMaxH3Transformer3DModel.set_parallel_manager = _set_parallel_manager


def _enable_parallel(self, dp_size, sp_size, enable_cp=False, ulysses_size=None, ring_size=None):
    """Enable sequence parallelism on the MiniMax-H3 transformer.

    Mirrors ``WanTransformer3DModel.enable_parallel``: creates a
    ``ParallelManager`` and propagates it to every attention module via
    ``set_parallel_manager``.
    """
    cp_size = 2 if enable_cp and sp_size % 2 == 0 else 1
    if cp_size > 1:
        sp_size = sp_size // 2
    parallel_manager = ParallelManager(dp_size, cp_size, sp_size, 1, ulysses_size, ring_size)
    self.set_parallel_manager(parallel_manager)


MiniMaxH3Transformer3DModel.enable_parallel = _enable_parallel


def _build_packed_sequence(
    text_embeds: torch.Tensor,
    video_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    text_idx: torch.Tensor,
    video_idx: torch.Tensor,
    audio_idx: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Build packed hidden_states from text/video/audio embeds."""
    hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    hidden_states = hidden_states.index_copy(1, text_idx, text_embeds)
    hidden_states = hidden_states.index_copy(1, video_idx, video_embeds.to(text_embeds.dtype))
    hidden_states = hidden_states.index_copy(1, audio_idx, audio_embeds.to(text_embeds.dtype))
    return hidden_states


def _split_sequence_for_sp(
    parallel_manager,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    adaln_indices: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> tuple:
    """Split tensors across SP group. Returns (sp_group, pad, hidden_states, rotary_emb, adaln_indices, attention_mask)."""
    if parallel_manager is None or parallel_manager.sp_size < 2:
        return None, None, hidden_states, rotary_emb, adaln_indices, attention_mask

    sp_group = parallel_manager.sp_group
    set_pad("mmh3_seq", hidden_states.shape[1], sp_group)
    _pad = get_pad("mmh3_seq")

    hidden_states = split_sequence(hidden_states, sp_group, dim=1, pad=_pad)
    cos, sin = rotary_emb
    cos = split_sequence(cos, sp_group, dim=0, pad=_pad)
    sin = split_sequence(sin, sp_group, dim=0, pad=_pad)
    rotary_emb = (cos, sin)
    adaln_indices = split_sequence(adaln_indices, sp_group, dim=0, pad=_pad)
    if attention_mask is not None:
        attention_mask = split_sequence(attention_mask, sp_group, dim=0, pad=_pad)
        attention_mask = split_sequence(attention_mask, sp_group, dim=1, pad=_pad)
    return sp_group, _pad, hidden_states, rotary_emb, adaln_indices, attention_mask


def _prepare_timestep_adaln_mask(
    self,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute timestep embedding, adaln_indices, and attention mask."""
    table = getattr(self, '_h3_packed_adaln_table', None)
    if table is not None:
        from x_base.operator.h3_nvfp4 import interpolate_h3_adaln

        temb = interpolate_h3_adaln(timestep, table)
    else:
        temb = self.time_proj(timestep)
        _adaln_cache = getattr(self, '_adaln_cache', None)
        if _adaln_cache is None or not _adaln_cache.has_current:
            temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

    attention_mask = None
    is_pad = token_tags < 0
    if bool(is_pad.any()):
        attention_mask = is_pad[None, :] == is_pad[:, None]

    return temb, adaln_indices, attention_mask


def _gather_and_output(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    sp_group,
    pad,
    timestep_indices: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    return_dict: bool,
) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
    """Gather sequence and compute output heads."""
    if getattr(self, '_h3_cuda_local_output', False):
        manager = self.parallel_manager
        span = LocalSpan.for_rank(len(timestep_indices), manager.sp_size, manager.sp_rank)
        if hidden_states.shape[1] != span.length:
            raise ValueError('H3 SP local output received an incompatible sequence shard')
        video, audio = gather_local_output(self, hidden_states, temb, span,
                                          timestep_indices, video_indices, audio_indices)
        if not return_dict:
            return video, audio
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)
    if sp_group is not None:
        hidden_states = gather_sequence(hidden_states, sp_group, dim=1, pad=pad)

    hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
    video_output = self.proj_out(hidden_states).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
    if not return_dict:
        return (video_output, audio_output)
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


@contextmanager
def request_rope_table(model):
    """Keep one rank-local RoPE table for the duration of a video request."""
    state = dict(model=model, positions=None, table=None, builds=0, hits=0)
    token = _ROPE_TABLE.set(state)
    try:
        yield
    finally:
        _ROPE_TABLE.reset(token)
        model._h3_rope_table_stats = {key: state[key] for key in ("builds", "hits")}
        logger.info("H3 request RoPE table: %s", model._h3_rope_table_stats)


def _compute_rope(
    self,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the original RoPE computation for exactly matching positions."""
    state = _ROPE_TABLE.get()
    if state is None or state["model"] is not self or torch.is_grad_enabled():
        return self.rope(position_ids)
    previous = state["positions"]
    version = _version(position_ids)
    if previous is not None:
        unchanged = (
            previous is position_ids and version is not None and version == state["version"]
        )
        snapshot = state["snapshot"]
        if snapshot is None and _version(previous) == state["version"]:
            snapshot = previous
        if unchanged or (
            snapshot is not None and snapshot.device == position_ids.device
            and snapshot.dtype == position_ids.dtype and torch.equal(snapshot, position_ids)
        ):
            state["hits"] += 1
            return state["table"]
    result = self.rope(position_ids)
    state.update(
        positions=position_ids, version=version, table=result,
        snapshot=position_ids.clone() if version is None else None,
        builds=state["builds"] + 1,
    )
    return result


def _project_modalities(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project video, audio, and text modalities. Returns (video_embeds, audio_embeds, text_embeds)."""
    if getattr(self, '_h3_precompute_conditioning', False):
        text_embeds = project_text(self, encoder_hidden_states)
    video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
    if not getattr(self, '_h3_precompute_conditioning', False):
        text_embeds = project_text(self, encoder_hidden_states)
    return video_embeds, audio_embeds, text_embeds


def _run_transformer_blocks(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run transformer blocks with optional gradient checkpointing."""
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
            )
        else:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    return hidden_states


# Sequence-level SP: split/gather sequence, rotary_emb, adaln_indices,
# and attention_mask around the block loop when sp_size > 1.
_original_mmh3_forward = MiniMaxH3Transformer3DModel.forward


@apply_lora_scale("attention_kwargs")
def _mmh3_forward_with_sp(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:

    # Update denoise step counter for rainfusion skip-timestep logic
    attention_manager.update_t_idx()

    parallel_manager = getattr(self, "parallel_manager", None)
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    # Publish real length before the single-card early return as well as SP.
    for block in self.transformer_blocks:
        if getattr(block.attn, "_h3_turbo_sla", False):
            block.attn._h3_sla_used_len = position_ids.shape[0]

    # --- AdaLN schedule cache: build/install table for this timestep ---
    _adaln_cache = getattr(self, "_adaln_cache", None)
    if _adaln_cache is not None:
        _adaln_cache.maybe_install(timestep)

    # --- No SP: delegate to the original forward (undecorated, LoRA already handled) ---
    local_io = getattr(self, '_h3_cuda_local_input', getattr(self, '_h3_parallel_config', None) is not None)
    if sp_size < 2 and not hasattr(self, '_h3_packed_adaln_table') and not local_io:
        return _original_mmh3_forward.__wrapped__(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    # --- SP path: full forward with split / gather around the block loop ---

    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
    sequence_length = position_ids.shape[0]
    if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
        raise ValueError(
            "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
            f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
        )

    # RoPE table cache: reuse across denoise steps when position_ids unchanged
    if local_io:
        hidden_states, rotary_emb, local_span = prepare_local(
            self, hidden_states, audio_hidden_states, encoder_hidden_states, position_ids,
            text_indices, video_indices, audio_indices,
        )
    else:
        rotary_emb = _compute_rope(self, position_ids)
        shared_exchange = getattr(self, '_h3_shared_sequence_exchange', None)
        if shared_exchange is not None:
            shared_exchange.set_tokens(sequence_length)
            shared_rotary = rotary_emb
        # 1. Project each modality and scatter into the packed sequence buffer.
        video_embeds, audio_embeds, text_embeds = _project_modalities(
            self, hidden_states, audio_hidden_states, encoder_hidden_states
        )
        hidden_states = _build_packed_sequence(
            text_embeds, video_embeds, audio_embeds, text_indices, video_indices, audio_indices, sequence_length
        )

    # 2. Timestep embedding, AdaLN indices, and padding mask.
    temb, adaln_indices, attention_mask = _prepare_timestep_adaln_mask(
        self, timestep, timestep_indices, token_tags,
    )

    # --- Split sequence across SP group before the block loop ---
    if local_io:
        # PHAA strips only CP tail padding. Existing padding masks retain the
        # complete real-token document on every attention rank.
        adaln_indices = local_span.slice_padded(adaln_indices)
    else:
        sp_group, _pad, hidden_states, rotary_emb, adaln_indices, attention_mask = _split_sequence_for_sp(
            parallel_manager, hidden_states, rotary_emb, adaln_indices, attention_mask
        )
        if getattr(self, '_h3_shared_sequence_exchange', None) is not None:
            rotary_emb = shared_rotary
    # Publish the meaningful sequence length (without SP padding) so that
    # RainFusion sparse attention can slice off the zero-padding rows before
    # dispatching to the rf_v2 kernel and pad the output back afterwards.
    if attention_manager.is_rainfusion_attention:
        attention_manager.set_rainfusion_seq_len(sequence_length)

    # Set _block_idx for AdaLN cache lookup
    for i, block in enumerate(self.transformer_blocks):
        block._block_idx = i

    # 3. Transformer blocks
    hidden_states = _run_transformer_blocks(
        self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
    )

    if local_io and getattr(self, '_h3_cuda_local_output', True):
        video, audio = gather_local_output(self, hidden_states, temb, local_span,
                                          timestep_indices, video_indices, audio_indices)
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio) if return_dict else (video, audio)
    if local_io:
        sp_group = parallel_manager.sp_group
        _pad = local_span.length * parallel_manager.sp_size - sequence_length
    return _gather_and_output(
        self, hidden_states, temb, sp_group, _pad,
        timestep_indices, video_indices, audio_indices, return_dict,
    )


MiniMaxH3Transformer3DModel.forward = _mmh3_forward_with_sp
MiniMaxH3Transformer3DModel.configure_fused_ops = configure_h3_fused_ops
MiniMaxH3Transformer3DModel.inference_context = h3_inference_context
MiniMaxH3TransformerBlock.forward = h3_block_forward


_original_adaln_forward = MiniMaxH3AdaLayerNormModulation.forward


def _patched_adaln_forward(self, temb):
    # Cache hit: skip SiLU + Linear + view + chunk
    cache = getattr(self, '_adaln_cache', None)
    if cache is not None and cache.has_current:
        return cache.lookup_block(self._block_idx)

    # File cache mode: weights are stripped, cannot fall back to computation
    if cache is not None and cache._file_cache_loaded:
        raise RuntimeError(
            f"AdaLN cache miss — weights have been pruned, cannot compute. "
            f"Block {self._block_idx}. Check cache file and sampler config."
        )

    # Runtime cache mode: original logic (weights still exist)
    linear_dtype = self.linear.weight.dtype
    if linear_dtype == torch.int8:
        linear_dtype = torch.bfloat16
    if getattr(self, '_h3_packed_adaln', False):
        temb = self.linear(temb).to(torch.bfloat16)
    else:
        temb = self.linear(nn.functional.silu(temb).to(linear_dtype))
    temb = temb.view(-1, 6 * self.hidden_size)
    return temb.chunk(6, dim=-1)


MiniMaxH3AdaLayerNormModulation.forward = _patched_adaln_forward


# Monkey-patch AdaLayerNormOut.forward for AdaLN schedule cache.


def _patched_adaln_out_forward(self, hidden_states, temb, timestep_indices):
    cache = getattr(self, '_adaln_cache', None)
    if cache is not None and cache.has_current:
        shift, scale = cache.lookup_final()
    elif cache is not None and cache._file_cache_loaded:
        raise RuntimeError(
            "AdaLN cache miss — weights have been pruned, cannot compute. "
            "Final layer. Check cache file and sampler config."
        )
    else:
        out_dtype = self.linear.weight.dtype
        if out_dtype == torch.int8:
            out_dtype = torch.bfloat16
        if getattr(self, '_h3_packed_adaln', False):
            modulation = self.linear(temb).to(hidden_states.dtype)
        else:
            modulation = self.linear(nn.functional.silu(temb).to(out_dtype))
        shift, scale = modulation.chunk(2, dim=-1)
    hidden_states = self.norm(hidden_states)
    if (getattr(self, '_h3_fused_indexed_affine', False)
            and hidden_states.is_cuda and not torch.is_grad_enabled()):
        from x_base.operator.h3_indexed_affine import _apply

        return _apply(hidden_states, scale, shift, timestep_indices, modulation=True)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


MiniMaxH3AdaLayerNormOut.forward = _patched_adaln_out_forward


_orig_seq_call = SequentialPipelineBlocks.__call__


@torch.no_grad()
def _timed_seq_call(self, pipeline, state):
    if os.environ.get("DEBUG_TIME") and self.model_name == "minimax-h3":
        is_dist = dist.is_initialized()
        is_rank0 = not is_dist or dist.get_rank() == 0
        accelerator = torch.get_device_module(device_type())
        for block_name, block in self.sub_blocks.items():
            if is_dist:
                dist.barrier()
            accelerator.synchronize()
            t0 = time.time()
            pipeline, state = block(pipeline, state)
            accelerator.synchronize()
            if is_dist:
                dist.barrier()
            if is_rank0:
                print(f"[DEBUG_TIME] {block_name}: {time.time() - t0:.3f}s")
        return pipeline, state
    return _orig_seq_call(self, pipeline, state)


SequentialPipelineBlocks.__call__ = _timed_seq_call
