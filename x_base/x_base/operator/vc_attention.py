"""Optional VC ExpCast backend for dense NPU attention.

Initialize while selecting the backend, before model device work. Forward only
checks tensor metadata and dispatches; native execution errors propagate.
"""
import importlib
import logging
import math
import numbers
import os
import struct

import torch

LOG = logging.getLogger("attention")


def _is_npu(value):
    return value.device.type in ("npu", "privateuseone")


def _unsupported_reason(q, k, v, attn_mask, dropout_p, is_causal, scale):
    if attn_mask is not None or dropout_p != 0 or is_causal:
        return "mask, dropout or causal attention requires native SDPA"
    values = (q, k, v)
    if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in values):
        return "requires rank-4 BNSD tensors"
    if not all(x.layout == torch.strided and _is_npu(x) for x in values):
        return "requires strided NPU tensors"
    if not all(x.device == q.device and x.dtype == q.dtype for x in values):
        return "Q/K/V must use the same device and dtype"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "requires FP16 or BF16 inputs"
    if any(x.requires_grad for x in values):
        return "gradient inputs require native SDPA"
    if (q.shape[:2] != k.shape[:2] or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or q.shape[-1] not in (64, 128)
            or any(size == 0 for x in values for size in x.shape)):
        return "requires matching B/H/D, K/V lengths, D64/128 and nonempty tensors"
    if scale is not None:
        if (isinstance(scale, bool) or not isinstance(scale, numbers.Real)
                or not math.isfinite(scale) or not 0 < scale <= 65504):
            return "custom scale is outside the VC FP16 gain contract"
        fp32 = struct.unpack("<f", struct.pack("<f", float(scale)))[0]
        if struct.unpack("<e", struct.pack("<e", fp32))[0] <= 0:
            return "custom scale rounds to zero in the VC FP16 gain contract"
    return None


class VCAttention:
    """Own optional dependency loading and bounded diagnostics for one manager."""

    def __init__(self):
        self._initialized = False
        self._attention = None
        self._unavailable_reason = None
        self._seen = set()

    def initialize(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            module = importlib.import_module("vc_attention")
            loaded = module.load_ascend_expcast()
            if not callable(loaded) or loaded.info.get("expcast") is not True:
                raise ValueError("VC package did not load the ExpCast backend")
            self._attention = module.ascend_attention_expcast
            if not callable(self._attention):
                raise TypeError("VC package has no callable ascend_attention_expcast")
        except Exception as error:
            # Handle registration/build failures only at initialization. The
            # provider caches failed registration; retry in a new process.
            self._attention = None
            self._unavailable_reason = f"initialization failed: {type(error).__name__}: {error}"
            self._fallback(self._unavailable_reason)
        else:
            LOG.info("[VC_ATTENTION] rank=%s initialized=expcast algorithm=INT8_QK_FP16_ExpCast_FP16_PV",
                     os.environ.get("RANK", "0"))

    def _fallback(self, reason):
        if reason not in self._seen:
            self._seen.add(reason)
            LOG.warning("[VC_ATTENTION] rank=%s fallback=sdpa reason=%s",
                        os.environ.get("RANK", "0"), reason)

    def __call__(self, q, k, v, *, fallback, attn_mask=None,
                 dropout_p=0.0, is_causal=False, scale=None):
        reason = self._unavailable_reason or _unsupported_reason(
            q, k, v, attn_mask, dropout_p, is_causal, scale)
        if self._attention is None:
            reason = reason or "backend has not been initialized"
        if reason:
            self._fallback(reason)
            return fallback(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale)
        # No fallback after dispatch: a kernel failure must remain visible.
        result = self._attention(q, k, v, softmax_scale=scale)
        if "enabled" not in self._seen:
            self._seen.add("enabled")
            LOG.info("[VC_ATTENTION] rank=%s enabled=expcast shape=%s dtype=%s",
                     os.environ.get("RANK", "0"), tuple(q.shape), q.dtype)
        return result
