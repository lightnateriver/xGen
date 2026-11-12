"""Local adapter for the validated MindIE Laser ABI, without global patching.

The backend file identity pins the preprocessing/postprocessing contract used
by the existing validated build. Other backend revisions select native dispatch.
No third-party source is modified or dynamically compiled here.
"""
import hashlib
import importlib
import os
from . import capabilities as caps
from .runtime import tensors_ok
from pathlib import Path

VALIDATED_SOURCE_SHA256 = "1b111bd231706b6bb3cf6b113d2199e763fb7cf948c73e040feab4ba5e5ccc94"


def load_backend():
    backend = importlib.import_module("mindiesd.layers.flash_attn.ascend_laser_attention")
    path = Path(backend.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != VALIDATED_SOURCE_SHA256:
        raise ValueError("MindIE Laser contract differs from the validated backend")
    return backend


def attention(backend, q, k, v, preprocess):
    """Match the selected BNSD Laser path including output double rounding."""
    prepared = preprocess(q, k, v)
    padding = (-k.shape[2]) % 256
    pre_tokens = padding if padding else (1 << 31) - 1
    _, result = backend.ops.laser_attention(
        *prepared, None, None, None, q.shape[-1] ** -0.5, q.shape[1],
        "BNSD", 1.0, pre_tokens, 1, True,
    )
    return backend.AscendLaserAttention.la_postprocess_output(
        result, q.dtype, q.shape[2], q.shape[-1]
    )


_preprocess_backend_status = None


def _preprocess_backend():
    global _preprocess_backend_status
    if _preprocess_backend_status is None:
        try:
            _preprocess_backend_status = (load_backend(), None)
        except (ImportError, OSError, ValueError, AttributeError) as error:
            _preprocess_backend_status = (None, str(error))
    return _preprocess_backend_status


def attention_with_preprocess(q, k, v, *, kernel_registry, diagnostics,
                              attn_mask=None, dropout_p=0.0, is_causal=False):
    from x_base.operator.attention import attention_manager
    native = attention_manager.attention
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    if attention_manager.is_vc_attention:
        diagnostics.fallback("laser_preprocess", "VC-Attention provides its own preprocessing")
        return native(q, k, v, **kwargs)
    valid = (attention_manager.is_laser_attention and not attention_manager.is_sage_attention
             and os.environ.get("MINDIE_SD_FA_TYPE", "ascend_laser_attention") == "ascend_laser_attention"
             and attn_mask is None and dropout_p == 0 and not is_causal
             and tensors_ok(q, k, v) and q.ndim == 4 and q.shape == k.shape == v.shape
             and q.shape[0] == 1 and q.shape[1] == 2 and q.shape[2] >= 4000 and q.shape[3] == 128
             and all(t.stride(3) == 1 and 128 <= t.stride(2) < (1 << 30)
                     and t.stride(1) >= 0 for t in (q, k, v)))
    if not valid:
        diagnostics.fallback("laser_preprocess", "requires the selected Laser backend and BF16 [1,2,L>=4000,128] with compatible strides")
        return native(q, k, v, **kwargs)
    kernel = kernel_registry.get("laser_preprocess", q)
    if kernel is None:
        return native(q, k, v, **kwargs)
    backend, reason = _preprocess_backend()
    if backend is None:
        diagnostics.fallback("laser_preprocess", "Laser adapter unavailable: " + reason)
        return native(q, k, v, **kwargs)
    # Dispatch errors intentionally propagate; no fallback after partial work.
    result = attention(backend, q, k, v,
                             lambda *values: kernel.preprocess(*values, tile=64, blocks=caps.VECTOR_BLOCKS))
    diagnostics.hit("laser_preprocess", f"shape={tuple(q.shape)}")
    return result
