"""Patch torch.nn.functional.rms_norm to use NPU fused operator.

Enabled by default. Set X_GEN_FUSED_RMSNORM=0 to disable.
"""

import os

import torch
import torch.nn.functional as F
from x_base.device import device_type

if device_type() == "npu":
    import torch_npu

_orig_rms_norm = F.rms_norm
_patched = False


def _npu_rms_norm(input, normalized_shape, weight=None, eps=None):
    if weight is not None:
        return torch_npu.npu_rms_norm(input, weight, epsilon=eps if eps is not None else 1e-6)[0]
    return _orig_rms_norm(input, normalized_shape, weight, eps)


def enable_fused_rmsnorm():
    """Patch F.rms_norm to route to torch_npu.npu_rms_norm."""
    global _patched
    if _patched:
        return
    F.rms_norm = _npu_rms_norm
    _patched = True


def disable_fused_rmsnorm():
    """Restore original F.rms_norm."""
    global _patched
    if not _patched:
        return
    F.rms_norm = _orig_rms_norm
    _patched = False


# Auto-enable unless explicitly disabled
if os.environ.get("X_GEN_FUSED_RMSNORM", "1") != "0":
    enable_fused_rmsnorm()
