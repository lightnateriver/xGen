# SPDX-License-Identifier: Apache-2.0
"""CUDA H3 RMSNorm and indexed AdaLN fusion with xGen BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_adaln(
    X, Weight, Shift, Scale, Indices, Branch, Gate, ResidualOut, Out,
    width: tl.constexpr, eps: tl.constexpr, sequence: tl.constexpr,
    shift_stride: tl.constexpr, scale_stride: tl.constexpr, gate_stride: tl.constexpr,
    fused_residual: tl.constexpr, columns: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, columns)
    mask = channel < width
    offset = row * width + channel
    index = tl.load(Indices + row % sequence).to(tl.int32)
    value = tl.load(X + offset, mask, 0).to(tl.float32)
    if fused_residual:
        branch = tl.load(Branch + offset, mask, 0).to(tl.float32)
        gate = tl.load(Gate + index * gate_stride + channel, mask, 0).to(tl.float32)
        gated = (branch * gate).to(tl.bfloat16).to(tl.float32)
        value = (value + gated).to(tl.bfloat16).to(tl.float32)
        tl.store(ResidualOut + offset, value, mask)
    weight = tl.load(Weight + channel, mask, 0).to(tl.float32)
    variance = tl.sum(value * value, 0) / width
    normalized = (value * tl.rsqrt(variance + eps) * weight).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(Scale + index * scale_stride + channel, mask, 0).to(tl.float32)
    shift = tl.load(Shift + index * shift_stride + channel, mask, 0).to(tl.float32)
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    product = (normalized * factor).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + offset, product + shift, mask)


def rms_adaln(x, norm, shift, scale, indices, *, branch=None, gate=None):
    """Return the modulated rows and, for the second norm, the updated residual."""
    weight = norm.weight
    width = x.shape[-1]
    tables = (shift, scale) if gate is None else (shift, scale, gate)
    if not (
        type(norm) is torch.nn.RMSNorm and x.is_cuda and x.dtype == torch.bfloat16
        and x.ndim == 3 and x.numel() and x.is_contiguous()
        and tuple(norm.normalized_shape) == (width,) and norm.eps is not None
        and weight is not None and weight.dtype == x.dtype and weight.is_contiguous()
        and indices.ndim == 1 and indices.numel() == x.shape[1] and indices.is_contiguous()
        and all(table.dtype == x.dtype and table.stride(-1) == 1 for table in tables)
        and (branch is None or (branch.dtype == x.dtype and branch.is_contiguous()))
    ):
        return None
    output = torch.empty_like(x)
    residual = torch.empty_like(x) if branch is not None else x
    _norm_adaln[(x.shape[0] * x.shape[1],)](
        x, weight, shift, scale, indices, x if branch is None else branch,
        scale if gate is None else gate, residual, output, width, norm.eps, x.shape[1],
        shift.stride(0), scale.stride(0), 0 if gate is None else gate.stride(0),
        branch is not None, triton.next_power_of_2(width), num_warps=8, enable_fp_fusion=False,
    )
    return residual, output
