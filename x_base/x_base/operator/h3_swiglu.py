# SPDX-License-Identifier: Apache-2.0
"""CUDA packed gate-first SwiGLU with eager BF16 intermediate rounding."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _swiglu(x, out, peaks, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
             COMPUTE_PEAKS: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    source = offset + offset // WIDTH * WIDTH
    gate = tl.load(x + source, mask, other=0).to(tl.float32)
    value = tl.load(x + source + WIDTH, mask, other=0).to(tl.float32)
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    result = (activated * value).to(tl.bfloat16)
    tl.store(out + offset, result, mask)
    if COMPUTE_PEAKS:
        peak = tl.reduce(tl.abs(result.to(tl.float32)), 0, _maximum)
        tl.store(peaks + tl.program_id(0), peak)


def gate_first_swiglu(x, *, with_amax=False):
    width = x.shape[-1] // 2
    output = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    blocks = triton.cdiv(output.numel(), 1024)
    peaks = torch.empty((blocks,), device=x.device, dtype=torch.float32) if with_amax else output
    _swiglu[(blocks,)](
        x, output, peaks, output.numel(), width, 1024, with_amax, enable_fp_fusion=False,
    )
    if with_amax:
        return output, peaks.amax()
    return output
