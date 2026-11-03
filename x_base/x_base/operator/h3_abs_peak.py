# SPDX-License-Identifier: Apache-2.0
"""Absolute BF16 peak reduction without a full activation temporary."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _abs_peak(x, peaks, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offset, offset < N, other=0).to(tl.float32)
    tl.store(peaks + tl.program_id(0), tl.reduce(tl.abs(values), 0, _maximum))


def activation_abs_peak(x):
    blocks = triton.cdiv(x.numel(), 8192)
    peaks = torch.empty((blocks,), device=x.device, dtype=torch.float32)
    _abs_peak[(blocks,)](x, peaks, x.numel(), 8192)
    return peaks.amax()
