# SPDX-License-Identifier: Apache-2.0
"""Fuse CUDA H3 partial rotate-half RoPE with eager BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_rope(q, k, cos, sin, q_out, k_out, N: tl.constexpr,
                  SEQ: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
                  ROTARY: tl.constexpr, BLOCK: tl.constexpr):
    is_key = tl.program_id(1) == 1
    source = tl.where(is_key, k, q)
    target = tl.where(is_key, k_out, q_out)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = offset % WIDTH
    seq = offset // (HEADS * WIDTH) % SEQ
    active = (offset < N) & (channel < ROTARY)
    value = tl.load(source + offset, offset < N, other=0).to(tl.float32)
    half = ROTARY // 2
    first = channel < half
    partner = tl.load(source + offset + tl.where(first, half, -half), active, other=0).to(tl.float32)
    freq = seq * ROTARY + channel
    c = tl.load(cos + freq, active, other=0).to(tl.bfloat16).to(tl.float32)
    s = tl.load(sin + freq, active, other=0).to(tl.bfloat16).to(tl.float32)
    left = (value * c).to(tl.bfloat16).to(tl.float32)
    right = (tl.where(first, -partner, partner) * s).to(tl.bfloat16).to(tl.float32)
    tl.store(target + offset, tl.where(channel < ROTARY, left + right, value), offset < N)


def try_h3_partial_rope(query, key, cos, sin):
    if (
        torch.is_grad_enabled() or not query.is_cuda or query.dtype != torch.bfloat16
        or query.ndim != 4 or query.shape != key.shape or key.dtype != query.dtype
        or cos.ndim != 2 or cos.shape != sin.shape or cos.shape[0] != query.shape[1]
        or cos.shape[-1] <= 0 or cos.shape[-1] % 2 or cos.shape[-1] > query.shape[-1]
        or any(t.device != query.device or not t.is_contiguous() for t in (query, key, cos, sin))
    ):
        return None
    _, seq, heads, width = query.shape
    query_out, key_out = torch.empty_like(query), torch.empty_like(key)
    _partial_rope[(triton.cdiv(query.numel(), 1024), 2)](
        query, key, cos, sin, query_out, key_out, query.numel(), seq, heads, width,
        cos.shape[-1], 1024, enable_fp_fusion=False,
    )
    return query_out, key_out
