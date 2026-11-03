# SPDX-License-Identifier: Apache-2.0
"""Normalize strided H3 Q/K projection views directly on CUDA."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rms(q, k, qw, kw, qo, ko, cos, sin, ROWS: tl.constexpr, SEQ: tl.constexpr,
            HEADS: tl.constexpr, WIDTH: tl.constexpr, Q_STRIDE: tl.constexpr,
            K_STRIDE: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr,
            TILE: tl.constexpr, BLOCK: tl.constexpr, ROTARY: tl.constexpr):
    is_key = tl.program_id(1) == 1
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.arange(0, BLOCK)
    batch = rows // (SEQ * HEADS)
    seq = rows // HEADS % SEQ
    head = rows % HEADS
    q_base = batch * Q_STRIDE[0] + seq * Q_STRIDE[1] + head * Q_STRIDE[2]
    k_base = batch * K_STRIDE[0] + seq * K_STRIDE[1] + head * K_STRIDE[2]
    base = tl.where(is_key, k_base, q_base)
    source = tl.where(is_key, k, q)
    weight = tl.where(is_key, kw, qw)
    target = tl.where(is_key, ko, qo)
    mask = (rows[:, None] < ROWS) & (col[None, :] < WIDTH)
    value = tl.load(source + base[:, None] + col[None, :], mask, other=0).to(tl.float32)
    w = tl.load(weight + col, col < WIDTH, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(value * value, 1) / WIDTH + tl.where(is_key, K_EPS, Q_EPS))
    output = (value * scale[:, None] * w[None, :]).to(tl.bfloat16)
    if ROTARY > 0:
        half = ROTARY // 2
        first = col < half
        paired_col = col + tl.where(first, half, -half)
        active = (rows[:, None] < ROWS) & (col[None, :] < ROTARY)
        partner = tl.load(source + base[:, None] + paired_col[None, :], active, other=0).to(tl.float32)
        partner_weight = tl.load(weight + paired_col, col < ROTARY, other=0).to(tl.float32)
        partner = (partner * scale[:, None] * partner_weight[None, :]).to(tl.bfloat16).to(tl.float32)
        freq = seq[:, None] * ROTARY + col[None, :]
        c = tl.load(cos + freq, active, other=0).to(tl.bfloat16).to(tl.float32)
        s = tl.load(sin + freq, active, other=0).to(tl.bfloat16).to(tl.float32)
        left = (output.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
        right = (tl.where(first[None, :], -partner, partner) * s).to(tl.bfloat16).to(tl.float32)
        output = tl.where(col[None, :] < ROTARY, (left + right).to(tl.bfloat16), output)
    tl.store(target + rows[:, None] * WIDTH + col[None, :], output, mask)


def try_h3_qk_rms(query, key, norm_q, norm_k, rotary_emb=None):
    if (
        torch.is_grad_enabled() or not query.is_cuda or query.ndim != 4 or query.numel() == 0
        or query.shape != key.shape or query.dtype != torch.bfloat16 or key.dtype != query.dtype
        or query.stride(-1) != 1 or key.stride(-1) != 1 or query.device != key.device
        or type(norm_q) is not torch.nn.RMSNorm or type(norm_k) is not torch.nn.RMSNorm
        or norm_q.eps is None or norm_k.eps is None
    ):
        return None
    _, seq, heads, width = query.shape
    rotary = 0
    cos = sin = query
    if rotary_emb is not None:
        cos, sin = rotary_emb
        if (
            cos.ndim != 2 or cos.shape != sin.shape or cos.shape[0] != seq
            or not 0 < cos.shape[-1] <= width or cos.shape[-1] % 2
            or any(t.device != query.device or not t.is_contiguous() for t in (cos, sin))
        ):
            return None
        rotary = cos.shape[-1]
    weights = (norm_q.weight, norm_k.weight)
    if any(
        w is None or w.shape != (width,) or w.dtype != query.dtype
        or w.device != query.device or not w.is_contiguous()
        for w in weights
    ):
        return None
    q_out = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    k_out = torch.empty_like(q_out)
    rows = query.numel() // width
    _qk_rms[(triton.cdiv(rows, 4), 2)](
        query, key, *weights, q_out, k_out, cos, sin, rows, seq, heads, width,
        query.stride(), key.stride(), norm_q.eps, norm_k.eps, 4, triton.next_power_of_2(width),
        rotary, enable_fp_fusion=False,
    )
    return q_out, k_out
