# SPDX-License-Identifier: Apache-2.0
"""CUDA H3 indexed modulation and gated residual with eager BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_affine(
    x, table, addend, indices, output, N: tl.constexpr, WIDTH: tl.constexpr,
    SEQ: tl.constexpr, TABLE_STRIDE: tl.constexpr, ADD_STRIDE: tl.constexpr,
    MODULATION: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    channel = offset % WIDTH
    row = tl.load(indices + (offset // WIDTH) % SEQ, mask, other=0).to(tl.int32)
    factor = tl.load(table + row * TABLE_STRIDE + channel, mask, other=0).to(tl.float32)
    if MODULATION:
        factor = (1.0 + factor).to(tl.bfloat16).to(tl.float32)
        add_offset = row * ADD_STRIDE + channel
    else:
        add_offset = offset
    value = tl.load(x + offset, mask, other=0).to(tl.float32)
    bias = tl.load(addend + add_offset, mask, other=0).to(tl.float32)
    # Eager produces BF16 intermediates after each elementwise operation.
    product = (value * factor).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offset, product + bias, mask)


def _apply(x, table, addend, indices, *, modulation):
    if (
        x.is_cuda and x.ndim == 3 and x.is_contiguous() and indices.is_contiguous()
        and indices.ndim == 1 and indices.numel() == x.shape[1]
        and x.dtype == table.dtype == addend.dtype == torch.bfloat16
        and table.stride(-1) == 1
        and (addend.stride(-1) == 1 if modulation else addend.is_contiguous())
    ):
        output = torch.empty_like(x)
        _indexed_affine[(triton.cdiv(x.numel(), 1024),)](
            x, table, addend, indices, output, x.numel(), x.shape[-1], x.shape[1],
            table.stride(0), addend.stride(0), modulation, 1024, enable_fp_fusion=False,
        )
        return output
    factor = table.index_select(0, indices)
    if modulation:
        return x * (1.0 + factor) + addend.index_select(0, indices)
    return addend + factor * x


def fused_h3_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    if not self._h3_fused_indexed_affine or torch.is_grad_enabled():
        return self._h3_eager_forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = self.adaln_proj(temb)
    normed = _apply(self.norm1(hidden_states), scale_attn, shift_attn, adaln_indices, modulation=True)
    attn = self.attn(normed, rotary_emb, attention_mask)
    hidden_states = _apply(attn, gate_attn, hidden_states, adaln_indices, modulation=False)
    normed = _apply(self.norm2(hidden_states), scale_ff, shift_ff, adaln_indices, modulation=True)
    ff = self.ff(normed)
    return _apply(ff, gate_ff, hidden_states, adaln_indices, modulation=False)
