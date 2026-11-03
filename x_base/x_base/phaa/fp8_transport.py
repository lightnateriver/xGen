# SPDX-License-Identifier: Apache-2.0
"""Per-head FP8 transport for CUDA PHAA all-to-all."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _encode(src, payload, scale_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(src + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 448.0, 1.0e-12)
    offset = row * (WIDTH + 4)
    tl.store(payload + offset + col, tl.clamp(values / scale, -448.0, 448.0), col < WIDTH)
    tl.store(scale_ptr + (offset + WIDTH) // 4, scale)


@triton.jit
def _decode(payload, scale_ptr, dst, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = row * (WIDTH + 4)
    scale = tl.load(scale_ptr + (offset + WIDTH) // 4)
    bits = tl.load(payload + offset + col, col < WIDTH, other=0)
    values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    tl.store(dst + row * WIDTH + col, values * scale, col < WIDTH)


def fp8_all_to_all(output, tensor, head_dim, group):
    """Exchange FP8 data and FP32 row scales in a single byte collective."""
    if head_dim % 4 or not output.is_contiguous():
        raise ValueError("PHAA FP8 transport requires contiguous output and head dim divisible by four")
    tensor = tensor.contiguous()
    rows = tensor.numel() // head_dim
    block = triton.next_power_of_2(head_dim)
    send = torch.empty((rows, head_dim + 4), dtype=torch.uint8, device=tensor.device)
    recv = torch.empty_like(send)
    _encode[(rows,)](tensor, send.view(torch.float8_e4m3fn), send.view(torch.float32), head_dim, block)
    dist.all_to_all_single(recv, send, group=group)
    _decode[(rows,)](recv, recv.view(torch.float32), output, head_dim, block)
