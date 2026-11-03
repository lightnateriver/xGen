# SPDX-License-Identifier: Apache-2.0
"""Per-head FP8 transport for CUDA PHAA all-to-all."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _encode(src, payload, scale_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            PEER_ROWS: tl.constexpr, STREAMS: tl.constexpr, SLOT: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(src + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 448.0, 1.0e-12)
    wire_row = row + (row // PEER_ROWS * (STREAMS - 1) + SLOT) * PEER_ROWS
    offset = wire_row * (WIDTH + 4)
    tl.store(payload + offset + col, tl.clamp(values / scale, -448.0, 448.0), col < WIDTH)
    tl.store(scale_ptr + (offset + WIDTH) // 4, scale)


@triton.jit
def _decode(payload, scale_ptr, dst, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            PEER_ROWS: tl.constexpr, STREAMS: tl.constexpr, SLOT: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    wire_row = row + (row // PEER_ROWS * (STREAMS - 1) + SLOT) * PEER_ROWS
    offset = wire_row * (WIDTH + 4)
    scale = tl.load(scale_ptr + (offset + WIDTH) // 4)
    bits = tl.load(payload + offset + col, col < WIDTH, other=0)
    values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    tl.store(dst + row * WIDTH + col, values * scale, col < WIDTH)


def fp8_all_to_all(output, tensor, head_dim, group):
    fp8_all_to_all_many((output,), (tensor,), head_dim, group)


def fp8_all_to_all_many(outputs, tensors, head_dim, group):
    """Exchange equal-shaped tensors and their row scales in one byte collective."""
    shape = tensors[0].shape
    if head_dim % 4 or any(x.shape != shape for x in (*outputs, *tensors)):
        raise ValueError("PHAA FP8 transport requires matching shapes and head dim divisible by four")
    if any(not out.is_contiguous() for out in outputs):
        raise ValueError("PHAA FP8 transport requires contiguous outputs")
    rows = tensors[0].numel() // head_dim
    peers = dist.get_world_size(group)
    if rows % peers or tensors[0].numel() % head_dim:
        raise ValueError("PHAA FP8 rows must divide evenly among peers")
    peer_rows = rows // peers
    streams = len(tensors)
    block = triton.next_power_of_2(head_dim)
    send = torch.empty((streams * rows, head_dim + 4), dtype=torch.uint8, device=tensors[0].device)
    recv = torch.empty_like(send)
    for slot, tensor in enumerate(tensors):
        _encode[(rows,)](tensor.contiguous(), send.view(torch.float8_e4m3fn), send.view(torch.float32),
                         head_dim, block, peer_rows, streams, slot)
    dist.all_to_all_single(recv, send, group=group)
    for slot, output in enumerate(outputs):
        _decode[(rows,)](recv, recv.view(torch.float32), output, head_dim, block, peer_rows, streams, slot)
