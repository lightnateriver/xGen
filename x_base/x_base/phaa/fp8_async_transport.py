# SPDX-License-Identifier: Apache-2.0
"""Experimental asynchronous FP8 transport and direct layouts for CUDA PHAA."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _encode(q, k, v, payload, scale_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            PEER_ROWS: tl.constexpr, STREAMS: tl.constexpr,
            SHAPE: tl.constexpr, STRIDES: tl.constexpr, STRIDED: tl.constexpr,
            ROWS: tl.constexpr, TILE: tl.constexpr, RETURN_SEQ: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    slot = tl.program_id(1)
    src = tl.where(slot == 0, q, tl.where(slot == 1, k, v)) if STREAMS == 3 else q
    col = tl.arange(0, BLOCK)
    if RETURN_SEQ > 0:
        # Wire order remains [peer, local sequence, batch, head, channel].
        head = row % SHAPE[3]
        batch = row // SHAPE[3] % SHAPE[2]
        seq = row // (SHAPE[3] * SHAPE[2])
        src_row = batch * STRIDES[0] + head * STRIDES[1] + seq * STRIDES[2]
        src_col = col * STRIDES[3]
    elif STRIDED:
        # PHAA QKV view is [peer, sequence, batch, head, head_dim].
        head = row % SHAPE[3]
        batch = row // SHAPE[3] % SHAPE[2]
        seq = row // (SHAPE[3] * SHAPE[2]) % SHAPE[1]
        peer = row // PEER_ROWS
        src_row = peer * STRIDES[0] + seq * STRIDES[1] + batch * STRIDES[2] + head * STRIDES[3]
        src_col = col
    else:
        src_row = row * WIDTH
        src_col = col
    mask = (row[:, None] < ROWS) & (col[None, :] < WIDTH)
    load_mask = mask
    if RETURN_SEQ > 0:
        load_mask = mask & (seq[:, None] < RETURN_SEQ)
    values = tl.load(src + src_row[:, None] + src_col[None, :], load_mask, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 1) / 448.0, 1.0e-12)
    wire_row = row + (row // PEER_ROWS * (STREAMS - 1) + slot) * PEER_ROWS
    offset = wire_row * (WIDTH + 4)
    tl.store(payload + offset[:, None] + col[None, :], tl.clamp(values / scale[:, None], -448.0, 448.0), mask)
    tl.store(scale_ptr + (offset + WIDTH) // 4, scale, row < ROWS)


@triton.jit
def _decode(payload, scale_ptr, q, k, v, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            PEER_ROWS: tl.constexpr, STREAMS: tl.constexpr,
            SQ: tl.constexpr, BATCHES: tl.constexpr, PEERS: tl.constexpr,
            HEADS_PER_PEER: tl.constexpr, CHUNK_HEADS: tl.constexpr, HEAD_START: tl.constexpr,
            ROWS: tl.constexpr, TILE: tl.constexpr, GATHER_SEQ: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    slot = tl.program_id(1)
    dst = tl.where(slot == 0, q, tl.where(slot == 1, k, v)) if STREAMS == 3 else q
    col = tl.arange(0, BLOCK)
    wire_row = row + (row // PEER_ROWS * (STREAMS - 1) + slot) * PEER_ROWS
    offset = wire_row * (WIDTH + 4)
    mask = (row[:, None] < ROWS) & (col[None, :] < WIDTH)
    scale = tl.load(scale_ptr + (offset + WIDTH) // 4, row < ROWS, other=0)
    bits = tl.load(payload + offset[:, None] + col[None, :], mask, other=0)
    values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    dst_row = row
    if GATHER_SEQ > 0:
        head = row % CHUNK_HEADS
        batch = row // CHUNK_HEADS % BATCHES
        seq = row // (CHUNK_HEADS * BATCHES)
        dst_row = (batch * CHUNK_HEADS + head) * GATHER_SEQ + seq
        mask = mask & (seq[:, None] < GATHER_SEQ)
    elif CHUNK_HEADS > 0:
        head = row % CHUNK_HEADS
        batch = row // CHUNK_HEADS % BATCHES
        seq = row // (CHUNK_HEADS * BATCHES) % SQ
        peer = row // PEER_ROWS
        dst_row = ((batch * SQ + seq) * PEERS + peer) * HEADS_PER_PEER + HEAD_START + head
    tl.store(dst + dst_row[:, None] * WIDTH + col[None, :], values * scale[:, None], mask)


@dataclass(frozen=True)
class AsyncFP8Options:
    direct_return: bool = False
    row_tiling: bool = False
    fused_return_pack: bool = False
    direct_gather: bool = False

    def __post_init__(self):
        if self.fused_return_pack and not self.direct_return:
            raise ValueError("PHAA fused return pack requires direct return")


class PendingFP8Exchange(NamedTuple):
    work: dist.Work
    send: torch.Tensor
    recv: torch.Tensor
    outputs: tuple | None
    decode_args: tuple


def submit_fp8_exchange(
    outputs, tensors, head_dim, group, *, row_tiling=False, return_shard_seq=0, direct_gather=False,
):
    """Pack and enqueue a collective on the active CUDA stream."""
    send, recv, decode_args = _pack_exchange(
        outputs, tensors, head_dim, group, row_tiling=row_tiling,
        return_shard_seq=return_shard_seq, direct_gather=direct_gather,
    )
    work = dist.all_to_all_single(recv, send, group=group, async_op=True)
    return PendingFP8Exchange(work, send, recv, outputs, decode_args)


def finish_fp8_exchange(pending, *, output=None, return_layout=()):
    """Wait on the consuming CUDA stream and unpack the received bytes."""
    stream = torch.cuda.current_stream(pending.recv.device)
    pending.work.block_current_stream()
    pending.send.record_stream(stream)
    pending.recv.record_stream(stream)
    outputs = (output,) if return_layout else pending.outputs
    _unpack_exchange(pending.recv, outputs, pending.decode_args, return_layout)


def _pack_exchange(
    outputs, tensors, head_dim, group, *, row_tiling=False, return_shard_seq=0, direct_gather=False,
):
    streams = len(tensors)
    if streams not in (1, 3) or (outputs is None and streams != 1):
        raise ValueError("PHAA FP8 transport expects one tensor or a QKV triple")
    if outputs is not None and len(outputs) != streams:
        raise ValueError("PHAA FP8 transport requires one output per input")
    shape = tensors[0].shape
    if head_dim % 4 or any(x.shape != shape for x in tensors):
        raise ValueError("PHAA FP8 transport requires matching shapes and head dim divisible by four")
    if direct_gather:
        if streams != 3 or len(shape) != 5 or shape[-1] != head_dim:
            raise ValueError("PHAA direct gather expects QKV head shards")
        peers, shard_seq, batches, chunk_heads, _ = shape
        valid_seq = outputs[0].shape[-2]
        expected = (batches, chunk_heads, valid_seq, head_dim)
        if not 0 < valid_seq <= peers * shard_seq or any(out.shape != expected for out in outputs):
            raise ValueError("PHAA direct gather requires matching attention output shapes")
    elif outputs is not None and any(out.shape != shape for out in outputs):
        raise ValueError("PHAA FP8 transport requires matching input and output shapes")
    if outputs is not None and any(not out.is_contiguous() for out in outputs):
        raise ValueError("PHAA FP8 transport requires contiguous outputs")
    peers = dist.get_world_size(group)
    rows = tensors[0].numel() // head_dim
    return_seq = 0
    if return_shard_seq:
        if outputs is not None or streams != 1 or len(shape) != 4 or shape[-1] != head_dim:
            raise ValueError("PHAA return pack expects an attention output and direct return")
        batches, chunk_heads, return_seq, _ = shape
        if not 0 < return_seq <= peers * return_shard_seq:
            raise ValueError("PHAA return pack received an invalid sequence length")
        rows = peers * return_shard_seq * batches * chunk_heads
        shape = (peers, return_shard_seq, batches, chunk_heads, head_dim)
    if rows % peers or tensors[0].numel() % head_dim:
        raise ValueError("PHAA FP8 rows must divide evenly among peers")
    peer_rows = rows // peers
    block = triton.next_power_of_2(head_dim)
    tile = 4 if row_tiling else 1
    send = torch.empty((streams * rows, head_dim + 4), dtype=torch.uint8, device=tensors[0].device)
    recv = torch.empty_like(send)
    strides = tensors[0].stride()
    if return_seq:
        inputs = tensors
    elif (streams == 3 and len(shape) == 5 and shape[0] == peers and shape[-1] == head_dim
            and strides[-1] == 1 and all(x.stride() == strides for x in tensors)):
        inputs = tensors
    else:
        inputs = tuple(tensor.contiguous() for tensor in tensors)
        strides = ()
    _encode[(triton.cdiv(rows, tile), streams)](*(inputs if streams == 3 else inputs * 3),
                             send.view(torch.float8_e4m3fn), send.view(torch.float32),
                             head_dim, block, peer_rows, streams, tuple(shape), strides, bool(strides),
                             rows, tile, return_seq)
    return send, recv, (rows, streams, head_dim, block, peer_rows, tile, direct_gather)


def _unpack_exchange(recv, outputs, decode_args, return_layout=()):
    rows, streams, head_dim, block, peer_rows, tile, direct_gather = decode_args
    gather_seq = 0
    layout = return_layout or (0, 0, 0, 0, 0, 0)
    if direct_gather:
        batches, chunk_heads, gather_seq, _ = outputs[0].shape
        layout = (0, batches, 0, 0, chunk_heads, 0)
    _decode[(triton.cdiv(rows, tile), streams)](recv, recv.view(torch.float32),
                             *(outputs if streams == 3 else outputs * 3),
                             head_dim, block, peer_rows, streams,
                             *layout, rows, tile, gather_seq)
