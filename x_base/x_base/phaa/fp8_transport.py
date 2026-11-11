# SPDX-License-Identifier: Apache-2.0
"""Per-head FP8 transport for CUDA PHAA all-to-all."""

import logging
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_mixed_stride_metrics = dict(calls=0, avoided_contiguous_bytes=0)
_peer_alignment_metrics = dict(calls=0, padded_calls=0, padding_bytes=0)


def mixed_stride_metrics():
    """Scalar diagnostics only; no tensor or communication-buffer cache."""
    return dict(_mixed_stride_metrics)


def peer_alignment_metrics():
    """Scalar counters for the opt-in peer-aligned wire layout."""
    return dict(_peer_alignment_metrics)


@triton.jit
def _encode(q, k, v, payload, scale_ptr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            PEER_ROWS: tl.constexpr, STREAMS: tl.constexpr,
            SHAPE: tl.constexpr, STRIDES: tl.constexpr, STRIDED: tl.constexpr,
            MIXED_STRIDES: tl.constexpr = False, PEER_PADDING: tl.constexpr = 0):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    src = tl.where(slot == 0, q, tl.where(slot == 1, k, v)) if STREAMS == 3 else q
    col = tl.arange(0, BLOCK)
    if STRIDED:
        # PHAA QKV view is [peer, sequence, batch, head, head_dim].
        head = row % SHAPE[3]
        batch = row // SHAPE[3] % SHAPE[2]
        seq = row // (SHAPE[3] * SHAPE[2]) % SHAPE[1]
        peer = row // PEER_ROWS
        if MIXED_STRIDES:
            s0 = tl.where(slot == 0, STRIDES[0][0], tl.where(slot == 1, STRIDES[1][0], STRIDES[2][0]))
            s1 = tl.where(slot == 0, STRIDES[0][1], tl.where(slot == 1, STRIDES[1][1], STRIDES[2][1]))
            s2 = tl.where(slot == 0, STRIDES[0][2], tl.where(slot == 1, STRIDES[1][2], STRIDES[2][2]))
            s3 = tl.where(slot == 0, STRIDES[0][3], tl.where(slot == 1, STRIDES[1][3], STRIDES[2][3]))
            src_row = peer * s0 + seq * s1 + batch * s2 + head * s3
        else:
            src_row = peer * STRIDES[0] + seq * STRIDES[1] + batch * STRIDES[2] + head * STRIDES[3]
    else:
        src_row = row * WIDTH
    values = tl.load(src + src_row + col, col < WIDTH, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 448.0, 1.0e-12)
    wire_row = row + (row // PEER_ROWS * (STREAMS - 1) + slot) * PEER_ROWS
    peer = row // PEER_ROWS
    offset = wire_row * (WIDTH + 4) + peer * PEER_PADDING
    tl.store(payload + offset + col, tl.clamp(values / scale, -448.0, 448.0), col < WIDTH)
    tl.store(scale_ptr + (offset + WIDTH) // 4, scale)
    if PEER_PADDING:
        if (row % PEER_ROWS == 0) & (slot == 0):
            tail = peer * (PEER_ROWS * STREAMS * (WIDTH + 4) + PEER_PADDING)
            tail += PEER_ROWS * STREAMS * (WIDTH + 4)
            padding_col = tl.arange(0, 16)
            tl.store(payload + tail + padding_col, 0.0, padding_col < PEER_PADDING)


@triton.jit
def _decode(payload, scale_ptr, q, k, v, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            PEER_ROWS: tl.constexpr, STREAMS: tl.constexpr, PEER_PADDING: tl.constexpr = 0):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    dst = tl.where(slot == 0, q, tl.where(slot == 1, k, v)) if STREAMS == 3 else q
    col = tl.arange(0, BLOCK)
    wire_row = row + (row // PEER_ROWS * (STREAMS - 1) + slot) * PEER_ROWS
    offset = wire_row * (WIDTH + 4) + (row // PEER_ROWS) * PEER_PADDING
    scale = tl.load(scale_ptr + (offset + WIDTH) // 4)
    bits = tl.load(payload + offset + col, col < WIDTH, other=0)
    values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    tl.store(dst + row * WIDTH + col, values * scale, col < WIDTH)


def fp8_all_to_all(output, tensor, head_dim, group):
    fp8_all_to_all_many((output,), (tensor,), head_dim, group)


def fp8_all_to_all_many(outputs, tensors, head_dim, group):
    """Exchange equal-shaped tensors and their row scales in one byte collective."""
    streams = len(tensors)
    if streams not in (1, 3) or len(outputs) != streams:
        raise ValueError("PHAA FP8 transport expects one tensor or a QKV triple")
    if any(tensor.device.type != "cuda" for tensor in (*outputs, *tensors)):
        raise ValueError("PHAA FP8 transport requires CUDA tensors")
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
    block = triton.next_power_of_2(head_dim)
    peer_bytes = peer_rows * streams * (head_dim + 4)
    align_peers = os.environ.get("XGEN_PHAA_FP8_ALIGN16", "0") == "1"
    peer_padding = (-peer_bytes) % 16 if align_peers else 0
    if align_peers:
        if _peer_alignment_metrics['calls'] == 0:
            logging.getLogger(__name__).info("PHAA FP8 per-peer 16-byte alignment enabled")
        _peer_alignment_metrics['calls'] += 1
        _peer_alignment_metrics['padded_calls'] += int(peer_padding > 0)
        _peer_alignment_metrics['padding_bytes'] += peers * peer_padding
    wire_shape = (peers, peer_bytes + peer_padding) if peer_padding else (streams * rows, head_dim + 4)
    send = torch.empty(wire_shape, dtype=torch.uint8, device=tensors[0].device)
    recv = torch.empty_like(send)
    strides = tensors[0].stride()
    mixed_strides = False
    if (streams == 3 and len(shape) == 5 and shape[0] == peers and shape[-1] == head_dim
            and strides[-1] == 1 and all(x.stride() == strides for x in tensors)):
        inputs = tensors
    elif (os.environ.get("XGEN_PHAA_FP8_MIXED_STRIDES", "0") == "1"
          and streams == 3 and len(shape) == 5 and shape[0] == peers
          and shape[-1] == head_dim and all(x.stride(-1) == 1 for x in tensors)):
        inputs = tensors
        strides = tuple(x.stride() for x in tensors)
        mixed_strides = True
        if _mixed_stride_metrics['calls'] == 0:
            logging.getLogger(__name__).info(
                "PHAA FP8 mixed-stride packing enabled: shape=%s strides=%s", tuple(shape), strides,
            )
        _mixed_stride_metrics['calls'] += 1
        _mixed_stride_metrics['avoided_contiguous_bytes'] += sum(
            x.numel() * x.element_size() for x in tensors if not x.is_contiguous()
        )
    else:
        inputs = tuple(tensor.contiguous() for tensor in tensors)
        strides = ()
    _encode[(rows, streams)](*(inputs if streams == 3 else inputs * 3),
                             send.view(torch.float8_e4m3fn), send.view(torch.float32),
                             head_dim, block, peer_rows, streams, tuple(shape), strides, bool(strides),
                             mixed_strides, peer_padding)
    dist.all_to_all_single(recv, send, group=group)
    _decode[(rows, streams)](recv, recv.view(torch.float32),
                             *(outputs if streams == 3 else outputs * 3),
                             head_dim, block, peer_rows, streams, peer_padding)
