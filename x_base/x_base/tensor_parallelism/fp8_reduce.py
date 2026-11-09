"""Experimental, bounded E4M3 wire reduction; never reduce packed bytes.

Every peer (including self) is decoded and accumulated in group-rank order.
The output dtype is unchanged. FP32 summation does not remove FP8 wire error.
"""
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def encode_chunk(values, width=128):
    flat = values.reshape(-1)
    padded = F.pad(flat, (0, (-flat.numel()) % width)).reshape(-1, width)
    scale = (padded.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12)
    data = (padded.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return torch.cat((data.view(torch.uint8), scale.view(torch.uint8)), -1).contiguous()


def decode_chunk(packet, width=128):
    data = packet[:, :width].contiguous().view(torch.float8_e4m3fn).float()
    scale = packet[:, width:].contiguous().view(torch.float32)
    return (data * scale).reshape(-1)


class BF16Reduction:
    """Observable control path using the existing BF16 all-reduce."""

    def __init__(self):
        self.metrics = dict(calls=0, logical_payload_bytes=0, exchange_host_seconds=0.)

    def __call__(self, partial, group):
        if partial.dtype != torch.bfloat16:
            raise ValueError('H3 BF16 TP transport requires BF16 partials')
        started = time.perf_counter()
        with torch.profiler.record_function('h3.tp.bf16.all_reduce'):
            dist.all_reduce(partial, group=group)
        self.metrics['exchange_host_seconds'] += time.perf_counter() - started
        self.metrics['calls'] += 1
        self.metrics['logical_payload_bytes'] += partial.numel()*partial.element_size()
        return partial


class FP8Reduction:
    """Scale-aware all-gather plus FP32 sum, with bounded scratch per call.

    Metrics are host enqueue envelopes (not synchronized GPU elapsed time).
    Use profiler ranges to measure device/collective latency without adding
    synchronization to each block. ``wire_bytes`` includes self in all-gather.
    """

    def __init__(self, chunk_elements=262144):
        if type(chunk_elements) is not int or chunk_elements < 128 or chunk_elements % 128:
            raise ValueError('FP8 TP chunk_elements must be a positive multiple of 128')
        self.chunk_elements = chunk_elements
        self.metrics = dict(calls=0, chunks=0, wire_bytes=0, max_packet_bytes=0,
                            pack_host_seconds=0., exchange_host_seconds=0., reduce_host_seconds=0.)

    def __call__(self, partial, group):
        if partial.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError('FP8 TP transport requires BF16/FP16 partials; FP32 heads stay FP32')
        peers = dist.get_world_size(group)
        if peers == 1:
            return partial
        if torch.is_grad_enabled() and partial.requires_grad:
            raise ValueError('H3 FP8 TP transport is inference-only')
        flat = partial.contiguous().view(-1)
        result = torch.empty_like(flat)
        for start in range(0, flat.numel(), self.chunk_elements):
            count = min(self.chunk_elements, flat.numel() - start)
            t = time.perf_counter()
            with torch.profiler.record_function('h3.tp.fp8.pack'):
                packet = encode_chunk(flat[start:start+count])
            self.metrics['pack_host_seconds'] += time.perf_counter() - t
            received = [torch.empty_like(packet) for _ in range(peers)]
            t = time.perf_counter()
            with torch.profiler.record_function('h3.tp.fp8.exchange'):
                dist.all_gather(received, packet, group=group)
            self.metrics['exchange_host_seconds'] += time.perf_counter() - t
            t = time.perf_counter()
            with torch.profiler.record_function('h3.tp.fp8.fp32_sum'):
                total = torch.zeros(count, dtype=torch.float32, device=partial.device)
                for peer in received:
                    total.add_(decode_chunk(peer)[:count])
                result[start:start+count] = total
            self.metrics['reduce_host_seconds'] += time.perf_counter() - t
            size = packet.numel()
            self.metrics['chunks'] += 1
            self.metrics['wire_bytes'] += size * peers
            self.metrics['max_packet_bytes'] = max(self.metrics['max_packet_bytes'], size)
        self.metrics['calls'] += 1
        return result.view_as(partial)
