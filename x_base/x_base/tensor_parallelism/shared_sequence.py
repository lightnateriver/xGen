"""Inference-only sequence ownership shared with tensor-parallel projections.

Both groups have the same ordered ranks, rather than forming a TP x CP mesh.
Reuse the existing E4M3 packet format; never sum quantized bytes.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .fp8_reduce import encode_chunk, decode_chunk


class SharedSequenceExchange:
    def __init__(self, cp_group, tp_group, wire='bf16', chunk_elements=4194304):
        if wire not in ('bf16', 'fp8'):
            raise ValueError('Shared TP/CP transport must be bf16 or fp8')
        if type(chunk_elements) is not int or chunk_elements < 128 or chunk_elements % 128:
            raise ValueError('Shared TP/CP packet size must be a positive multiple of 128')
        cp_ranks = dist.get_process_group_ranks(cp_group)
        tp_ranks = dist.get_process_group_ranks(tp_group)
        if cp_ranks != tp_ranks:
            raise ValueError('Shared TP/CP groups must contain the same ordered ranks')
        self.cp_group, self.tp_group = cp_group, tp_group
        self.peers = len(cp_ranks)
        self.rank = dist.get_rank(cp_group)
        self.wire, self.chunk_elements = wire, chunk_elements
        self.tokens = None
        self.metrics = dict(wire=wire, tp=self.peers, cp=self.peers, shared_ranks=cp_ranks,
                            gather_calls=0, reduce_scatter_calls=0, attention_calls=0,
                            gather_wire_bytes=0, reduction_wire_bytes=0,
                            reduction_chunks=0, max_packet_bytes=0,
                            attention_layout='ulysses_head_shard',
                            qkv_all_to_all='fused_with_TP_projection')

    def set_tokens(self, tokens):
        if type(tokens) is not int or tokens < 1:
            raise ValueError('Shared TP/CP requires a positive real sequence length')
        self.tokens = tokens

    def _check(self, value):
        if self.tokens is None or value.ndim != 3 or not value.is_floating_point():
            raise ValueError('Shared TP/CP requires initialized batch/sequence/channel tensors')
        if torch.is_grad_enabled() and value.requires_grad:
            raise ValueError('Shared TP/CP transport is inference-only')

    def gather(self, local):
        """CP row shard -> replicated input for the TP column GEMM."""
        self._check(local)
        slots = (self.tokens+self.peers-1)//self.peers
        if local.shape[1] != slots:
            raise ValueError('CP input rows do not match the padded sequence ownership')
        sequence = local.movedim(1, 0).contiguous()
        self.metrics['gather_calls'] += 1
        with torch.profiler.record_function('h3.shared_tp_cp.gather.'+self.wire):
            if self.wire == 'bf16':
                received = torch.empty((slots*self.peers, *sequence.shape[1:]),
                                       dtype=local.dtype, device=local.device)
                dist.all_gather_into_tensor(received, sequence, group=self.cp_group)
                self.metrics['gather_wire_bytes'] += sequence.numel()*sequence.element_size()*self.peers
            else:
                packet = encode_chunk(sequence)
                received_packets = torch.empty((packet.shape[0]*self.peers, packet.shape[1]),
                                               dtype=packet.dtype, device=packet.device)
                dist.all_gather_into_tensor(received_packets, packet, group=self.cp_group)
                packets = received_packets.view(self.peers, *packet.shape)
                received = torch.empty((slots*self.peers, *sequence.shape[1:]),
                                       dtype=local.dtype, device=local.device)
                for rank, incoming in enumerate(packets):
                    received[rank*slots:(rank+1)*slots].copy_(
                        decode_chunk(incoming)[:sequence.numel()].view_as(sequence))
                self.metrics['gather_wire_bytes'] += packet.numel()*self.peers
        return received[:self.tokens].movedim(0, 1).contiguous()

    def reduce_scatter(self, partial, group):
        """TP row-GEMM partials -> sum and retain only this CP rank's rows."""
        self._check(partial)
        if group is not self.tp_group or partial.shape[1] != self.tokens:
            raise ValueError('TP partial rows/group do not match shared CP ownership')
        slots = (self.tokens+self.peers-1)//self.peers
        sequence = partial.movedim(1, 0).contiguous()
        sequence = F.pad(sequence, (0, 0, 0, 0, 0, slots*self.peers-self.tokens))
        self.metrics['reduce_scatter_calls'] += 1
        with torch.profiler.record_function('h3.shared_tp_cp.reduce_scatter.'+self.wire):
            if self.wire == 'bf16':
                result = torch.empty((slots, *sequence.shape[1:]),
                                     dtype=partial.dtype, device=partial.device)
                dist.reduce_scatter_tensor(result, sequence, group=self.tp_group)
                self.metrics['reduction_wire_bytes'] += sequence.numel()*sequence.element_size()
            else:
                owners = sequence.reshape(self.peers, -1)
                result = torch.empty(owners.shape[1], dtype=partial.dtype, device=partial.device)
                for start in range(0, owners.shape[1], self.chunk_elements):
                    count = min(self.chunk_elements, owners.shape[1]-start)
                    packets = torch.stack([encode_chunk(v[start:start+count]) for v in owners])
                    received = torch.empty_like(packets)
                    dist.all_to_all_single(received, packets, group=self.tp_group)
                    total = torch.zeros(count, dtype=torch.float32, device=partial.device)
                    for incoming in received:
                        total.add_(decode_chunk(incoming)[:count])
                    result[start:start+count].copy_(total)
                    self.metrics['reduction_chunks'] += 1
                    self.metrics['reduction_wire_bytes'] += packets.numel()
                    self.metrics['max_packet_bytes'] = max(self.metrics['max_packet_bytes'], packets[0].numel())
                result = result.view(slots, *sequence.shape[1:])
        return result.movedim(0, 1).contiguous()
