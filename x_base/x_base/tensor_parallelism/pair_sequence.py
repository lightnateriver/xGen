"""Two-rank linear projections around an unchanged eight-rank SP stream."""
import torch
import torch.distributed as dist


class PairSequenceExchange:
    def __init__(self, group, wire='bf16', chunk_elements=4194304):
        if dist.get_world_size(group) != 2:
            raise ValueError('Pair sequence exchange requires exactly two ranks')
        self.group = group
        self.rank = dist.get_rank(group)
        if wire not in ('bf16', 'fp8') or type(chunk_elements) is not int or chunk_elements < 128 or chunk_elements % 128:
            raise ValueError('Pair transport requires BF16/FP8 and aligned bounded packets')
        self.wire, self.chunk_elements = wire, chunk_elements
        self.metrics = dict(tp=2, wire=wire, ranks=dist.get_process_group_ranks(group),
                            gather_calls=0, column_exchange_calls=0, reduce_scatter_calls=0,
                            logical_payload_bytes=0, wire_bytes=0)

    def gather(self, local):
        """Gather the two SP token pieces in their original rank order."""
        rows = local.movedim(1, 0).contiguous()
        with torch.profiler.record_function('h3.tp2_sp8.gather'):
            if self.wire == 'fp8':
                from .fp8_reduce import encode_chunk, decode_chunk
                packet = encode_chunk(rows)
                received = [torch.empty_like(packet), torch.empty_like(packet)]
                dist.all_gather(received, packet, group=self.group)
                full = torch.cat([decode_chunk(p)[:rows.numel()].view_as(rows).to(rows.dtype) for p in received])
                self.metrics['wire_bytes'] += 2*packet.numel()
            else:
                full = torch.empty((rows.shape[0]*2, *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
                dist.all_gather_into_tensor(full, rows, group=self.group)
                self.metrics['wire_bytes'] += full.numel()*full.element_size()
        self.metrics['gather_calls'] += 1
        self.metrics['logical_payload_bytes'] += full.numel()*full.element_size()
        return full.movedim(0, 1).contiguous()

    def scatter_columns(self, value, parts):
        """[B,2L,parts*D] -> [B,L,parts*2D], preserving fused Q/K/V order."""
        batch, tokens, width = value.shape
        if tokens % 2 or width % parts:
            raise ValueError('Pair column layout must divide token owners and fused parts')
        slots, channels = tokens//2, width//parts
        send = value.reshape(batch, 2, slots, parts, channels).permute(1, 2, 0, 3, 4).contiguous()
        received = torch.empty_like(send)
        with torch.profiler.record_function('h3.tp2_sp8.column_to_sp'):
            if self.wire == 'bf16':
                dist.all_to_all_single(received, send, group=self.group)
                self.metrics['wire_bytes'] += send.numel()*send.element_size()
            else:
                from .fp8_reduce import encode_chunk, decode_chunk
                packets = torch.stack([encode_chunk(v) for v in send])
                incoming = torch.empty_like(packets)
                dist.all_to_all_single(incoming, packets, group=self.group)
                self.metrics['wire_bytes'] += packets.numel()
                for i, packet in enumerate(incoming):
                    received[i].copy_(decode_chunk(packet)[:send[i].numel()].view_as(send[i]))
        self.metrics['column_exchange_calls'] += 1
        self.metrics['logical_payload_bytes'] += send.numel()*send.element_size()
        return received.permute(2, 1, 3, 0, 4).reshape(batch, slots, parts*2*channels).contiguous()

    def reduce_scatter(self, partial):
        """Sum the two row-GEMM partials, restoring each rank's SP token piece."""
        rows = partial.movedim(1, 0).contiguous()
        if rows.shape[0] % 2:
            raise ValueError('Pair row output must contain two equal token pieces')
        result = torch.empty((rows.shape[0]//2, *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
        with torch.profiler.record_function('h3.tp2_sp8.reduce_scatter'):
            if self.wire == 'bf16':
                dist.reduce_scatter_tensor(result, rows, group=self.group)
                self.metrics['wire_bytes'] += rows.numel()*rows.element_size()
            else:
                from .fp8_reduce import encode_chunk, decode_chunk
                owners = rows.reshape(2, -1)
                flat = result.view(-1)
                for start in range(0, flat.numel(), self.chunk_elements):
                    count = min(self.chunk_elements, flat.numel()-start)
                    packets = torch.stack([encode_chunk(v[start:start+count]) for v in owners])
                    incoming = torch.empty_like(packets)
                    dist.all_to_all_single(incoming, packets, group=self.group)
                    self.metrics['wire_bytes'] += packets.numel()
                    total = torch.zeros(count, dtype=torch.float32, device=rows.device)
                    for packet in incoming:
                        total.add_(decode_chunk(packet)[:count])
                    flat[start:start+count].copy_(total)
        self.metrics['reduce_scatter_calls'] += 1
        self.metrics['logical_payload_bytes'] += rows.numel()*rows.element_size()
        return result.movedim(0, 1).contiguous()
