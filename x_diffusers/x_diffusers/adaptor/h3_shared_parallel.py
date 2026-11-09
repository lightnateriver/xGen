"""H3 layerwise TP/CP on the same ranks, using existing xGen components.

Between sublayers the residual stream is CP sequence-sharded. A column TP
projection consumes gathered rows and produces the exact head-sharded layout
that Ulysses normally obtains through its QKV all-to-all. Fusing these inverse
layout changes avoids sending QKV out and immediately sending it back.
Row projections reduce-scatter their sums back to CP sequence ownership.
"""
import torch
from torch import nn


class GatherColumnInput(nn.Module):
    def __init__(self, projection, exchange):
        super().__init__()
        self.projection, self.exchange = projection, exchange

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, local):
        return self.projection(self.exchange.gather(local))


class SharedTPAttention:
    """Existing dense backend on the CP8/Ulysses internal head partition."""
    def __init__(self, exchange, local_heads, chunks):
        self.exchange, self.local_heads, self.chunks = exchange, local_heads, chunks

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from ..framework.transformer.minimax_h3 import _rope_minimax_h3

        full = self.exchange.gather(hidden_states)
        query, key, value = attn.to_qkv(full).chunk(3, dim=-1)
        del full
        query, key, value = (v.unflatten(-1, (self.local_heads, -1)) for v in (query, key, value))
        normalized = None
        fused_rope = getattr(attn, '_h3_fused_partial_rope', False)
        if getattr(attn, '_h3_fused_qk_rms', False):
            from x_base.operator.h3_qk_rms import try_h3_qk_rms
            normalized = try_h3_qk_rms(query, key, attn.norm_q, attn.norm_k,
                                     rotary_emb if fused_rope else None)
        query, key = (attn.norm_q(query), attn.norm_k(key)) if normalized is None else normalized
        if rotary_emb is not None and not (fused_rope and normalized is not None):
            rotated = None
            if fused_rope:
                from x_base.operator.h3_partial_rope import try_h3_partial_rope
                rotated = try_h3_partial_rope(query, key, *rotary_emb)
            query, key = rotated if rotated is not None else _rope_minimax_h3(query, key, *rotary_emb)
        # Keep the existing PHAA head-chunk cadence (e.g. seven heads -> 4/3),
        # while the TP projections already supply its gathered QKV layout.
        outputs = []
        for q, k, v in zip(torch.tensor_split(query, self.chunks, dim=2),
                           torch.tensor_split(key, self.chunks, dim=2),
                           torch.tensor_split(value, self.chunks, dim=2)):
            outputs.append(attn._h3_attention_backend(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                                                     v.transpose(1, 2).contiguous(), attn_mask=attention_mask))
        result = torch.cat(outputs, dim=1).transpose(1, 2).flatten(2, 3)
        self.exchange.metrics['attention_calls'] += 1
        return attn.to_out[1](attn.to_out[0](result))


def prepare_shared_block(block, manager, exchange, *, fp8=True, chunks=2):
    from .h3_parallel import shard_projection

    attn = block.attn
    if attn.fused_projections or attn.heads % manager.tp_size:
        raise ValueError('Shared TP/CP requires original QKV and divisible attention heads')
    local_heads = attn.heads//manager.tp_size
    if type(chunks) is not int or not 1 <= chunks <= local_heads:
        raise ValueError('Shared TP/CP attention chunk count exceeds local heads')
    originals = (attn.to_q, attn.to_k, attn.to_v)
    biases = [v.bias for v in originals]
    if any(v is None for v in biases) and not all(v is None for v in biases):
        raise ValueError('Shared TP/CP QKV biases must all exist or all be absent')
    merged = nn.Linear(originals[0].in_features, sum(v.out_features for v in originals),
                       bias=biases[0] is not None, device='meta', dtype=originals[0].weight.dtype)
    merged.weight = nn.Parameter(torch.cat([v.weight.detach() for v in originals]), requires_grad=False)
    if biases[0] is not None:
        merged.bias = nn.Parameter(torch.cat(biases), requires_grad=False)
    attn.to_qkv = shard_projection(merged, manager, column=True, parts=3, fp8=fp8)
    attn.to_q = attn.to_k = attn.to_v = None
    attn.fused_projections = True
    attn.to_out[0] = shard_projection(attn.to_out[0], manager, column=False, fp8=fp8,
                                    reduction=exchange.reduce_scatter)
    column = shard_projection(block.ff.net[0].proj, manager, column=True, parts=2, fp8=fp8)
    block.ff.net[0].proj = GatherColumnInput(column, exchange)
    block.ff.net[2] = shard_projection(block.ff.net[2], manager, column=False, fp8=fp8,
                                     reduction=exchange.reduce_scatter)
    attn.set_processor(SharedTPAttention(exchange, local_heads, chunks))
    attn._h3_shared_tp_cp = True
    # The public attention shape remains full-head CP input/output. Only the
    # internal TP GEMM/backend sees the rank's head partition.
    return exchange
