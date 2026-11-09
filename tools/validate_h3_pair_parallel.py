"""Eight-rank CPU/Gloo native-block oracle for TP2 projections plus SP8."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from x_base.sequence_parallelism.mesh import ParallelManager
from x_base.tensor_parallelism.pair_sequence import PairSequenceExchange
from x_diffusers.adaptor.h3_pair_parallel import prepare_pair_block
from x_diffusers.framework.transformer.h3_local_io import LocalSpan
from x_diffusers.framework.transformer.minimax_h3 import _rope_minimax_h3
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock


class Attention:
    def __init__(self, tokens, group=None):
        self.tokens, self.group = tokens, group

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        qkv = attn.to_qkv(hidden_states)
        if self.group is not None:
            rows = qkv.movedim(1, 0).contiguous()
            full = torch.empty((rows.shape[0]*8, *rows.shape[1:]), dtype=rows.dtype)
            dist.all_gather_into_tensor(full, rows, group=self.group)
            qkv = full[:self.tokens].movedim(0, 1)
        q,k,v = (x.unflatten(-1,(attn.heads,-1)) for x in qkv.chunk(3,dim=-1))
        q,k = attn.norm_q(q),attn.norm_k(k)
        if rotary_emb is not None:
            q,k = _rope_minimax_h3(q,k,*rotary_emb)
        result = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
        result = result.transpose(1,2).flatten(2,3)
        if self.group is not None:
            result = LocalSpan.for_rank(self.tokens,8,dist.get_rank()).slice_padded(result,dim=1)
        return attn.to_out[1](attn.to_out[0](result))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    assert dist.get_world_size()==8
    mesh = ParallelManager(1,1,4,2,register_tp_global=False)
    exchange = PairSequenceExchange(mesh.tp_group)
    checks=[]
    for batch,tokens,rotary in ((1,1,False),(2,13,True),(1,33,True)):
        torch.manual_seed(700+tokens)
        dense = MiniMaxH3TransformerBlock(hidden_size=128,num_attention_heads=16,
            attention_head_dim=16,ffn_dim=256,time_embed_dim=32,norm_eps=1e-5,qk_norm_eps=1e-5).eval()
        attn=dense.attn
        qkv=(attn.to_q,attn.to_k,attn.to_v)
        attn.to_qkv=nn.Linear(128,768,bias=qkv[0].bias is not None)
        attn.to_qkv.weight=nn.Parameter(torch.cat([p.weight for p in qkv]),requires_grad=False)
        if qkv[0].bias is not None:
            attn.to_qkv.bias=nn.Parameter(torch.cat([p.bias for p in qkv]),requires_grad=False)
        attn.to_q=attn.to_k=attn.to_v=None;attn.fused_projections=True
        dense.attn.set_processor(Attention(tokens))
        pair=deepcopy(dense);prepare_pair_block(pair,exchange)
        pair.attn.set_processor(Attention(tokens,dist.group.WORLD))
        x=torch.randn(batch,tokens,128);temb=torch.randn(3,32);indices=torch.arange(tokens)%3
        angles=torch.randn(tokens,8);rope=(angles.cos(),angles.sin()) if rotary else None
        expected=dense(x,temb,indices,rope)
        span=LocalSpan.for_rank(tokens,8,rank)
        actual=pair(span.slice_padded(x,dim=1),temb,span.slice_padded(indices),rope)
        rows=actual.movedim(1,0).contiguous();full=torch.empty((rows.shape[0]*8,*rows.shape[1:]),dtype=rows.dtype)
        dist.all_gather_into_tensor(full,rows)
        observed=full[:tokens].movedim(0,1)
        torch.testing.assert_close(observed,expected,rtol=2e-5,atol=2e-6)
        checks.append(dict(batch=batch,tokens=tokens,rotary=rotary,max_abs=float((observed-expected).abs().max())))
    from x_base.operator.h3_fp8 import H3FP8Linear
    from x_diffusers.adaptor.h3_pair_parallel import slice_projection
    original=H3FP8Linear(torch.randn(768,128),torch.randn(768))
    for column,parts in ((True,3),(True,2),(False,1)):
        sliced=slice_projection(original,exchange,column=column,parts=parts)
        axis=0 if column else 1
        expected=torch.cat([v.chunk(2,dim=axis)[exchange.rank] for v in original.weight.chunk(parts,dim=axis)],dim=axis)
        assert torch.equal(sliced.weight.view(torch.uint8),expected.view(torch.uint8))
        assert torch.equal(sliced.weight_scale,original.weight_scale)
        assert column or sliced.bias is None
    # Each exchange is compared to an independent full-row protocol oracle,
    # including FP8 scales, owner order, tail padding and FP32 summation.
    from x_base.tensor_parallelism.fp8_reduce import encode_chunk,decode_chunk
    for wire in ('bf16','fp8'):
        transport=PairSequenceExchange(mesh.tp_group,wire,chunk_elements=256)
        def quant(value):
            return decode_chunk(encode_chunk(value))[:value.numel()].view_as(value).to(value.dtype) if wire=='fp8' else value
        local=torch.randn(2,17,128,dtype=torch.bfloat16)+rank
        peers=[torch.empty_like(local) for _ in range(2)]
        dist.all_gather(peers,local,group=mesh.tp_group)
        expected=torch.cat([quant(v.movedim(1,0).contiguous()).movedim(0,1) for v in peers],dim=1)
        assert torch.equal(transport.gather(local),expected)
        value=torch.randn(2,34,3*64,dtype=torch.bfloat16)+rank
        peers=[torch.empty_like(value) for _ in range(2)]
        dist.all_gather(peers,value,group=mesh.tp_group)
        received=[quant(v.reshape(2,2,17,3,64).permute(1,2,0,3,4).contiguous()[transport.rank]) for v in peers]
        expected=torch.stack(received).permute(2,1,3,0,4).reshape(2,17,3*128)
        assert torch.equal(transport.scatter_columns(value,3),expected)
        value=torch.randn(2,34,128,dtype=torch.bfloat16)+rank
        peers=[torch.empty_like(value) for _ in range(2)]
        dist.all_gather(peers,value,group=mesh.tp_group)
        selected=[v.movedim(1,0).contiguous().chunk(2)[transport.rank] for v in peers]
        def chunk_quant(v):
            flat=v.flatten()
            return torch.cat([decode_chunk(encode_chunk(flat[i:i+256]))[:min(256,flat.numel()-i)] for i in range(0,flat.numel(),256)]).view_as(v)
        expected=sum(chunk_quant(v) if wire=='fp8' else v.float() for v in selected).to(value.dtype).movedim(0,1)
        assert torch.equal(transport.reduce_scatter(value),expected)
        assert transport.metrics['wire_bytes']>0
        checks.append(dict(wire=wire,protocol_oracle='byte_equal',metrics=transport.metrics))
    print(json.dumps(dict(rank=rank,status='passed',ranks=exchange.metrics['ranks'],cases=checks)),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
