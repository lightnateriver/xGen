"""CPU/Gloo eight-rank mathematical gate for H3 shared TP8/CP8.

Run with torchrun --standalone --nproc-per-node=8. No checkpoint, CUDA
allocation, model-service launch or performance/quality claim is involved.
"""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from x_base.tensor_parallelism.shared_sequence import SharedSequenceExchange
from x_base.tensor_parallelism.fp8_reduce import FP8Reduction, encode_chunk, decode_chunk
from x_diffusers.adaptor.h3_shared_parallel import prepare_shared_block
from x_diffusers.framework.transformer.h3_local_io import LocalSpan
from x_diffusers.framework.transformer.minimax_h3 import _rope_minimax_h3
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock


class DenseAttention:
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        q,k,v=(layer(hidden_states).unflatten(-1,(attn.heads,attn.head_dim))
               for layer in (attn.to_q,attn.to_k,attn.to_v))
        q,k=attn.norm_q(q),attn.norm_k(k)
        if rotary_emb is not None:
            q,k=_rope_minimax_h3(q,k,*rotary_emb)
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                           attn_mask=attention_mask).transpose(1,2).flatten(2,3)
        return attn.to_out[1](attn.to_out[0](out))


def backend(q,k,v,*,attn_mask=None):
    return F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)


def collect(local,group,tokens):
    rows=local.movedim(1,0).contiguous()
    full=torch.empty((rows.shape[0]*8,*rows.shape[1:]),dtype=rows.dtype)
    dist.all_gather_into_tensor(full,rows,group=group)
    return full[:tokens].movedim(0,1).contiguous()


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    rank=dist.get_rank()
    assert dist.get_world_size()==8
    cp=dist.new_group(list(range(8)),backend='gloo')
    tp=dist.new_group(list(range(8)),backend='gloo')
    results=[]
    for wire in ('bf16','fp8'):
        exchange=SharedSequenceExchange(cp,tp,wire,chunk_elements=128)
        for batch,tokens in ((1,1),(2,13),(1,33)):
            torch.manual_seed(100+tokens)
            full=torch.randn(batch,tokens,128,dtype=torch.bfloat16)
            span=LocalSpan.for_rank(tokens,8,rank)
            local=span.slice_padded(full,dim=1)
            exchange.set_tokens(tokens)
            gathered=exchange.gather(local)
            if wire=='bf16':
                expected=full
            else:
                pieces=[]
                for peer in range(8):
                    value=LocalSpan.for_rank(tokens,8,peer).slice_padded(full,dim=1).movedim(1,0).contiguous()
                    pieces.append(decode_chunk(encode_chunk(value))[:value.numel()].view_as(value).to(value.dtype))
                expected=torch.cat(pieces)[:tokens].movedim(0,1).contiguous()
            assert torch.equal(gathered,expected), ('gather layout',rank,batch,tokens,wire)
            partial=(full.float()*(rank+1)/8).bfloat16()
            actual=exchange.reduce_scatter(partial,tp)
            if wire=='fp8':
                reference=FP8Reduction(chunk_elements=128)(partial.clone(),tp)
                expected=span.slice_padded(reference,dim=1)
                assert torch.equal(actual,expected), ('FP8 reduction differs from original',rank,batch,tokens)
            else:
                reference=partial.clone();dist.all_reduce(reference,group=tp)
                expected=span.slice_padded(reference,dim=1)
                torch.testing.assert_close(actual,expected,rtol=0.02,atol=0.02)
            results.append(dict(wire=wire,batch=batch,tokens=tokens,transport='passed'))
    # Native H3 block: non-square attention projection, real RMS/AdaLN/SwiGLU,
    # independent token positions, multiple batches and empty padded CP ranks.
    for batch,tokens,rotary in ((1,1,False),(2,13,True),(1,33,True)):
        torch.manual_seed(200+tokens)
        dense=MiniMaxH3TransformerBlock(hidden_size=128,num_attention_heads=16,
            attention_head_dim=16,ffn_dim=256,time_embed_dim=32,norm_eps=1e-5,qk_norm_eps=1e-5).eval()
        dense.attn.set_processor(DenseAttention())
        shared=deepcopy(dense)
        exchange=SharedSequenceExchange(cp,tp,'bf16');exchange.set_tokens(tokens)
        prepare_shared_block(shared,SimpleNamespace(tp_group=tp,tp_size=8,tp_rank=rank),exchange,fp8=False,chunks=2)
        shared.attn._h3_attention_backend=backend
        x=torch.randn(batch,tokens,128);temb=torch.randn(3,32);indices=torch.arange(tokens)%3
        angles=torch.randn(tokens,8)
        rope=(angles.cos(),angles.sin()) if rotary else None
        expected=dense(x,temb,indices,rope)
        span=LocalSpan.for_rank(tokens,8,rank)
        actual=shared(span.slice_padded(x,dim=1),temb,span.slice_padded(indices),rope)
        actual=collect(actual,cp,tokens)
        error=(actual-expected).abs().max().item()
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        assert exchange.metrics['gather_calls']==exchange.metrics['reduce_scatter_calls']==2
        results.append(dict(batch=batch,tokens=tokens,rotary=rotary,native_block_max_abs=error))
    print(json.dumps(dict(rank=rank,status='passed',cases=results)),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
