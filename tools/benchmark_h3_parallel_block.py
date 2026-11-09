"""Matched SP4 / TP2-CP2 resident H3 block latency (random, full-width weights).

No text encoder, VAE, full-model IO or video generation is timed. Profiling is
separate from warmed latency; overlapping device spans must not be added.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist


class Spans:
    def __init__(self):
        self.events = []
        self.active = False
        self.restore = []

    def call(self, name, function, *args, **kwargs):
        if not self.active:
            return function(*args, **kwargs)
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin.record()
        result = function(*args, **kwargs)
        end.record()
        self.events.append((name, begin, end))
        return result

    def wrap(self, owner, name, label):
        original = getattr(owner, name)
        self.restore.append((owner, name, original))
        setattr(owner, name, lambda *a, **kw: self.call(label, original, *a, **kw))

    def totals(self):
        result = {}
        for name, begin, end in self.events:
            result[name] = result.get(name, 0.) + begin.elapsed_time(end)
        return result

    def close(self):
        for owner, name, value in reversed(self.restore):
            setattr(owner, name, value)
        self.restore.clear()


def quantize_sp(block):
    from x_base.operator.h3_fp8 import H3FP8Linear
    attn = block.attn
    attn.to_qkv = H3FP8Linear(torch.cat([attn.to_q.weight, attn.to_k.weight, attn.to_v.weight]))
    attn.to_q = attn.to_k = attn.to_v = None
    attn.fused_projections = True
    for parent, name in ((attn.to_out, '0'), (block.ff.net[0], 'proj'), (block.ff.net, '2')):
        layer = getattr(parent, name)
        setattr(parent, name, H3FP8Linear(layer.weight, layer.bias))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, nargs='+', default=[4096,16384,32768])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--profile-repeat', type=int, default=3)
    parser.add_argument('--tp4-fp8-chunks', type=int, nargs='+', default=None,
                        help='Opt-in eight-rank SP8 / TP4-CP2 packet comparison. Original four-rank recipe stays unchanged.')
    args = parser.parse_args()
    assert all(n > 0 and n % 4 == 0 for n in args.tokens)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device)
    rank = dist.get_rank()
    world=dist.get_world_size()
    assert world == (8 if args.tp4_fp8_chunks else 4)
    assert all(n % world == 0 for n in args.tokens)
    import x_diffusers  # noqa: F401
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock, MiniMaxH3RotaryPosEmbed
    from x_base.sequence_parallelism.mesh import ParallelManager
    from x_base import set_pad
    from x_base.operator.h3_adaln_owner import H3OwnedAdaLN
    from x_base.operator.h3_sage_cuda import H3SageCUDA
    from x_base.phaa.globals import enable_phaa, set_phaa_split_num
    from x_diffusers.adaptor.h3_parallel import prepare_block
    from x_diffusers.adaptor.h3_parallel_config import H3ParallelConfig
    from x_diffusers.adaptor.h3_cuda import install_dit_fusions
    from x_base.phaa import fp8_transport

    torch.manual_seed(42)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        original = MiniMaxH3TransformerBlock(hidden_size=5376, num_attention_heads=56,
            attention_head_dim=128, ffn_dim=14336, time_embed_dim=2688,
            norm_eps=1e-6, qk_norm_eps=1e-6).cpu().eval().requires_grad_(False)
    finally:
        torch.set_default_dtype(previous_dtype)
    enable_phaa()
    set_phaa_split_num(2)
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16).to(device)
    baselines = {}
    tp4_samples = {}
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    def save():
        result = dict(rank=rank,status='running',geometry=dict(hidden=5376,ffn=14336,heads=56,head_dim=128,
            time_embed_dim=2688,layers=1),config=dict(warmup=args.warmup,repeat=args.repeat,
            profile_repeat=args.profile_repeat,cp_wire='fp8',phaa=2,weight_seed=42),records=records,
            environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
        (args.output/f'rank-{rank}.json').write_text(json.dumps(result,indent=2))
    modes=([('sp8',1,8,'bf16',262144)]+[(f'tp4cp2-fp8-{chunk}',4,2,'fp8',chunk) for chunk in args.tp4_fp8_chunks]
           if args.tp4_fp8_chunks else [('sp4',1,4,'bf16',262144),('tp2cp2-bf16',2,2,'bf16',262144),('tp2cp2-fp8',2,2,'fp8',262144)])
    for label, tp, cp, tp_wire, chunk_elements in modes:
        block = copy.deepcopy(original)
        manager = ParallelManager(1,1,cp,tp,register_tp_global=False)
        reduction = None
        if tp > 1:
            reduction = prepare_block(block,manager,H3ParallelConfig('tp_cp',tp,cp,tp_wire,'fp8',chunk_elements))
        else:
            quantize_sp(block)
        block.adaln_proj = H3OwnedAdaLN(block.adaln_proj,0,exact_cache=True)
        block.attn.parallel_manager = manager
        block.attn._h3_attention_backend = H3SageCUDA()
        block.attn._h3_phaa_fp8 = True
        block.attn.processor.comm_stream = torch.cuda.Stream()
        install_dit_fusions(SimpleNamespace(transformer_blocks=[block]),frozenset((
            'XGEN_H3_FUSED_INDEXED_AFFINE','XGEN_H3_FUSED_QK_RMS','XGEN_H3_FUSED_PARTIAL_ROPE')))
        block.to(device)
        torch.cuda.synchronize()
        for n in args.tokens:
            cp_rank = dist.get_rank(manager.sp_group)
            length, start = n//cp, cp_rank*(n//cp)
            generator = torch.Generator().manual_seed(1000+n)
            global_input = torch.randn(1,n,5376,generator=generator,dtype=torch.bfloat16)
            hidden = global_input[:,start:start+length].contiguous().to(device)
            del global_input
            positions = torch.arange(start,start+length,device=device)
            rotary = rope(positions[:,None].expand(-1,3).contiguous())
            indices = (positions % 6).long().contiguous()
            generator = torch.Generator().manual_seed(314)
            embeddings = torch.randn(args.warmup+args.repeat+args.profile_repeat+1,2,2688,
                                     generator=generator,dtype=torch.float32).to(device)
            set_pad('mmh3_seq',n,manager.sp_group)
            if rank == 0:
                (args.output/'current.json').write_text(json.dumps(dict(mode=label,tokens=n,phase='warmup')))
                print(f'begin {label} tokens={n}',flush=True)
            for i in range(args.warmup):
                output = block(hidden,embeddings[i],indices,rotary,None)
                del output
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples=[]
            for i in range(args.repeat):
                dist.barrier()
                torch.cuda.synchronize()
                began,ended=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                began.record()
                t=time.perf_counter()
                output=block(hidden,embeddings[args.warmup+i],indices,rotary,None)
                ended.record()
                torch.cuda.synchronize()
                samples.append(dict(wall_ms=(time.perf_counter()-t)*1000,device_ms=began.elapsed_time(ended)))
                if i != args.repeat-1:
                    del output
            peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
            assert torch.isfinite(output).all()
            # Compare 64 fixed global rows, outside latency/memory measurement.
            selected=torch.linspace(0,n-1,64,device=device).long()
            mask=(selected>=start)&(selected<start+length)
            sample=torch.zeros(64,5376,device=device,dtype=torch.float32)
            if tp == 1 or manager.tp_rank == 0:
                sample[mask]=output[0].index_select(0,selected[mask]-start).float()
            dist.all_reduce(sample)
            cpu_sample=sample.cpu()
            if label in ('sp4','sp8'):
                baselines[n]=cpu_sample
                relative=0.
            else:
                reference=baselines[n]
                relative=((cpu_sample-reference).square().mean().sqrt()/reference.square().mean().sqrt()).item()
                assert relative<0.05,relative
            packet_sample_identical=None
            if args.tp4_fp8_chunks and tp==4:
                if n not in tp4_samples:tp4_samples[n]=cpu_sample.clone()
                packet_sample_identical=torch.equal(cpu_sample,tp4_samples[n])
                assert packet_sample_identical,'Changing packet size changed sampled CUDA predictions'
            del sample,cpu_sample
            profile=Spans()
            profile.wrap(block.attn,'forward','attention_envelope')
            profile.wrap(block.ff,'forward','ffn_envelope')
            profile.wrap(block.adaln_proj,'forward','adaln_envelope')
            profile.wrap(dist,'all_to_all_single','cp_nccl')
            profile.wrap(fp8_transport,'fp8_all_to_all_many','cp_pack_exchange_unpack')
            original_reduce=dist.all_reduce
            def profile_reduce(tensor,*a,**kw):
                if tensor.numel()==1:
                    return profile.call('tp_activation_scale',original_reduce,tensor,*a,**kw)
                return original_reduce(tensor,*a,**kw)
            profile.restore.append((dist,'all_reduce',original_reduce))
            dist.all_reduce=profile_reduce
            for name,module in [('qkv',block.attn.to_qkv),('out',block.attn.to_out[0]),
                                ('gate_up',block.ff.net[0].proj),('down',block.ff.net[2])]:
                profile.wrap(module,'_local_forward',name+'_quant_gemm')
                if getattr(module,'reduce_fn',None) is not None:
                    profile.wrap(module,'reduce_fn','tp_data_reduce')
            profiles=[]
            try:
                for i in range(args.profile_repeat):
                    dist.barrier()
                    torch.cuda.synchronize()
                    profile.events=[]
                    profile.active=True
                    output=block(hidden,embeddings[args.warmup+args.repeat+i],indices,rotary,None)
                    profile.active=False
                    torch.cuda.synchronize()
                    profiles.append(profile.totals())
            finally:
                profile.active=False
                profile.close()
            records.append(dict(mode=label,tp=tp,cp=cp,tp_wire=tp_wire,cp_wire='fp8',tokens=n,
                chunk_elements=chunk_elements,packet_sample_identical=packet_sample_identical,
                local_tokens=length,samples=samples,profiles_ms=profiles,peak=peak,sample_relative_rmse=relative,
                tp_metrics=dict(reduction.metrics) if reduction is not None else {},
                adaln_projection_calls=block.adaln_proj.projection_calls,adaln_cache_hits=block.adaln_proj.cache_hits,
                sage_calls=block.attn._h3_attention_backend.kernel_calls))
            save()
            if rank==0:print(f'finished {label} tokens={n}',flush=True)
            del hidden,output,rotary,indices,embeddings,profiles,profile,module
            gc.collect()
            torch.cuda.empty_cache()
        del block,reduction,manager
        gc.collect()
        torch.cuda.empty_cache()
    save()
    p=args.output/f'rank-{rank}.json'
    result=json.loads(p.read_text());result['status']='passed';p.write_text(json.dumps(result,indent=2))
    dist.barrier()
    if rank==0:print('all cases passed',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
