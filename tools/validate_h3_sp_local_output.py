"""Distributed output-only parity: original SP gather versus optional local heads.

CPU: XGEN_DEVICE=cuda torchrun --standalone --nproc-per-node=8 tools/validate_h3_sp_local_output.py --output outputs/sp-local-output
CUDA: same command plus --device cuda, only on reserved GPUs.
No model checkpoint or denoising work is performed.
"""
import argparse,copy,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.distributed as dist


class IndexedNorm(nn.Module):
    def forward(self,x,temb,indices):
        normalized=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
        scale,shift=temb[indices].chunk(2,-1)
        return (normalized*(1+scale).to(x.dtype)).to(x.dtype)+shift.to(x.dtype)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.device=='cuda':torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl' if args.device=='cuda' else 'gloo')
    import x_diffusers
    from x_diffusers.framework.transformer.minimax_h3 import _gather_and_output
    from x_diffusers.framework.transformer.h3_local_io import LocalSpan
    rank,world=dist.get_rank(),dist.get_world_size()
    device=torch.device(args.device,int(os.environ['LOCAL_RANK'])) if args.device=='cuda' else torch.device('cpu')
    manager=SimpleNamespace(sp_size=world,sp_rank=rank,sp_group=dist.group.WORLD)
    torch.manual_seed(42)
    model=SimpleNamespace(parallel_manager=manager,norm_out=IndexedNorm().to(device),
        proj_out=nn.Linear(128,24).to(device).float(),audio_proj_out=nn.Linear(128,8).to(device).float())
    records=[]
    with torch.inference_mode():
        for total in (32,35,3):
            full=torch.randn(2,total,128,device=device,dtype=torch.bfloat16)
            temb=torch.randn(3,256,device=device,dtype=torch.bfloat16)
            indices=torch.arange(total,device=device)%3
            order=torch.randperm(total,device=device)
            vi,ai=order[:max(1,total//2)],order[max(1,total//2):]
            span=LocalSpan.for_rank(total,world,rank)
            hidden=span.slice_padded(full,dim=1)
            pad=span.length*world-total
            model._h3_cuda_local_output=False
            expected=_gather_and_output(model,hidden,temb,dist.group.WORLD,pad,indices,vi,ai,False)
            model._h3_cuda_local_output=True
            for return_dict in (False,True):
                actual=_gather_and_output(model,hidden,temb,dist.group.WORLD,pad,indices,vi,ai,return_dict)
                if return_dict:actual=(actual.sample,actual.audio_sample)
                errors=[(a-b).abs().max().item() if a.numel() else 0 for a,b in zip(actual,expected)]
                assert max(errors)<1e-5,(rank,total,errors)
                assert all(a.dtype==torch.float32 for a in actual)
                records.append(dict(total=total,padding=pad,return_dict=return_dict,max_abs=errors))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,world=world,status='passed',cases=records),indent=2))
    dist.barrier();dist.destroy_process_group()
    if rank==0:print(json.dumps(dict(status='passed',world=world,cases=len(records))))


if __name__=='__main__':main()
