"""Eight-rank real-width Token Refiner gates, random weights, no checkpoint load."""
import argparse,copy,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist


def errors(reference, actual):
    values=torch.zeros(4,device=reference.device,dtype=torch.float64)
    maximum=torch.zeros((),device=reference.device)
    for start in range(0,reference.shape[1],512):
        a,b=reference[:,start:start+512].float(),actual[:,start:start+512].float()
        diff=a-b
        maximum=torch.maximum(maximum,diff.abs().max())
        values+=torch.stack((diff.double().square().sum(),a.double().square().sum(),
                             b.double().square().sum(),(a.double()*b.double()).sum()))
    return dict(max_abs=maximum.item(),relative_rmse=(values[0]/values[1]).sqrt().item(),
                cosine=(values[3]/(values[1]*values[2]).sqrt()).item(),finite=bool(torch.isfinite(actual).all()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl');assert dist.get_world_size()==8
    import x_diffusers
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefiner
    from x_base import set_pad,get_pad,split_sequence,gather_sequence
    from x_base.operator.h3_dense_cuda import H3DenseCUDA
    from x_diffusers.adaptor.h3_input_refiner import install_input_refiner,convert_ffn
    torch.manual_seed(42)
    device=torch.device('cuda',rank)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        base=MiniMaxH3TokenRefiner(5376,56,128,14336,2,1e-5,1e-5,1e-5).eval()
    torch.set_default_dtype(torch.float32)
    plain_fp8=copy.deepcopy(base)
    # Match the production lifecycle: prepare weights before SP initialization.
    sp_bf16=SimpleNamespace(token_refiner=copy.deepcopy(base))
    sp_fp8=SimpleNamespace(token_refiner=copy.deepcopy(base))
    install_input_refiner(sp_bf16,'cudnn-sp')
    install_input_refiner(sp_fp8,'cudnn-sp-fp8')
    for model in (sp_bf16,sp_fp8):
        assert not hasattr(model,'parallel_manager')
        model.parallel_manager=SimpleNamespace(sp_group=dist.group.WORLD,tp_size=1)
    convert_ffn(plain_fp8)
    dense=H3DenseCUDA()
    for block in plain_fp8.refiner_blocks:block.attn._h3_attention_backend=dense
    records=[]
    with torch.inference_mode():
        for tokens in (3,33,1003,8195):
            torch.manual_seed(1000+tokens)
            x=torch.randn(1,tokens,5376,device=device,dtype=torch.bfloat16)
            old_pad=set_pad('pad',1010,dist.group.WORLD)
            reference=base(x)
            actual=sp_bf16.token_refiner(x)
            bf16=errors(reference,actual)
            assert bf16['finite'] and bf16['relative_rmse']<.005 and bf16['max_abs']<.1,(rank,tokens,bf16)
            actual=sp_fp8.token_refiner(x)
            combined=errors(reference,actual)
            assert combined['finite'] and combined['relative_rmse']<.03 and combined['cosine']>.999,(rank,tokens,combined)
            fp8_reference=plain_fp8(x)
            sp_only=errors(fp8_reference,actual)
            # These whole-module paths have DIFFERENT BF16 attention inputs
            # to nonlinear quantization. Retain their difference as diagnostic;
            # isolate FFN/SP equivalence on IDENTICAL input below instead.
            assert sp_only['finite'],(rank,tokens,sp_only)
            ffn_parity=[]
            padding=(-tokens)%8
            local=split_sequence(x,dist.group.WORLD,dim=1,pad=padding)
            for original_block,local_block in zip(plain_fp8.refiner_blocks,sp_fp8.token_refiner.refiner_blocks):
                expected=original_block.ff(x)
                gathered=gather_sequence(local_block.ff(local),dist.group.WORLD,dim=1,pad=padding)
                comparison=errors(expected,gathered)
                assert comparison['finite'] and comparison['relative_rmse']<.005,(rank,tokens,comparison)
                ffn_parity.append(comparison)
                del expected,gathered
            assert get_pad('pad')==old_pad,'Input refiner overwrote main DiT padding'
            assert actual.shape==reference.shape and actual.dtype==torch.bfloat16
            records.append(dict(tokens=tokens,bf16_sp_vs_original=bf16,fp8_sp_vs_original=combined,
                                fp8_sp_vs_fp8_full_diagnostic=sp_only,
                                identical_input_fp8_ffn_sp_parity=ffn_parity,main_pad_preserved=True))
            del x,reference,actual,fp8_reference,local
            torch.cuda.empty_cache()
        torch.manual_seed(53)
        x=torch.randn(1,33,5376,device=device,dtype=torch.bfloat16)
        before=sp_fp8.token_refiner._h3_cudnn_backend.calls
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            output=sp_fp8.token_refiner(x)
            torch.cuda.synchronize()
        ops={event.key:event.count for event in profiler.key_averages()
             if 'scaled_dot_product' in event.key or '_scaled_mm' in event.key}
        assert ops.get('aten::_scaled_dot_product_cudnn_attention',0)==2,ops
        assert sp_fp8.token_refiner._h3_cudnn_backend.calls-before==2
        assert sp_fp8.token_refiner._h3_sp_metrics['fp8_ffn_projections']==4
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed',rank=rank,world=8,cases=records,profile_ops=ops,
        metrics=sp_fp8.token_refiner._h3_sp_metrics,scope='Synthetic random-weight full-width module gate, not trained model/video quality.',
        thresholds=dict(bf16_relative_rmse=.005,bf16_max_abs=.1,fp8_relative_rmse=.03,fp8_minimum_cosine=.999))
    (args.output/f'rank-{rank}.json').write_text(json.dumps(result,indent=2))
    dist.barrier();dist.destroy_process_group()
    if rank==0:print(json.dumps(dict(status='passed',world=8,cases=len(records),cuDNN_ops=ops)))


if __name__=='__main__':main()
