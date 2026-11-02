"""NPU operator parity against pinned upstream; not an E2E/video quality test."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch_npu
from h3_ascend.context import VideoLayout, HeadShard


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('spark','veda'),required=True)
    parser.add_argument('--predictor')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(42);torch.npu.manual_seed(42)
    layout=VideoLayout((37,24,42),542,128,(128,414),frames=124)
    result=dict(backend=args.backend,grid=layout.grid,prefix=layout.prefix,
                scope='synthetic QKV transport/operator parity; video quality is a separate gate',rows=[])
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    try:
        for start,count in [(0,4),(53,3)]:
            shard=HeadShard(start,count)
            if args.backend=='spark':
                from h3_ascend.backends.spark import SparkAttention
                reference=SparkAttention(layout)
                candidate=SparkAttention(layout)
            else:
                from h3_ascend.backends.veda import VedaAttention as Original
                from x_base.operator.h3_attention.veda import VedaAttention
                reference=Original(layout,shard,predictor_path=args.predictor,device='npu:0')
                candidate=VedaAttention(layout,shard,predictor_path=args.predictor,
                                        shared_bundle=reference.impl.bundle,device='npu:0')
            q,k,v=[torch.randn(layout.sequence_length,count,128,device='npu',dtype=torch.bfloat16)
                   for _ in range(3)]
            for layer in (0,24,49):
                expected=reference(q,k,v,layer=layer).clone()
                actual=candidate(q,k,v,layer=layer).clone()
                torch.npu.synchronize()
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-3)
                difference=(actual.float()-expected.float())
                relative=(difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-8)).item()
                timings=[]
                for _ in range(3):
                    begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                    begin.record();candidate(q,k,v,layer=layer);end.record()
                    torch.npu.synchronize();timings.append(begin.elapsed_time(end))
                result['rows'].append(dict(head_start=start,heads=count,layer=layer,
                    max_abs_error=difference.abs().max().item(),relative_rmse=relative,
                    candidate_device_ms=timings,passed=True))
                path.write_text(json.dumps(result,indent=2))
            del reference,candidate,q,k,v,expected,actual,difference
            gc.collect();torch.npu.empty_cache()
        result['passed']=True
    except Exception:
        import traceback
        result.update(passed=False,error=traceback.format_exc())
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-started
        path.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
