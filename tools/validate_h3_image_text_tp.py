"""Independent distributed oracle for Image text quantized TP gather/reduce/bias routing.

The local matmul is replaced with exact CPU F.linear here; real W8A8 kernels are
validated by the checkpoint run. This checks communication independently of quantization error.
"""
import importlib.util
from pathlib import Path
from types import MethodType
import torch
import torch.distributed as dist
import torch.nn.functional as F
from x_base.tensor_parallelism.layers import ColumnParallelLinear, RowParallelLinear
from x_base.tensor_parallelism.state import set_tp_group

spec=importlib.util.spec_from_file_location('image_optim',Path(__file__).resolve().parents[1]/'x_diffusers/x_diffusers/framework/transformer/h3_image_optimizations.py')
optim=importlib.util.module_from_spec(spec);spec.loader.exec_module(optim)
optim._image_text_tp_quant_linear=lambda layer,x,include_bias:F.linear(x,layer.weight,layer.bias if include_bias else None)
dist.init_process_group('gloo');world=dist.get_world_size();rank=dist.get_rank()
set_tp_group(dist.group.WORLD,world,rank);torch.set_num_threads(2);torch.manual_seed(421)
checks=0
try:
    for shape in [(1,7,8),(2,5,8)]:
        x=torch.randn(shape);weight=torch.randn(12,8);bias=torch.randn(12)
        for gather in (False,True):
            layer=ColumnParallelLinear(8,12,gather_output=gather,dtype=torch.float32)
            layer.load_shard(weight,bias);layer.forward=MethodType(optim._image_text_column_quant_forward,layer)
            expected=F.linear(x,weight,bias)
            if not gather:expected=expected.chunk(world,dim=-1)[rank]
            torch.testing.assert_close(layer(x),expected,rtol=1e-6,atol=1e-6);checks+=1
        for parallel in (False,True):
            for custom in (False,True):
                calls=[]
                def reduce(output,group):calls.append(True);dist.all_reduce(output,group=group);return output
                layer=RowParallelLinear(8,12,input_is_parallel=parallel,reduce_fn=reduce if custom else None,dtype=torch.float32)
                layer.load_shard(weight,bias);layer.forward=MethodType(optim._image_text_row_quant_forward,layer)
                actual=layer(x.chunk(world,dim=-1)[rank] if parallel else x)
                torch.testing.assert_close(actual,F.linear(x,weight,bias),rtol=1e-5,atol=2e-6)
                assert len(calls)==int(custom and world>1);checks+=1
    if rank==0:print(f'PASS: {checks} TP gather/reduce/input-partition/bias checks per rank, world={world}',flush=True)
finally:dist.destroy_process_group()
