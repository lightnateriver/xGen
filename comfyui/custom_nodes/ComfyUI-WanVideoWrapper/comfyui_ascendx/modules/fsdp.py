from .quant import WeightQuantLinearModule
import torch
import functools
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, transformer_auto_wrap_policy

def fsdp(
        model,
        device_id,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        process_group=None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
):

    weight_quant_layers = []

    def _recursive_search(module, prefix=''):
        for name, child in module.named_children():
            child_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, WeightQuantLinearModule):
                weight_quant_layers.append(child)
            _recursive_search(child, child_name)

    _recursive_search(model)

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model),
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype),
        device_id=device_id,
        ignored_modules=weight_quant_layers,
        sync_module_states=sync_module_states)
    
    return model