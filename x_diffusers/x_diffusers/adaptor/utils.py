import os
import torch
from x_base.device import device_type

if device_type() == "npu":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist

from x_base.cfg_parallelism.utils import init_CFG_Parallel


def init_env():
    if device_type() == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.set_num_threads(24)
        dist.init_process_group(backend="nccl", init_method="env://")
        return
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)

    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    torch.set_num_threads(24)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    torch_npu.npu.set_device(local_rank)


def init_cfg_env(cfg_parallel_size):
    """
    初始化cfg并行环境
    """
    init_env()
    init_CFG_Parallel(cfg_parallel_size)
