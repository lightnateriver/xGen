"""Opt-in startup warmup for the existing H3 DiT TP/CP process groups."""
import logging
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def warmup_groups(manager, config):
    """Run before resident weight transfer; never enter the denoising loop."""
    if not config.warmup_mib:
        return None
    started = time.perf_counter()
    device = torch.device('cuda', torch.cuda.current_device())
    free_before, _ = torch.cuda.mem_get_info(device)
    operations = []
    shared = getattr(config, 'mode', None) in ('tp_sp', 'tp_cp_shared')
    if manager.tp_size > 1 and shared:
        group = manager.tp_group
        # W8A8 row GEMMs share their activation maximum with the TP group.
        scale = torch.zeros(128, device=device, dtype=torch.float32)
        dist.all_reduce(scale, op=dist.ReduceOp.MAX, group=group)
        dtype = torch.uint8 if config.tp_comm == 'fp8' else torch.bfloat16
        send = torch.zeros(262144, device=device, dtype=dtype)
        receive = torch.empty(send.numel()*manager.tp_size, device=device, dtype=dtype)
        dist.all_gather_into_tensor(receive, send, group=group)
        operations += ['tp.fp32_max', 'tp.'+config.tp_comm+'_all_gather']
        if config.tp_comm == 'fp8':
            dist.all_to_all_single(receive, torch.cat([send]*manager.tp_size), group=group)
            operations.append('tp.uint8_all_to_all')
        else:
            dist.reduce_scatter_tensor(send, receive, group=group)
            operations.append('tp.bf16_reduce_scatter')
        torch.cuda.synchronize(device)
        del scale, send, receive
    elif manager.tp_size > 1:
        group = manager.tp_group
        if config.tp_comm == 'fp8':
            # The row activation maximum and packed partial-output exchange.
            scale = torch.zeros(128, device=device, dtype=torch.float32)
            dist.all_reduce(scale, op=dist.ReduceOp.MAX, group=group)
            send = torch.zeros(1024 * 1024, device=device, dtype=torch.uint8)
            receive = torch.empty(send.numel() * manager.tp_size, device=device, dtype=send.dtype)
            dist.all_gather_into_tensor(receive, send, group=group)
            operations += ['tp.fp32_max', 'tp.uint8_all_gather']
            torch.cuda.synchronize(device)
            del scale, send, receive
        else:
            value = torch.zeros(262144, device=device, dtype=torch.bfloat16)
            dist.all_reduce(value, group=group)
            operations.append('tp.bf16_all_reduce')
            torch.cuda.synchronize(device)
            del value
    cp_bytes = 0
    if manager.sp_size > 1 and not (shared and manager.tp_size > 1):
        dtype = torch.uint8 if config.cp_comm == 'fp8' else torch.bfloat16
        itemsize = 1 if config.cp_comm == 'fp8' else 2
        count = config.warmup_mib * 1024**2 // (itemsize * manager.sp_size) * manager.sp_size
        send = torch.zeros(count, device=device, dtype=dtype)
        receive = torch.empty_like(send)
        dist.all_to_all_single(receive, send, group=manager.sp_group)
        torch.cuda.synchronize(device)
        cp_bytes = count * itemsize
        operations.append('cp.all_to_all')
        del send, receive
    free_after, _ = torch.cuda.mem_get_info(device)
    stats = dict(completed=True, calls=1, seconds=time.perf_counter()-started,
                 tp=manager.tp_size, cp=manager.sp_size, operations=operations,
                 cp_buffer_bytes=cp_bytes, free_before=free_before, free_after=free_after)
    # free delta also includes PyTorch's cached warmup tensors, not just NCCL.
    logger.info('H3 DiT collective startup warmup completed before weight transfer: %s', stats)
    return stats
