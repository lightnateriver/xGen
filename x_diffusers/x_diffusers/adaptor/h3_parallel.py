"""H3-only TP adapter; reuse xGen mesh, Linear, FP8 GEMM and PHAA.

Weights are sliced on CPU before the existing resident-device transfer.
Fused QKV and SwiGLU halves are sliced separately, never as one flat matrix.
"""
import logging
from functools import wraps

import torch
import torch.distributed as dist
from torch import nn

from .h3_parallel_config import H3ParallelConfig

logger = logging.getLogger(__name__)


def initialize(model, args):
    from x_base.sequence_parallelism.mesh import ParallelManager

    config = H3ParallelConfig.from_args(args, 'cuda', dist.get_world_size())
    if config.mode not in ('tp_cp', 'tp_cp_shared', 'tp_sp'):
        return None
    if hasattr(model, '_h3_parallel_config'):
        raise ValueError('H3 TP/CP layout is immutable; restart the worker to change it')
    heads = model.config.num_attention_heads
    if heads % (max(config.tp, config.cp) if config.mode in ('tp_cp_shared', 'tp_sp') else config.tp * config.cp):
        raise ValueError('H3 attention heads must divide TP * CP')
    if config.mode == 'tp_cp_shared':
        from x_base.tensor_parallelism.shared_sequence import SharedSequenceExchange
        manager = ParallelManager(1, 1, config.cp, 1, register_tp_global=False)
        tp_manager = ParallelManager(1, 1, 1, config.tp, register_tp_global=False)
        model._h3_shared_tp_manager = tp_manager
        model._h3_shared_sequence_exchange = SharedSequenceExchange(
            manager.sp_group, tp_manager.tp_group, config.tp_comm, config.tp_chunk_elements)
    elif config.mode == 'tp_sp':
        from x_base.tensor_parallelism.pair_sequence import PairSequenceExchange
        manager = ParallelManager(1, 1, config.cp, 1, register_tp_global=False)
        model._h3_pair_manager = ParallelManager(1, 1, 4, 2, register_tp_global=False)
        model._h3_pair_exchange = PairSequenceExchange(model._h3_pair_manager.tp_group, config.tp_comm,
                                                       config.tp_chunk_elements)
    else:
        manager = ParallelManager(1, 1, config.cp, config.tp, register_tp_global=False)
    model.set_parallel_manager(manager)
    model._h3_parallel_config = config
    if config.warmup_mib:
        from .h3_collective_warmup import warmup_groups
        model._h3_parallel_warmup = warmup_groups(manager, config)
        if config.mode == 'tp_cp_shared':
            model._h3_shared_tp_warmup = warmup_groups(tp_manager, config)
        elif config.mode == 'tp_sp':
            model._h3_pair_warmup = warmup_groups(model._h3_pair_manager, config)
    for block in model.transformer_blocks:
        block.attn._h3_tp_cp = True
    projection_manager = getattr(model, '_h3_pair_manager', getattr(model, '_h3_shared_tp_manager', manager))
    logger.info('Experimental H3 parallelism=%s TP=%d CP(Ulysses)=%d TP_wire=%s CP_wire=%s TP_packet_elements=%d ranks_tp=%s ranks_cp=%s',
                config.mode, config.tp, config.cp, config.tp_comm, config.cp_comm, config.tp_chunk_elements,
                dist.get_process_group_ranks(projection_manager.tp_group) if config.tp > 1 else [],
                dist.get_process_group_ranks(manager.sp_group) if config.cp > 1 else [])
    if config.mode == 'tp_cp_shared':
        logger.info('H3 shared-rank TP/CP: CP residual rows, TP8 QKV/O/FFN; QKV all-to-all fused into TP head layout; row sums reduce-scatter to CP rows')
    return config


def shard_projection(original, manager, *, column, parts=1, fp8=True, reduction=None):
    from x_base.tensor_parallelism.layers import ColumnParallelLinear, RowParallelLinear
    from x_base.operator.h3_fp8 import H3FP8TPLinear

    if not isinstance(original, nn.Linear) or original.weight.device.type != 'cpu':
        raise ValueError('H3 TP requires original Linear CPU weights before device transfer')
    group = manager.tp_group
    peers, rank = manager.tp_size, manager.tp_rank
    if peers < 2 or group is None:
        raise ValueError('H3 shard_projection requires an explicit multi-rank TP group')
    axis = 0 if column else 1
    if original.weight.shape[axis] % (peers * parts):
        raise ValueError('H3 fused projection dimensions must divide parts * TP')
    # Preserve the unsharded per-tensor weight scale, rather than independently
    # quantizing each shard with a different scale.
    scale = (original.weight.detach().float().abs().amax() / 448).clamp_min(torch.finfo(torch.float32).tiny)
    shards = [part.chunk(peers, dim=axis)[rank] for part in original.weight.detach().chunk(parts, dim=axis)]
    weight = torch.cat(shards, dim=axis).contiguous()
    bias = original.bias
    if column and bias is not None:
        bias = torch.cat([part.chunk(peers)[rank] for part in bias.detach().chunk(parts)])
    klass = ColumnParallelLinear if column else RowParallelLinear
    options = {} if column else dict(reduce_fn=reduction)
    layer = klass(original.in_features, original.out_features, bias=bias is not None,
                  dtype=original.weight.dtype, tp_group=group, **options)
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None
    if not fp8:
        return layer
    layer = H3FP8TPLinear(layer)
    layer.weight.copy_((weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn))
    layer.weight_scale.copy_(scale)
    # Row GEMM must share its activation scale across input partitions.
    if not column:
        layer.activation_group = group
    return layer


def prepare_block(block, manager, config, *, fp8=True):
    from x_base.tensor_parallelism.fp8_reduce import BF16Reduction, FP8Reduction

    if manager.tp_size == 1:
        raise ValueError('TP1 uses the original projection preparation')
    attn = block.attn
    attn._h3_tp_cp = True
    if attn.fused_projections:
        raise ValueError('H3 TP requires original unfused Q/K/V weights')
    projections = [attn.to_q, attn.to_k, attn.to_v]
    if any(p.bias is None for p in projections) and not all(p.bias is None for p in projections):
        raise ValueError('H3 QKV biases must all exist or all be absent')
    merged = nn.Linear(projections[0].in_features, sum(p.out_features for p in projections),
                       bias=projections[0].bias is not None, dtype=projections[0].weight.dtype, device='meta')
    merged.weight = nn.Parameter(torch.cat([p.weight.detach() for p in projections]), requires_grad=False)
    if projections[0].bias is not None:
        merged.bias = nn.Parameter(torch.cat([p.bias.detach() for p in projections]), requires_grad=False)
    reduction = FP8Reduction(chunk_elements=config.tp_chunk_elements) if config.tp_comm == 'fp8' else BF16Reduction()
    attn.to_qkv = shard_projection(merged, manager, column=True, parts=3, fp8=fp8)
    attn.to_q = attn.to_k = attn.to_v = None
    attn.fused_projections = True
    attn.heads //= manager.tp_size
    attn.inner_dim //= manager.tp_size
    attn.to_out[0] = shard_projection(attn.to_out[0], manager, column=False, fp8=fp8, reduction=reduction)
    block.ff.net[0].proj = shard_projection(block.ff.net[0].proj, manager, column=True, parts=2, fp8=fp8)
    block.ff.net[2] = shard_projection(block.ff.net[2], manager, column=False, fp8=fp8, reduction=reduction)
    return reduction


def attach_pipeline(pipe, args):
    """Called instead of enable_sp only for the opt-in H3 path."""
    from x_base.sequence_parallelism.mesh import ParallelManager

    model = pipe.transformer
    config = model._h3_parallel_config
    if getattr(args, 'h3_cuda_vae_sp', False):
        # VAE still spans all devices; it does not inherit DiT's smaller CP group.
        pipe._h3_vae_parallel_manager = (model.parallel_manager if config.mode == 'tp_cp_shared'
            else ParallelManager(1, 1, dist.get_world_size(), 1, register_tp_global=False))
        model._h3_condition_parallel_manager = pipe._h3_vae_parallel_manager
    logger.info('H3 local IO enabled; FP32 prediction gather, text TP unchanged, VAE SP=%d',
                dist.get_world_size() if getattr(args, 'h3_cuda_vae_sp', False) else 1)
    return config


def stage_reference_audio(pipe, device):
    """Reference audio follows the same whole-module staging contract as decode."""
    from .h3_cuda_precision import offload_audio

    audio = pipe.audio_vae
    original = audio.encode
    @wraps(original)
    def encode(*values, **kwargs):
        audio.to(device)
        try:
            return original(*values, **kwargs)
        finally:
            offload_audio(audio)
    audio.encode = encode
