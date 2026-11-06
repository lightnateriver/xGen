"""Optional CUDA FP8 recipes on the existing H3 pipeline, operators and TP."""

import logging
import os
from functools import wraps

import torch
import torch.distributed as dist
from torch import nn


logger = logging.getLogger(__name__)


def resolve(args, backend):
    """Resolve the hardware-specific default without loading CUDA-only operators."""
    requested = getattr(args, 'h3_cuda_precision', 'auto')
    if requested not in ('auto', 'none', 'fp8-bf16'):
        raise ValueError('Unknown H3 CUDA precision profile')
    if requested == 'auto':
        legacy = any(getattr(args, name, None) for name in (
            'h3_nvfp4_checkpoint', 'h3_nvfp4_resident', 'h3_nvfp4_compile',
            'transformer_path', 'adaln_cache_file', 'h3_turbo8', 'h3_turbo_sla_lora',
            'h3_refiner', 'lora_path_list', 'fsdp', 'inf_vram_blocks_num', 'cache_dit',
        )) or getattr(args, 'h3_attention', 'legacy') != 'legacy'
        legacy = (legacy or getattr(args, 'atten', 'none') != 'none'
                  or getattr(args, 'matmul_quant', 'none') != 'none'
                  or getattr(args, 'dtype', 'bf16') != 'bf16'
                  or getattr(args, 'phaa_num', 0) is None)
        requested = ('fp8-bf16' if backend == 'cuda' and getattr(args, 'model', None) == 'MiniMax-H3'
                     and getattr(args, 'task_type', None) in ('t2va', 'fl2va') and not legacy else 'none')
    if requested != 'none' and (backend != 'cuda' or getattr(args, 'model', None) != 'MiniMax-H3'):
        raise ValueError('fp8-bf16 requires CUDA MiniMax-H3; use auto or none on NPU and other models')
    args.h3_cuda_precision = requested
    if requested == 'fp8-bf16':
        if getattr(args, 'phaa_num', 0) == 0:
            args.phaa_num = 2
        args.h3_cuda_vae_sp = True
    return requested


def validate(args, backend):
    if getattr(args, 'h3_cuda_precision', 'none') == 'none':
        return
    if (backend != 'cuda' or args.model != 'MiniMax-H3' or args.dtype != 'bf16'
            or args.task_type not in ('t2va', 'fl2va') or args.sp != 8
            or not dist.is_initialized() or dist.get_world_size() != 8 or args.text_encoder_tp != 8):
        raise ValueError('fp8-bf16 requires CUDA BF16 T2VA/FL2VA and world=SP=text TP=8')
    conflicts = ('h3_nvfp4_checkpoint', 'h3_nvfp4_resident', 'h3_nvfp4_compile', 'transformer_path',
                 'adaln_cache_file', 'h3_turbo8', 'h3_turbo_sla_lora', 'h3_refiner', 'lora_path_list',
                 'fsdp', 'inf_vram_blocks_num', 'cache_dit')
    if any(getattr(args, name, None) for name in conflicts) or args.atten != 'none' or args.matmul_quant != 'none':
        raise ValueError('fp8-bf16 cannot combine with packed weights, LoRA, caches or other attention/quantization recipes; use --h3_cuda_precision none')
    if os.environ.get('XGEN_PHAA_FP8') == '1':
        raise ValueError('fp8-bf16 requires BF16 communication; unset XGEN_PHAA_FP8')
    if type(args.phaa_num) is not int or args.phaa_num < 1:
        raise ValueError('fp8-bf16 requires PHAA; the recommended value is 2')
    if not getattr(args, 'h3_cuda_vae_sp', False):
        raise ValueError('fp8-bf16 requires --h3_cuda_vae_sp')


def validate_manager(manager, args, backend):
    """A resident HTTP worker cannot silently change its precision recipe."""
    profile = resolve(args, backend)
    previous = getattr(manager, '_h3_cuda_precision_key', None)
    if manager.pipe is not None and previous != profile and (previous == 'fp8-bf16' or profile == 'fp8-bf16'):
        raise ValueError('Restart the service to change h3_cuda_precision')
    if profile != 'none':
        validate(args, backend)
    manager._h3_cuda_precision_key = profile


def prepare_transformer(pipe, args, device):
    from x_base.operator.h3_adaln_owner import H3OwnedAdaLN, check_shared_timestep
    from x_base.operator.h3_dense_cuda import H3DenseCUDA
    from x_base.operator.h3_fp8 import H3FP8Linear

    model = pipe.transformer
    dense = H3DenseCUDA()
    converted = 0
    for index, block in enumerate(model.transformer_blocks):
        attn = block.attn
        if attn.fused_projections or not all(isinstance(getattr(attn, name), nn.Linear) for name in ('to_q','to_k','to_v')):
            raise ValueError('FP8 recipe expects original unfused BF16 H3 projections')
        projections = (attn.to_q, attn.to_k, attn.to_v)
        biases = [layer.bias for layer in projections]
        if any(b is None for b in biases) and not all(b is None for b in biases):
            raise ValueError('H3 QKV biases must either all exist or all be absent')
        attn.to_qkv = H3FP8Linear(torch.cat([layer.weight.detach() for layer in projections]),
                                torch.cat(biases) if biases[0] is not None else None)
        attn.to_q = attn.to_k = attn.to_v = None
        attn.fused_projections = True
        targets = ((attn.to_out, '0'), (block.ff.net[0], 'proj'), (block.ff.net, '2'))
        for parent, name in targets:
            original = getattr(parent, name)
            setattr(parent, name, H3FP8Linear(original.weight, original.bias))
        converted += 4
        block.adaln_proj = H3OwnedAdaLN(block.adaln_proj, index, exact_cache=True)
        attn._h3_attention_backend = dense
    model.time_embedder.register_forward_hook(check_shared_timestep)
    # These fusions preserve the ordinary BF16/FP8 Linear interfaces. Packed FP4 FFN fusions do not.
    from .h3_cuda import install_dit_fusions

    flags = frozenset(('XGEN_H3_FUSED_INDEXED_AFFINE','XGEN_H3_FUSED_PARTIAL_ROPE','XGEN_H3_FUSED_QK_RMS'))
    install_dit_fusions(model, flags)
    resident = len(model.transformer_blocks)
    if resident != 50:
        raise ValueError('fp8-bf16 expects the complete 50-block H3 Base checkpoint')
    model.to(device)
    model._h3_cuda_precision = args.h3_cuda_precision
    model._h3_dense_cuda = dense
    model._h3_cuda_resident_blocks = resident
    logger.info('H3 CUDA precision=%s FP8_main_projections=%d resident_blocks=%d AdaLN=rank-owner exact_cache=%s GEMM=torch._scaled_mm',
                args.h3_cuda_precision, converted, resident, True)


def retain_module(module,device):
    """Exclude complete modules from the existing Diffusers offloader."""
    for child in list(module.modules()):
        registry=getattr(child,'_diffusers_hook',None)
        if registry is None:
            continue
        for name in ('group_offloading','lazy_prefetch_group_offloading','layer_execution_tracker'):
            hook=registry.get_hook(name)
            if hook is not None:
                group=getattr(hook,'group',None)
                if group is not None:
                    group.offload_()
                registry.remove_hook(name,recurse=False)
    module.to(device)


def offload_audio(audio):
    audio.to(torch.device('cpu'))
    # Legacy weight_norm caches an unregistered weight tensor. Unused encoder
    # caches can still be meta placeholders after loading; they carry no data.
    for child in audio.modules():
        if (hasattr(child,'weight_g') and hasattr(child,'weight_v')
                and child.weight.device.type=='cuda'):
            child.weight=child.weight.to('cpu')


def prepare_vaes(pipe,args,device):
    from diffusers.hooks import apply_group_offloading
    from ..framework.vae.h3_cuda_chunks import install_cpu_decode_chunks,install_staged_decoder

    install_cpu_decode_chunks(pipe.vae)
    pipe._h3_cuda_cpu_decode_chunks=True
    apply_group_offloading(pipe.vae,offload_type='leaf_level',onload_device=device,
                           offload_device=torch.device('cpu'),use_stream=False,low_cpu_mem_usage=True)
    install_staged_decoder(pipe.vae,device,retain_module)
    # Audio weight_norm pre-hooks run before leaf onload hooks. Move the complete
    # module before decode instead, as required by its upstream offload contract.
    audio=pipe.audio_vae
    audio.to(torch.device('cpu'))
    original_audio_decode=audio.decode
    @wraps(original_audio_decode)
    def decode_audio(*values,**kwargs):
        audio.to(device)
        try:
            return original_audio_decode(*values,**kwargs)
        finally:
            offload_audio(audio)
    audio.decode=decode_audio
    original=pipe.vae.decode
    @wraps(original)
    def decode(*values,**kwargs):
        import gc
        import time

        started=time.perf_counter()
        torch.cuda.synchronize()
        before=torch.cuda.memory_reserved()
        gc.collect()
        torch.cuda.empty_cache()
        event=dict(seconds=time.perf_counter()-started,reserved_before=before,
                   reserved_after=torch.cuda.memory_reserved(),allocated=torch.cuda.memory_allocated())
        pipe._h3_cuda_decode_trim=event
        logger.info('H3 CUDA before video VAE decode: unused memory reclaimed, DiT retained: %s',event)
        return original(*values,**kwargs)
    pipe.vae.decode=decode
    logger.info('H3 CUDA spatial VAE tiles; DiT resident blocks remain on device through decode')


def prepare_text_tp(pipe):
    from x_base.tensor_parallelism.layers import ColumnParallelLinear, RowParallelLinear
    from x_base.operator.h3_fp8 import H3FP8TPLinear

    targets = [(parent, name, child) for parent in pipe.text_encoder.modules()
               for name, child in parent.named_children() if isinstance(child,(ColumnParallelLinear,RowParallelLinear))]
    for parent, name, child in targets:
        setattr(parent,name,H3FP8TPLinear(child))
    if not targets:
        raise ValueError('No existing text TP projections found for the H3 FP8 profile')
    logger.info('H3 CUDA precision FP8 text TP projections=%d',len(targets))
