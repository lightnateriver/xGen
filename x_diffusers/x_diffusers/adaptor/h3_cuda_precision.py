"""Optional CUDA FP8 recipes on the existing H3 pipeline, operators and TP."""

import logging
import os
from functools import wraps

import torch
import torch.distributed as dist
from torch import nn


logger = logging.getLogger(__name__)

CONDITION_PARALLEL = ('h3_cuda_vision_sp', 'h3_cuda_vae_encoder_parallel',
                      'h3_cuda_audio_encoder_parallel', 'h3_cuda_audio_decoder_parallel')


def condition_parallel_key(args):
    return tuple(getattr(args, name, 'none' if name.endswith('decoder_parallel') else False)
                 for name in CONDITION_PARALLEL)


def resolve(args, backend):
    """Resolve the hardware-specific default without loading CUDA-only operators."""
    from .h3_cuda_options import resolve_options
    resolve_options(args, backend)
    requested = getattr(args, 'h3_cuda_precision', 'auto')
    if requested not in ('auto', 'none', 'fp8-bf16', 'fp8-sage'):
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
        raise ValueError(f'{requested} requires CUDA MiniMax-H3; use auto or none on NPU and other models')
    portable = requested == 'none'
    if portable:
        from .h3_features import validate as validate_features
        validate_features(args, backend)
    linear_tp = getattr(args, 'h3_cuda_linear_tp', 1)
    if type(linear_tp) is not int or linear_tp not in (1, 2):
        raise ValueError('h3_cuda_linear_tp must be 1 or 2')
    if linear_tp == 2 and (backend != 'cuda' or requested not in ('fp8-bf16', 'fp8-sage')
            or getattr(args, 'sp', 1) != 8 or getattr(args, 'h3_parallel_mode', 'sp') != 'sp'):
        raise ValueError('h3_cuda_linear_tp=2 requires CUDA H3 FP8 and original SP8 mode')
    if getattr(args, 'h3_cuda_local_output', False) and not portable and backend != 'cuda':
        raise ValueError('h3_cuda_local_output requires a CUDA H3 FP8 precision profile')
    if getattr(args, 'h3_cuda_local_input', False) and not portable and backend != 'cuda':
        raise ValueError('h3_cuda_local_input requires a CUDA H3 FP8 precision profile')
    if getattr(args, 'h3_cuda_conditioning_offload', False) and (
            (not portable and backend != 'cuda') or getattr(args, 'task_type', None) != 'ref2va'):
        raise ValueError('h3_cuda_conditioning_offload requires an explicit CUDA FP8 Ref2VA profile')
    chunk = getattr(args, 'h3_cuda_refiner_ff_chunk_size', 0)
    if type(chunk) is not int or chunk < 0 or (chunk and (chunk < 128 or chunk > 32768 or chunk % 128)):
        raise ValueError('h3_cuda_refiner_ff_chunk_size must be zero or 128..32768 in multiples of 128')
    if chunk and ((not portable and backend != 'cuda') or getattr(args, 'task_type', None) != 'ref2va'):
        raise ValueError('h3_cuda_refiner_ff_chunk_size requires an explicit CUDA FP8 Ref2VA profile')
    input_refiner = getattr(args, 'h3_cuda_input_refiner', 'none')
    if input_refiner not in ('none', 'cudnn-sp', 'cudnn-sp-fp8', 'custom'):
        raise ValueError('Unknown h3_cuda_input_refiner profile')
    if input_refiner != 'none' and not portable and (backend != 'cuda'
            or getattr(args, 'task_type', None) not in ('t2va', 'fl2va', 'ref2va')
            or (getattr(args, 'h3_parallel_mode', 'sp') == 'sp' and getattr(args, 'sp', 1) != 8)):
        raise ValueError('h3_cuda_input_refiner requires a CUDA FP8 H3 SP8 profile')
    if getattr(args, 'h3_cuda_precompute_conditioning', False) and not portable and (backend != 'cuda'
            or getattr(args, 'task_type', None) not in ('t2va', 'fl2va', 'ref2va')
            or (getattr(args, 'h3_parallel_mode', 'sp') == 'sp' and getattr(args, 'sp', 1) != 8)):
        raise ValueError('h3_cuda_precompute_conditioning requires a CUDA FP8 H3 SP8 profile')
    parallel_key = condition_parallel_key(args)
    if parallel_key[-1] not in ('none', 'long'):
        raise ValueError('Unknown h3_cuda_audio_decoder_parallel profile')
    if (any(parallel_key[:3]) or parallel_key[-1] != 'none') and (
            not portable and (backend != 'cuda' or getattr(args, 'task_type', None) not in ('t2va', 'fl2va', 'ref2va')
            or (getattr(args, 'h3_parallel_mode', 'sp') == 'sp' and getattr(args, 'sp', 1) != 8))):
        raise ValueError('H3 condition parallelism requires a CUDA FP8 H3 SP8 profile')
    if getattr(args, 'h3_cuda_reference_offload', False) and (
            (not portable and backend != 'cuda') or getattr(args, 'task_type', None) != 'ref2va'
            or not getattr(args, 'h3_cuda_conditioning_offload', False)):
        raise ValueError('h3_cuda_reference_offload requires CUDA FP8 Ref2VA with h3_cuda_conditioning_offload')
    from .h3_cuda_adaln import resolve_assets
    resolve_assets(args, backend, requested)
    args.h3_cuda_precision = requested
    from .h3_vae_config import resolve_vae_backend
    resolve_vae_backend(args)
    if requested in ('fp8-bf16', 'fp8-sage'):
        if getattr(args, 'phaa_num', 0) == 0:
            args.phaa_num = 2
        args.h3_cuda_vae_sp = True
    return requested


def validate(args, backend):
    profile = getattr(args, 'h3_cuda_precision', 'none')
    if profile == 'none':
        return
    new_parallel = getattr(args, 'h3_parallel_mode', 'sp') in ('tp_cp', 'tp_cp_shared', 'tp_sp')
    if new_parallel:
        from .h3_parallel_config import H3ParallelConfig
        H3ParallelConfig.from_args(args, backend, dist.get_world_size() if dist.is_initialized() else 1)
    tasks = ('t2va', 'fl2va', 'ref2va')
    if (backend != 'cuda' or args.model != 'MiniMax-H3' or args.dtype != 'bf16'
            or args.task_type not in tasks
            or (not new_parallel and args.sp != 8)
            or not dist.is_initialized() or dist.get_world_size() != 8 or args.text_encoder_tp != 8):
        layout = ('world=text TP=8 and a valid DiT TP/CP layout' if new_parallel else 'world=SP=text TP=8')
        raise ValueError(f'{profile} requires CUDA BF16 {"/".join(tasks)} and {layout}')
    conflicts = ('h3_nvfp4_checkpoint', 'h3_nvfp4_resident', 'h3_nvfp4_compile',
                 'h3_turbo8', 'h3_turbo_sla_lora', 'h3_refiner', 'lora_path_list',
                 'fsdp', 'inf_vram_blocks_num', 'cache_dit')
    if any(getattr(args, name, None) for name in conflicts) or args.atten != 'none' or args.matmul_quant != 'none':
        raise ValueError(f'{profile} cannot combine with packed weights, LoRA, caches or other attention/quantization recipes; use --h3_cuda_precision none')
    from .h3_cuda_adaln import validate_options
    validate_options(args)
    if not new_parallel and args.h3_cuda_precision == 'fp8-bf16' and getattr(args, 'h3_cuda_phaa_fp8', os.environ.get('XGEN_PHAA_FP8') == '1'):
        raise ValueError('fp8-bf16 requires BF16 communication; unset XGEN_PHAA_FP8')
    if type(args.phaa_num) is not int or args.phaa_num < 1:
        raise ValueError(f'{profile} requires PHAA; the recommended value is 2')
    if not getattr(args, 'h3_cuda_vae_sp', False):
        raise ValueError(f'{profile} requires --h3_cuda_vae_sp')


def validate_manager(manager, args, backend):
    """A resident HTTP worker cannot silently change its precision recipe."""
    profile = resolve(args, backend)
    from ..framework.vae.omni_decoder import validate_options
    validate_options(args, backend)
    if manager.pipe is not None:
        installed = getattr(manager.pipe, '_h3_vae_decode_backend', None)
        selected = getattr(args, 'h3_vae_backend', 'legacy')
        if selected != getattr(manager.pipe, '_h3_vae_backend_name', 'legacy'):
            raise ValueError('Restart the service to change H3 VAE decode backend')
        if (installed is not None) != (selected == 'native-stream') or (installed is not None and
                str(installed.path) != str(args.h3_vae_model_path)):
            raise ValueError('Restart the service to change H3 VAE decode backend or checkpoint')
    pair_tp = getattr(args, 'h3_cuda_linear_tp', 1)
    if manager.pipe is not None and pair_tp != getattr(manager, '_h3_cuda_linear_tp_key', 1):
        raise ValueError('Restart the service to change H3 linear TP')
    condition_key = condition_parallel_key(args)
    if manager.pipe is not None and condition_key != getattr(manager, '_h3_condition_parallel_key', (False, False, False, 'none')):
        raise ValueError('Restart the service to change H3 condition parallelism')
    input_refiner = getattr(args, 'h3_cuda_input_refiner', 'none')
    precompute = bool(getattr(args, 'h3_cuda_precompute_conditioning', False))
    if manager.pipe is not None and precompute != getattr(manager, '_h3_precompute_conditioning_key', False):
        raise ValueError('Restart the service to change h3_cuda_precompute_conditioning')
    if manager.pipe is not None and input_refiner != getattr(manager, '_h3_cuda_input_refiner_key', 'none'):
        raise ValueError('Restart the service to change h3_cuda_input_refiner')
    refiner_chunk = getattr(args, 'h3_cuda_refiner_ff_chunk_size', 0)
    reference_offload = bool(getattr(args, 'h3_cuda_reference_offload', False))
    if manager.pipe is not None and reference_offload != getattr(manager, '_h3_cuda_reference_offload_key', False):
        raise ValueError('Restart the service to change h3_cuda_reference_offload')
    if manager.pipe is not None and refiner_chunk != getattr(manager, '_h3_cuda_refiner_ff_chunk_key', 0):
        raise ValueError('Restart the service to change h3_cuda_refiner_ff_chunk_size')
    local_output = bool(getattr(args, 'h3_cuda_local_output', False))
    if manager.pipe is not None and local_output != getattr(manager, '_h3_cuda_local_output_key', False):
        raise ValueError('Restart the service to change h3_cuda_local_output')
    conditioning_offload = bool(getattr(args, 'h3_cuda_conditioning_offload', False))
    if manager.pipe is not None and conditioning_offload != getattr(manager, '_h3_cuda_conditioning_offload_key', False):
        raise ValueError('Restart the service to change h3_cuda_conditioning_offload')
    from .h3_parallel_config import H3ParallelConfig
    parallel_key = H3ParallelConfig.from_args(args, backend, dist.get_world_size() if dist.is_initialized() else 1)
    previous_parallel = getattr(manager, '_h3_parallel_key', H3ParallelConfig())
    if manager.pipe is not None and parallel_key != previous_parallel:
        raise ValueError('Restart the service to change H3 TP/CP layout or transport dtype')
    previous = getattr(manager, '_h3_cuda_precision_key', None)
    if manager.pipe is not None and previous != profile and (previous in ('fp8-bf16', 'fp8-sage') or profile in ('fp8-bf16', 'fp8-sage')):
        raise ValueError('Restart the service to change h3_cuda_precision')
    if profile != 'none':
        validate(args, backend)
        if manager.pipe is not None and bool(getattr(args, 'adaln_cache_file', None)) != bool(getattr(getattr(manager.pipe, 'transformer', None), '_h3_cuda_adaln_table', False)):
            raise ValueError('Restart the manager to change CUDA AdaLN table mode')
    from .h3_cuda_options import startup_key
    key = startup_key(args)
    from .h3_features import enabled as portable_enabled
    portable_now = portable_enabled(args)
    previous_portable_key = getattr(manager, '_h3_portable_startup_key', None)
    if manager.pipe is not None and portable_now and previous not in ('fp8-bf16', 'fp8-sage'):
        if previous_portable_key is None or previous_portable_key != key:
            raise ValueError('Restart the service to enable or change portable H3 startup options')
    cuda_resident = (profile in ('fp8-bf16', 'fp8-sage') or previous in ('fp8-bf16', 'fp8-sage')
                     or portable_now or getattr(manager, '_h3_portable_enabled', False))
    manager._h3_portable_enabled = portable_now
    if cuda_resident and manager.pipe is not None and getattr(manager, '_h3_cuda_options_key', key) != key:
        raise ValueError('Restart the manager/service to change H3 CUDA startup options')
    from .h3_cuda_options import commit_environment
    commit_environment(args)
    if cuda_resident:
        manager._h3_cuda_options_key = key
    if getattr(args, 'model', None) == 'MiniMax-H3':
        manager._h3_portable_startup_key = key
    manager._h3_cuda_linear_tp_key = pair_tp
    manager._h3_cuda_precision_key = profile
    manager._h3_condition_parallel_key = condition_key
    manager._h3_parallel_key = parallel_key
    manager._h3_cuda_local_output_key = local_output
    manager._h3_cuda_conditioning_offload_key = conditioning_offload
    manager._h3_cuda_refiner_ff_chunk_key = refiner_chunk
    manager._h3_cuda_reference_offload_key = reference_offload
    manager._h3_cuda_input_refiner_key = input_refiner
    manager._h3_precompute_conditioning_key = precompute
    logger.info('H3 CUDA effective startup options=%s', dict(key))


def prepare_transformer(pipe, args, device):
    from x_base.operator.h3_adaln_owner import H3OwnedAdaLN, check_shared_timestep
    from x_base.operator.h3_dense_cuda import H3DenseCUDA
    from x_base.operator.h3_fp8 import H3FP8Linear

    model = pipe.transformer
    table_mode = bool(getattr(args, 'adaln_cache_file', None))
    if table_mode:
        from pathlib import Path
        from .h3_cuda_adaln import assert_weights_absent
        assert_weights_absent(model)
        model._h3_cuda_adaln_table = True
        model._h3_cuda_adaln_checkpoint = str(Path(args.transformer_path).resolve())
        model._h3_cuda_adaln_load_count = 0
    config = None
    if getattr(args, 'h3_parallel_mode', 'sp') in ('tp_cp', 'tp_cp_shared', 'tp_sp'):
        from .h3_parallel import initialize
        config = initialize(model, args)
    if args.h3_cuda_precision == 'fp8-sage':
        from x_base.operator.h3_sage_cuda import H3SageCUDA

        attention_backend = H3SageCUDA()
    else:
        attention_backend = H3DenseCUDA()
    converted = 0
    for index, block in enumerate(model.transformer_blocks):
        attn = block.attn
        if attn.fused_projections or not all(isinstance(getattr(attn, name), nn.Linear) for name in ('to_q','to_k','to_v')):
            raise ValueError('FP8 recipe expects original unfused BF16 H3 projections')
        projections = (attn.to_q, attn.to_k, attn.to_v)
        biases = [layer.bias for layer in projections]
        if any(b is None for b in biases) and not all(b is None for b in biases):
            raise ValueError('H3 QKV biases must either all exist or all be absent')
        if config is not None and config.mode == 'tp_cp_shared':
            from .h3_shared_parallel import prepare_shared_block
            prepare_shared_block(block, model._h3_shared_tp_manager,
                                 model._h3_shared_sequence_exchange, chunks=args.phaa_num)
        elif config is not None and config.mode != 'tp_sp' and config.tp > 1:
            from .h3_parallel import prepare_block
            prepare_block(block, model.parallel_manager, config)
        else:
            attn.to_qkv = H3FP8Linear(torch.cat([layer.weight.detach() for layer in projections]),
                                    torch.cat(biases) if biases[0] is not None else None)
            attn.to_q = attn.to_k = attn.to_v = None
            attn.fused_projections = True
            targets = ((attn.to_out, '0'), (block.ff.net[0], 'proj'), (block.ff.net, '2'))
            for parent, name in targets:
                original = getattr(parent, name)
                setattr(parent, name, H3FP8Linear(original.weight, original.bias))
        converted += 4
        if not table_mode:
            block.adaln_proj = H3OwnedAdaLN(block.adaln_proj, index, exact_cache=True)
        attn._h3_attention_backend = attention_backend
    if not table_mode:
        model.time_embedder.register_forward_hook(check_shared_timestep)
    # These fusions preserve the ordinary BF16/FP8 Linear interfaces. Packed FP4 FFN fusions do not.
    from .h3_cuda import install_dit_fusions

    flags = frozenset(('XGEN_H3_FUSED_INDEXED_AFFINE','XGEN_H3_FUSED_PARTIAL_ROPE','XGEN_H3_FUSED_QK_RMS'))
    if os.environ.get('XGEN_H3_FUSED_OUTPUT_ADALN') == '1':
        flags |= {'XGEN_H3_FUSED_OUTPUT_ADALN'}
    install_dit_fusions(model, flags)
    resident = len(model.transformer_blocks)
    expected_blocks = model.config.num_layers if config is not None else 50
    if resident != expected_blocks or resident < 1:
        raise ValueError(f'{args.h3_cuda_precision} expects the complete 50-block H3 Base checkpoint')
    input_refiner = getattr(args, 'h3_cuda_input_refiner', 'none')
    if input_refiner != 'none':
        from .h3_input_refiner import install_input_refiner
        install_input_refiner(model, input_refiner, sp=getattr(args, 'h3_cuda_input_refiner_sp', None),
                              attention=getattr(args, 'h3_cuda_input_refiner_attention', None),
                              ffn=getattr(args, 'h3_cuda_input_refiner_ffn', None))
    if getattr(args, 'h3_cuda_precompute_conditioning', False):
        model._h3_precompute_conditioning = True
    if config is not None and config.mode == 'tp_sp':
        from .h3_pair_parallel import install_pair_parallel
        install_pair_parallel(model)
    from .h3_cuda_options import local_io_default
    for name in ('h3_cuda_local_input', 'h3_cuda_local_output'):
        value = getattr(args, name, None)
        setattr(model, '_'+name, local_io_default(getattr(config, 'mode', 'sp')) if value is None else value)
    if model._h3_cuda_local_input:
        for block in model.transformer_blocks:
            block.attn._h3_tp_cp = True
    model.to(device)
    if getattr(args, 'h3_cuda_refiner_ff_chunk_size', 0):
        from .h3_cuda_refiner import install_ff_chunking
        install_ff_chunking(model, args.h3_cuda_refiner_ff_chunk_size)
    model._h3_cuda_precision = args.h3_cuda_precision
    if args.h3_cuda_precision == 'fp8-sage':
        model._h3_sage_cuda = attention_backend
    else:
        model._h3_dense_cuda = attention_backend
    model._h3_cuda_resident_blocks = resident
    if getattr(args, 'h3_cuda_local_output', False):
        model._h3_cuda_local_output = True
        logger.info('Experimental H3 SP local output enabled: FP32 prediction gather, original block loop unchanged')
    logger.info('H3 CUDA precision=%s FP8_main_projections=%d resident_blocks=%d AdaLN=%s GEMM=torch._scaled_mm',
                args.h3_cuda_precision, converted, resident, 'BF16 table, weights absent' if table_mode else 'rank-owner exact_cache=True')


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
                and child.weight.device.type in ('cuda', 'npu')):
            child.weight=child.weight.to('cpu')


def prepare_vaes(pipe,args,device):
    from diffusers.hooks import apply_group_offloading
    from ..framework.vae.h3_cuda_chunks import install_cpu_decode_chunks,install_staged_decoder

    omni_decode = getattr(args, 'h3_vae_backend', 'legacy') == 'native-stream'
    if not omni_decode:
        install_cpu_decode_chunks(pipe.vae)
        pipe._h3_cuda_cpu_decode_chunks=True
    apply_group_offloading(pipe.vae,offload_type='leaf_level',onload_device=device,
                           offload_device=torch.device('cpu'),use_stream=False,low_cpu_mem_usage=True)
    if not omni_decode:
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
    def prepare_video_decode():
        import gc
        import time

        if getattr(getattr(pipe, 'transformer', None), '_h3_precompute_conditioning', False):
            from .h3_conditioning import release_conditioning
            release_conditioning(pipe.transformer)
        started=time.perf_counter()
        torch.cuda.synchronize()
        before=torch.cuda.memory_reserved()
        gc.collect()
        torch.cuda.empty_cache()
        event=dict(seconds=time.perf_counter()-started,reserved_before=before,
                   reserved_after=torch.cuda.memory_reserved(),allocated=torch.cuda.memory_allocated())
        pipe._h3_cuda_decode_trim=event
        logger.info('H3 CUDA before video VAE decode: unused memory reclaimed, DiT retained: %s',event)
    # Both old and native decoders must release request-scoped conditioning
    # and reclaim unused allocations before staging decoder weights.
    pipe._h3_prepare_video_decode=prepare_video_decode
    @wraps(original)
    def decode(*values,**kwargs):
        pipe._h3_prepare_video_decode()
        return original(*values,**kwargs)
    pipe.vae.decode=decode
    if getattr(args, 'h3_parallel_mode', 'sp') in ('tp_cp', 'tp_cp_shared', 'tp_sp') or getattr(args, 'task_type', None) == 'ref2va':
        from .h3_parallel import stage_reference_audio
        stage_reference_audio(pipe, device)
    if getattr(args, 'h3_cuda_conditioning_offload', False):
        from .h3_cuda_staging import install_conditioning_staging
        install_conditioning_staging(pipe, device, defer_until_dit=bool(getattr(args, 'h3_cuda_reference_offload', False)))
    if any(condition_parallel_key(args)[:3]) or condition_parallel_key(args)[-1] != 'none':
        from .h3_condition_parallel import install_condition_parallel
        install_condition_parallel(pipe, args, device)
    logger.info('H3 CUDA VAE backend=%s; DiT resident blocks remain on device through decode',
                getattr(args, 'h3_vae_backend', 'legacy'))


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
