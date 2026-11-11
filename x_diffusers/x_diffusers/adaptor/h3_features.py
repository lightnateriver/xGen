"""Hardware-neutral H3 scheduling options; CUDA spellings remain compatible."""
import logging

logger = logging.getLogger(__name__)

BOOL_FIELDS = ('local_input', 'local_output', 'precompute_conditioning', 'vision_sp',
               'vae_encoder_parallel', 'audio_encoder_parallel', 'input_refiner_sp',
               'conditioning_offload', 'reference_offload')
VALUE_FIELDS = ('dit_residency', 'input_refiner_attention', 'input_refiner_ffn',
                'audio_decoder_parallel', 'refiner_ff_chunk_size')
ALIASES = {'h3_'+name: 'h3_cuda_'+name for name in BOOL_FIELDS + VALUE_FIELDS}


def add_aliases(parser):
    """Aliases use one argparse destination and preserve explicit --no-* values."""
    if '--h3_features' not in parser._option_string_actions:
        parser.add_argument('--h3_features', choices=('none', 'full-parallel'), default=None,
                            help='Portable H3 T2VA/FL2VA/Ref2VA scheduling; preserves operator precision')
    for public, legacy in ALIASES.items():
        for prefix in ('--', '--no-'):
            old, new = prefix+legacy, prefix+public
            action = parser._option_string_actions.get(old)
            if action is not None and new not in parser._option_string_actions:
                action.option_strings.append(new)
                parser._option_string_actions[new] = action
    return parser


def normalize_mapping(values):
    result = dict(values)
    for public, legacy in ALIASES.items():
        if public in result:
            if legacy in result and result[legacy] != result[public]:
                raise ValueError('Conflicting H3 option aliases: '+public+' / '+legacy)
            result[legacy] = result.pop(public)
    return result


def resolve_defaults(args):
    if getattr(args, 'h3_features', None) not in (None, 'none', 'full-parallel'):
        raise ValueError('Unknown portable H3 feature preset')
    if getattr(args, 'h3_features', None) != 'full-parallel':
        return
    for name in BOOL_FIELDS:
        key = 'h3_cuda_'+name
        if name not in ('conditioning_offload', 'reference_offload') and getattr(args, key, None) is None:
            setattr(args, key, True)
    for name, value in (('input_refiner_attention', 'legacy'), ('input_refiner_ffn', 'bf16'),
                        ('audio_decoder_parallel', 'long'), ('dit_residency', 'resident')):
        key = 'h3_cuda_'+name
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def enabled(args):
    return (any(getattr(args, 'h3_cuda_'+name, False) for name in BOOL_FIELDS)
            or getattr(args, 'h3_cuda_audio_decoder_parallel', None) == 'long'
            or bool(getattr(args, 'h3_cuda_refiner_ff_chunk_size', 0))
            or getattr(args, 'h3_cuda_input_refiner', None) not in (None, 'none')
            or bool(getattr(args, '_h3_residency_explicit', False))
            or getattr(args, 'h3_parallel_mode', 'sp') != 'sp'
            or getattr(args, 'h3_features', None) == 'full-parallel')


def validate(args, backend):
    if not enabled(args):
        return
    if backend not in ('cuda', 'npu') or getattr(args, 'model', None) != 'MiniMax-H3':
        raise ValueError('Portable H3 scheduling requires CUDA or NPU MiniMax-H3')
    if getattr(args, 'dtype', 'bf16') != 'bf16':
        raise ValueError('Portable H3 scheduling currently requires BF16 module interfaces')
    if getattr(args, 'task_type', None) not in ('t2va', 'fl2va', 'ref2va'):
        raise ValueError('Portable H3 scheduling requires an H3 workflow')
    if getattr(args, 'h3_parallel_mode', 'sp') == 'sp' and getattr(args, 'sp', 1) not in (2, 4, 8):
        raise ValueError('Portable H3 component parallelism requires SP2/4/8')
    if getattr(args, 'fsdp', None) or int(getattr(args, 'inf_vram_blocks_num', 0) or 0):
        raise ValueError('Portable scheduling owns residency; disable FSDP/infinite-vram hooks')
    if getattr(args, 'h3_refiner', False) and any(getattr(args, 'h3_cuda_'+name, False)
            for name in ('conditioning_offload', 'reference_offload')):
        raise ValueError('The 2K Refiner coordinator owns residency; disable standalone conditioning/reference offload')
    if backend == 'npu':
        if getattr(args, 'h3_cuda_input_refiner_attention', 'legacy') != 'legacy':
            raise ValueError('cuDNN input refiner attention is CUDA-only; use legacy on NPU')
        if getattr(args, 'h3_cuda_input_refiner_ffn', 'bf16') != 'bf16':
            raise ValueError('FP8 input-refiner GEMM is CUDA-only; use bf16 on NPU')


def configure_model(pipe, args, device):
    """Install scheduling independently of quantization and attention kernels."""
    import torch
    from .h3_parallel import initialize, prepare_block
    from .h3_shared_parallel import prepare_shared_block
    from .h3_cuda_precision import retain_module

    model = pipe.transformer
    config = None
    if getattr(args, 'h3_parallel_mode', 'sp') != 'sp':
        config = initialize(model, args)
        from x_base import attention_manager
        def attention(q, k, v, *, attn_mask=None, **kwargs):
            return attention_manager.attention(q, k, v, attn_mask=attn_mask, dropout_p=0., is_causal=False)
        for block in model.transformer_blocks:
            block.attn._h3_attention_backend = attention
            if config.mode == 'tp_cp_shared':
                prepare_shared_block(block, model._h3_shared_tp_manager,
                                     model._h3_shared_sequence_exchange, fp8=False, chunks=args.phaa_num)
            elif config.mode != 'tp_sp' and config.tp > 1:
                prepare_block(block, model.parallel_manager, config, fp8=False)
            elif config.mode == 'tp_sp':
                # The pair adapter expects fused QKV before slicing.
                block.attn.fuse_projections()
        if config.mode == 'tp_sp':
            from .h3_pair_parallel import install_pair_parallel
            install_pair_parallel(model)
    for name in ('local_input', 'local_output'):
        setattr(model, '_h3_cuda_'+name, bool(getattr(args, 'h3_cuda_'+name, False)))
    if model._h3_cuda_local_input:
        for block in model.transformer_blocks:
            block.attn._h3_tp_cp = True
    model._h3_precompute_conditioning = bool(getattr(args, 'h3_cuda_precompute_conditioning', False))
    if getattr(args, 'h3_cuda_input_refiner', 'none') != 'none':
        from .h3_input_refiner import install_input_refiner
        install_input_refiner(model, args.h3_cuda_input_refiner,
            sp=args.h3_cuda_input_refiner_sp, attention=args.h3_cuda_input_refiner_attention,
            ffn=args.h3_cuda_input_refiner_ffn)
    if getattr(args, 'h3_cuda_refiner_ff_chunk_size', 0):
        from .h3_cuda_refiner import install_ff_chunking
        install_ff_chunking(model, args.h3_cuda_refiner_ff_chunk_size)
    # Keep original modules/quantization; remove competing offload hooks only
    # after existing quantization has run. Subsequent requests reuse weights.
    if not getattr(args, 'h3_refiner', False):
        retain_module(model, torch.device(device))
    model._h3_portable_scheduling = True
    logger.info('H3 portable scheduling installed: backend=%s mode=%s conditioning=%s',
                torch.device(device).type, getattr(args, 'h3_parallel_mode', 'sp'),
                model._h3_precompute_conditioning)


def configure_components(pipe, args, device):
    import torch
    from functools import wraps
    from .h3_condition_parallel import install_condition_parallel
    from .h3_cuda_staging import install_conditioning_staging

    coordinator_owned = bool(getattr(args, 'h3_refiner', False))
    if not coordinator_owned and getattr(args, 'h3_cuda_vae_encoder_parallel', False):
        # Quantization may create CPU decoder buffers after the loader's VAE
        # transfer. The encoder's root offload hook makes pipeline.to() skip
        # this complete VAE, so finish the native decoder transfer first.
        pipe.vae.decoder.to(device)
        pipe.vae.post_quant_conv.to(device)
    install_condition_parallel(pipe, args, device, stage_weights=not coordinator_owned)
    if not coordinator_owned and (getattr(args, 'h3_cuda_audio_encoder_parallel', False)
            or getattr(args, 'h3_cuda_audio_decoder_parallel', 'none') != 'none'):
        # Reference encoding returns audio weights to CPU. The native decoder
        # has no onload hook, unlike the CUDA FP8 recipe's decoder wrapper.
        from .h3_cuda_precision import offload_audio
        audio = pipe.audio_vae
        original_audio_decode = audio.decode
        @wraps(original_audio_decode)
        def decode_audio(*values, **kwargs):
            audio.to(torch.device(device))
            try:
                return original_audio_decode(*values, **kwargs)
            finally:
                offload_audio(audio)
        audio.decode = decode_audio
    if getattr(args, 'h3_cuda_conditioning_offload', False):
        install_conditioning_staging(pipe, device,
            defer_until_dit=bool(getattr(args, 'h3_cuda_reference_offload', False)))
    original = pipe.vae.decode
    def release():
        from .h3_conditioning import release_conditioning
        release_conditioning(pipe.transformer)
    previous = getattr(pipe, '_h3_prepare_video_decode', None)
    def prepare():
        release()
        if previous is not None:
            previous()
    pipe._h3_prepare_video_decode = prepare
    @wraps(original)
    def decode(*values, **kwargs):
        release()
        return original(*values, **kwargs)
    pipe.vae.decode = decode
