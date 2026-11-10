"""Pure configuration for optional CUDA H3 features; no model/device imports."""
import argparse
import logging
import os

logger = logging.getLogger(__name__)
PRESETS = ('none', 'sage-fp8-full-parallel')
BOOL_OPTIONS = (
    'h3_cuda_local_input', 'h3_cuda_local_output', 'h3_cuda_precompute_conditioning',
    'h3_cuda_vision_sp', 'h3_cuda_vae_encoder_parallel', 'h3_cuda_audio_encoder_parallel',
    'h3_cuda_conditioning_offload', 'h3_cuda_reference_offload',
    'h3_cuda_phaa_fp8', 'h3_cuda_phaa_direct_pack', 'h3_cuda_input_refiner_sp',
)
STARTUP_OPTIONS = BOOL_OPTIONS + (
    'h3_cuda_preset', 'h3_cuda_dit_residency', 'h3_cuda_input_refiner',
    'h3_cuda_input_refiner_attention', 'h3_cuda_input_refiner_ffn',
    'h3_cuda_audio_decoder_parallel', 'h3_cuda_refiner_ff_chunk_size', 'h3_features',
)


def add_arguments(parser, *, service=False):
    group = parser.add_argument_group('Composable CUDA H3 options')
    group.add_argument('--h3_cuda_preset', choices=PRESETS, default=None,
                       help='Explicit cuda-5090 preset; omitted preserves existing defaults')
    group.add_argument('--h3_cuda_dit_residency', choices=('inherit', 'resident', 'conditioning', 'reference'), default=None,
                       help='DiT placement during conditioning; resident retains all blocks on GPU')
    group.add_argument('--h3_cuda_local_input', action=argparse.BooleanOptionalAction, default=None,
                       help='Project only local SP video/audio rows; independent of SAP output')
    group.add_argument('--h3_cuda_phaa_fp8', action=argparse.BooleanOptionalAction, default=None,
                       help='Override PHAA FP8 wire; omitted inherits XGEN_PHAA_FP8')
    group.add_argument('--h3_cuda_phaa_direct_pack', action=argparse.BooleanOptionalAction, default=None,
                       help='Read mixed QKV strides directly, avoiding BF16 contiguous copies')
    group.add_argument('--h3_cuda_input_refiner_sp', action=argparse.BooleanOptionalAction, default=None)
    group.add_argument('--h3_cuda_input_refiner_attention', choices=('legacy', 'cudnn'), default=None)
    group.add_argument('--h3_cuda_input_refiner_ffn', choices=('bf16', 'fp8'), default=None)
    if service:
        for name in BOOL_OPTIONS:
            if name not in ('h3_cuda_local_input', 'h3_cuda_phaa_fp8', 'h3_cuda_phaa_direct_pack', 'h3_cuda_input_refiner_sp'):
                group.add_argument('--'+name, action=argparse.BooleanOptionalAction, default=None)
        group.add_argument('--h3_cuda_input_refiner', choices=('none', 'cudnn-sp', 'cudnn-sp-fp8'), default=None)
        group.add_argument('--h3_cuda_audio_decoder_parallel', choices=('none', 'long'), default=None)
        group.add_argument('--h3_cuda_refiner_ff_chunk_size', type=int, default=None)
    return parser


def local_io_default(mode):
    return mode in ('tp_cp', 'tp_cp_shared')


def resolve_options(args, backend):
    """Expand an explicit preset only into unspecified fields, then canonicalize."""
    if not hasattr(args, '_h3_residency_explicit'):
        args._h3_residency_explicit = getattr(args, 'h3_cuda_dit_residency', None) not in (None, 'inherit')
    from .h3_features import resolve_defaults
    resolve_defaults(args)
    preset = getattr(args, 'h3_cuda_preset', None) or 'none'
    if preset not in PRESETS:
        raise ValueError('Unknown H3 CUDA preset')
    full = preset == 'sage-fp8-full-parallel'
    environment = {}
    if full:
        if backend != 'cuda' or getattr(args, 'model', None) != 'MiniMax-H3' or getattr(args, 'task_type', None) != 'ref2va':
            raise ValueError('sage-fp8-full-parallel requires CUDA H3 Ref2VA')
        if getattr(args, 'h3_cuda_precision', 'auto') == 'auto':
            args.h3_cuda_precision = 'fp8-sage'
        for key, value in (('NCCL_MAX_CTAS', '2'), ('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'),
                           ('XGEN_H3_FUSED_OUTPUT_ADALN', '1')):
            environment[key] = os.environ.get(key, value)
    args.h3_cuda_preset = preset
    mode = getattr(args, 'h3_parallel_mode', 'sp')
    alias = getattr(args, 'h3_cuda_linear_tp', 1)
    if type(alias) is not int or alias not in (1, 2):
        raise ValueError('h3_cuda_linear_tp must be 1 or 2')
    if alias == 2:
        if backend != 'cuda' or getattr(args, 'sp', 1) != 8:
            raise ValueError('h3_cuda_linear_tp=2 requires CUDA H3 FP8 and original SP8 mode')
        if mode not in ('sp', 'tp_sp') or getattr(args, 'h3_dit_tp', 1) not in (1, 2):
            raise ValueError('h3_cuda_linear_tp=2 conflicts with the selected DiT parallel layout')
        args.h3_parallel_mode, args.h3_dit_tp = mode = 'tp_sp', 2
        args.h3_cuda_linear_tp = 1  # The experimental alias resolves to one layout.

    defaults = dict(h3_cuda_local_input=local_io_default(mode),
                    h3_cuda_local_output=local_io_default(mode))
    for key in BOOL_OPTIONS:
        if key in ('h3_cuda_phaa_fp8', 'h3_cuda_phaa_direct_pack', 'h3_cuda_input_refiner_sp'):
            continue
        value = getattr(args, key, None)
        if value is None:
            value = defaults.get(key, False) or (full and key in (
                'h3_cuda_local_input', 'h3_cuda_local_output', 'h3_cuda_precompute_conditioning',
                'h3_cuda_vision_sp', 'h3_cuda_vae_encoder_parallel', 'h3_cuda_audio_encoder_parallel'))
        if type(value) is not bool:
            raise ValueError(key+' must be boolean')
        setattr(args, key, value)

    old = getattr(args, 'h3_cuda_input_refiner', None)
    legacy = {'none': (False, 'legacy', 'bf16'), 'cudnn-sp': (True, 'cudnn', 'bf16'),
              'cudnn-sp-fp8': (True, 'cudnn', 'fp8')}
    if old not in (None, 'custom', *legacy):
        raise ValueError('Unknown h3_cuda_input_refiner profile')
    inherited = legacy.get(old, (True, 'cudnn', 'fp8') if full else legacy['none'])
    for key, default in zip(('h3_cuda_input_refiner_sp', 'h3_cuda_input_refiner_attention', 'h3_cuda_input_refiner_ffn'), inherited):
        value = getattr(args, key, None)
        setattr(args, key, default if value is None else value)
    if type(args.h3_cuda_input_refiner_sp) is not bool or args.h3_cuda_input_refiner_attention not in ('legacy', 'cudnn') or args.h3_cuda_input_refiner_ffn not in ('bf16', 'fp8'):
        raise ValueError('Invalid input Token Refiner SP/attention/FFN configuration')
    effective = (args.h3_cuda_input_refiner_sp, args.h3_cuda_input_refiner_attention, args.h3_cuda_input_refiner_ffn)
    args.h3_cuda_input_refiner = next((name for name, value in legacy.items() if value == effective), 'custom')
    if getattr(args, 'h3_cuda_audio_decoder_parallel', None) is None:
        args.h3_cuda_audio_decoder_parallel = 'long' if full else 'none'

    residency = getattr(args, 'h3_cuda_dit_residency', None) or 'inherit'
    if residency not in ('inherit', 'resident', 'conditioning', 'reference'):
        raise ValueError('Invalid H3 DiT residency policy')
    if residency != 'inherit':
        expected = (residency != 'resident', residency == 'reference')
        actual = (args.h3_cuda_conditioning_offload, args.h3_cuda_reference_offload)
        if any(actual) and actual != expected:
            raise ValueError('DiT residency conflicts with conditioning/reference offload flags')
        args.h3_cuda_conditioning_offload, args.h3_cuda_reference_offload = expected
    args.h3_cuda_dit_residency = ('reference' if args.h3_cuda_reference_offload else
                                'conditioning' if args.h3_cuda_conditioning_offload else 'resident')
    for key, env, default in (
        ('h3_cuda_phaa_fp8', 'XGEN_PHAA_FP8', full and getattr(args, 'h3_cuda_precision', 'none') != 'fp8-bf16'),
        ('h3_cuda_phaa_direct_pack', 'XGEN_PHAA_FP8_MIXED_STRIDES', full),
    ):
        value = getattr(args, key, None)
        explicit = value is not None
        if value is None:
            value = default if full else backend == 'cuda' and os.environ.get(env, '0') == '1'
            if key == 'h3_cuda_phaa_fp8' and mode != 'sp' and getattr(args, 'h3_cp_comm', 'auto') != 'auto':
                value = args.h3_cp_comm == 'fp8'
            if key == 'h3_cuda_phaa_direct_pack' and full and not args.h3_cuda_phaa_fp8:
                value = False
            if key == 'h3_cuda_phaa_direct_pack' and mode == 'tp_cp_shared':
                value = False
        if type(value) is not bool:
            raise ValueError(key+' must be boolean')
        setattr(args, key, value)
        if (explicit or full) and backend == 'cuda':
            environment[env] = '1' if value else '0'
    cp_wire = getattr(args, 'h3_cp_comm', 'auto')
    if mode != 'sp' and cp_wire != 'auto' and args.h3_cuda_phaa_fp8 != (cp_wire == 'fp8'):
        raise ValueError('Explicit PHAA FP8 setting conflicts with h3_cp_comm')
    if args.h3_cuda_phaa_direct_pack and (not args.h3_cuda_phaa_fp8 or mode == 'tp_cp_shared'):
        raise ValueError('Direct PHAA packing requires FP8 PHAA; it is not applicable to shared TP8/CP8')
    args._h3_cuda_env_updates = environment
    if backend != 'cuda' and (args.h3_cuda_phaa_fp8 or args.h3_cuda_phaa_direct_pack):
        raise ValueError('H3 FP8 PHAA options require CUDA')
    return {name: getattr(args, name, None) for name in STARTUP_OPTIONS}


def bootstrap_preset_environment(preset):
    """Fresh-process allocator/NCCL defaults; no request-side mutation."""
    if preset == 'sage-fp8-full-parallel':
        for name, value in (('NCCL_MAX_CTAS', '2'), ('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')):
            os.environ.setdefault(name, value)


def startup_key(args):
    """Only startup state; duration, prompt and reference media remain per request."""
    values = tuple((name, getattr(args, name, None)) for name in STARTUP_OPTIONS)
    runtime = tuple((name, getattr(args, name, None)) for name in (
        'sp', 'text_encoder_tp', 'dtype', 'atten', 'matmul_quant', 'vae_matmul_quant',
        'phaa_num', 'rope_fused', 'transformer_path', 'adaln_cache_file'))
    return values + runtime + (('_h3_residency_explicit', bool(getattr(args, '_h3_residency_explicit', False))),
                     ('model', getattr(args, 'model', None)),
                     ('weights', getattr(args, 'pretrained_model_name_or_path', None)),
                     ('workflow', 'ref2va' if getattr(args, 'task_type', None) == 'ref2va' else 'fl2va'))


def needs_resolution(args):
    return (getattr(args, 'h3_features', None) not in (None, 'none')
            or getattr(args, 'h3_cuda_precision', 'auto') != 'none'
            or any(getattr(args, key, None) is not None for key in BOOL_OPTIONS)
            or getattr(args, 'h3_cuda_preset', None) not in (None, 'none')
            or getattr(args, 'h3_cuda_dit_residency', None) not in (None, 'inherit')
            or getattr(args, 'h3_cuda_input_refiner', None) not in (None, 'none')
            or getattr(args, 'h3_cuda_input_refiner_attention', None) not in (None, 'legacy')
            or getattr(args, 'h3_cuda_input_refiner_ffn', None) not in (None, 'bf16')
            or getattr(args, 'h3_cuda_audio_decoder_parallel', None) not in (None, 'none')
            or bool(getattr(args, 'h3_cuda_refiner_ff_chunk_size', 0)))


def commit_environment(args):
    """Call only after configuration validation, or before a fresh CLI process initializes CUDA."""
    for name, value in getattr(args, '_h3_cuda_env_updates', {}).items():
        os.environ[name] = value
