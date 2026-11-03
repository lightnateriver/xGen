"""CPU-only opt-in contract shared by the existing CLI and video service."""
from contextlib import contextmanager
from pathlib import Path
import os


def add_arguments(parser):
    group = parser.add_argument_group('Experimental H3 Refiner (disabled by default)')
    group.add_argument('--h3_refiner', action='store_true', help='EXPERIMENTAL: H3 latent -> Regenerate 2K; restart to disable')
    group.add_argument('--h3_refiner_weights', default=None, help='Open-H3-Regenerate-2K source bundle')
    group.add_argument('--h3_refiner_prepared', default=None, help='Prepared Refiner transformer and AdaLN assets')
    group.add_argument('--h3_refiner_seconds', type=int, choices=(5, 10, 15), default=5)
    group.add_argument('--h3_refiner_loading', choices=('prefetch', 'sync'), default='prefetch')
    return parser


def profile(seconds):
    if type(seconds) is not int or seconds not in (5, 10, 15):
        raise ValueError('H3 Refiner duration must be an integer: 5, 10 or 15 seconds')
    output = seconds * 24
    generation = ((output - 5 + 16) // 17) * 17 + 5
    return dict(seconds=seconds, output_frames=output, generation_frames=generation,
                latent_frames=(generation-5)//17*5+2, audio_frames=round(generation/24*40))


def model_key(args):
    """Request shapes are not weight-affecting; changing workflow/assets is."""
    fields = ('model', 'task_type', 'pretrained_model_name_or_path', 'transformer_path',
              'adaln_cache_file', 'num_inference_steps', 'dtype', 'sp', 'text_encoder_tp',
              'matmul_quant', 'vae_matmul_quant', 'atten', 'phaa_num', 'rope_fused',
              'h3_refiner_weights', 'h3_refiner_prepared', 'h3_refiner_loading',
              'h3_attention', 'h3_veda_predictor', 'h3_turbo8', 'h3_nvfp4_checkpoint')
    return tuple(getattr(args, name, None) for name in fields)


def validate_settings(args):
    if not getattr(args, 'h3_refiner', False):
        return None
    if args.model != 'MiniMax-H3' or args.task_type not in ('t2va', 'fl2va', 'ref2va'):
        raise ValueError('H3 Refiner requires MiniMax-H3 t2va/fl2va/ref2va')
    cuda = os.environ.get('XGEN_DEVICE', 'npu') == 'cuda'
    if args.sp != 8 or args.dtype != 'bf16':
        raise ValueError('Experimental H3 Refiner requires SP8 and BF16')
    if cuda and (args.task_type != 'fl2va' or not getattr(args, 'h3_nvfp4_checkpoint', None) or
                 getattr(args, 'h3_nvfp4_resident', False) or getattr(args, 'h3_nvfp4_compile', False)):
        raise ValueError('CUDA H3 Refiner requires a packed FL2VA Base checkpoint with staged residency')
    if cuda and args.atten != 'none':
        raise ValueError('Experimental CUDA H3 Refiner requires --atten none (native attention)')
    if not cuda and args.atten != 'laser':
        raise ValueError('Experimental NPU H3 Refiner requires Laser attention')
    points = (9,) if getattr(args, "h3_turbo8", False) else (21, 51)
    if args.num_inference_steps not in points:
        raise ValueError('Base schedule does not match the selected inference profile')
    if (args.width, args.height) != (2560, 1440) or args.save_fps != 24:
        raise ValueError('Experimental H3 Refiner supports landscape 2560x1440 at 24fps only')
    if not 1 <= args.phaa_num <= 7:
        raise ValueError('H3 Refiner requires phaa_num in [1,7]')
    if cuda and args.matmul_quant != 'none':
        raise ValueError('Experimental CUDA H3 Refiner requires --matmul_quant none')
    if args.matmul_quant not in ('none', 'int8') or args.vae_matmul_quant != 'none':
        raise ValueError('Base supports BF16/INT8; Refiner and VAE require BF16 (vae_matmul_quant=none)')
    if args.vae_lightning != 'none' or int(args.inf_vram_blocks_num) != 0:
        raise ValueError('H3 Refiner owns staged residency: use vae_lightning=none and inf_vram_blocks_num=0')
    if any((getattr(args, 'adopt_sr', False), getattr(args, 'cache_dit', False),
            getattr(args, 'h3_turbo_sla_lora', None), getattr(args, 'lora_path_list', None),
            getattr(args, 'ten_second', False), getattr(args, 'frame_interpolation', False),
            getattr(args, 'frame_interpolation_sr', False), getattr(args, 'fsdp', None),
            getattr(args, 'save_memory', False),
            getattr(args, 'turbo_mode', 'default') != 'default')):
        raise ValueError('H3 Refiner cannot be combined with SeedVR2, cache, Turbo/SLA, extra LoRA or interpolation')
    for name in ('h3_refiner_weights', 'h3_refiner_prepared'):
        value = getattr(args, name, None)
        if not value or not Path(value).is_dir():
            raise ValueError(f'{name} must name an existing local directory')
    return profile(args.h3_refiner_seconds)


def validate_inputs(args):
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if not isinstance(args.prompt, str) or not 0 < len(args.prompt.strip()) <= 32768:
        raise ValueError('prompt must contain 1..32768 characters')
    refs = getattr(args, 'reference_paths', None) or []
    first, last = getattr(args, 'keyframe_image', ''), getattr(args, 'keyframe_last_image', '')
    if not isinstance(refs, (list, tuple)) or any(not isinstance(p, str) for p in refs):
        raise ValueError('reference_paths must be a list of image paths')
    if not isinstance(first, str) or not isinstance(last, str):
        raise ValueError('Keyframes must be image paths')
    if args.task_type == 't2va' and (refs or first or last):
        raise ValueError('t2va does not accept images')
    if args.task_type == 'fl2va' and (not first or last or refs):
        raise ValueError('Initial fl2va support requires one first-frame image only')
    if args.task_type == 'ref2va' and (len(refs) != 5 or first or last):
        raise ValueError('Initial ref2va support requires exactly five reference images')
    paths = refs if args.task_type == 'ref2va' else ([first] if first else [])
    if len(set(paths)) != len(paths):
        raise ValueError('Reference image paths must be distinct')
    from PIL import Image
    for path in paths:
        with Image.open(path) as image:
            if min(image.size) < 1 or image.width * image.height > 4096 * 4096:
                raise ValueError('Reference images must contain 1..4096*4096 pixels')
            image.verify()


@contextmanager
def native_padding(pipe, spec):
    """Permit native temporal padding on this instance, never patch a global class."""
    original = type(pipe)
    if spec['generation_frames'] / pipe.fps > pipe.max_duration:
        padded = type('H3RefinerPaddedWorkflow', (original,), {
            'max_duration': property(lambda self: spec['generation_frames'] / self.fps)})
        pipe.__class__ = padded
    try:
        yield
    finally:
        pipe.__class__ = original
