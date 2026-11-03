"""CPU-only experimental Base-attention contract. No optional imports on startup."""
import os
from pathlib import Path


def add_arguments(parser):
    group = parser.add_argument_group('Experimental H3 Base attention (disabled by default)')
    group.add_argument('--h3_attention', choices=('legacy', 'spark', 'veda'), default='legacy')
    group.add_argument('--h3_veda_predictor', default=None, help='Published Veda predictor safetensors')
    group.add_argument('--h3_veda_sparsity', type=float, choices=(0.7, 0.9), default=0.9,
                       help='Experimental Veda nominal block sparsity')
    return parser


def validate(args):
    if getattr(args, 'h3_attention', 'legacy') == 'legacy':
        return
    if args.h3_attention not in ('spark', 'veda'):
        raise ValueError('Unknown H3 attention backend')
    if args.h3_attention == 'spark' and os.environ.get('TASK_QUEUE_ENABLE') not in ('0', '1'):
        raise ValueError('Spark NPU graph capture requires TASK_QUEUE_ENABLE=1 (or 0) before process startup')
    if any(type(getattr(args, field)) is not int for field in
           ('sp', 'num_inference_steps', 'frames', 'width', 'height', 'save_fps')):
        raise ValueError('Sparse H3 parallelism, schedule and video dimensions must be integers')
    i2va = args.h3_attention == 'veda' and args.task_type == 'fl2va'
    if args.model != 'MiniMax-H3' or (args.task_type != 't2va' and not i2va):
        raise ValueError('Experimental H3 attention supports T2VA and the Veda I2VA profile')
    if args.h3_attention == 'veda' and getattr(args, 'h3_veda_sparsity', 0.9) not in (0.7, 0.9):
        raise ValueError('Veda sparsity must be 0.7 or 0.9')
    if args.dtype != 'bf16' or args.atten != 'laser' or args.sp != 8:
        raise ValueError('Use BF16, SP8 and atten=laser (legacy components retain Laser)')
    points = 9 if getattr(args, "h3_turbo8", False) else 51
    if i2va:
        if (not getattr(args, 'keyframe_image', None) or getattr(args, 'h3_refiner', False)
                or getattr(args, 'h3_turbo8', False)):
            raise ValueError('Veda I2VA requires one first frame and the original 768P Base model')
        points = 21
    frames = 243 if i2va else 124
    if args.num_inference_steps != points or args.frames != frames:
        raise ValueError('Initial H3 attention validation requires the configured schedule and native frame count')
    expected = (2560, 1440) if getattr(args, 'h3_refiner', False) else (1344, 768)
    if (args.width, args.height) != expected or args.save_fps != 24:
        raise ValueError('Use 5s landscape 768P, or the existing 2K Refiner chain, at 24fps')
    if getattr(args, 'h3_refiner', False) and (type(args.h3_refiner_seconds) is not int or args.h3_refiner_seconds != 5):
        raise ValueError('Initial H3 attention supports five seconds only')
    if (not i2va and getattr(args, 'keyframe_image', None)) or any(
            getattr(args, key, None) for key in ('keyframe_last_image', 'reference_paths',
                                               'h3_turbo_sla_lora', 'lora_path_list', 'cache_dit')):
        raise ValueError('Initial H3 attention does not accept conditioning media, LoRA or DiT cache')
    if args.h3_attention == 'veda' and not Path(args.h3_veda_predictor or '').is_file():
        raise ValueError('Veda requires an existing h3_veda_predictor file')


def key(args):
    return (getattr(args, 'h3_attention', 'legacy'), getattr(args, 'h3_veda_predictor', None),
            getattr(args, 'h3_turbo8', False), getattr(args, 'h3_veda_sparsity', 0.9))
