"""CPU-only experimental Base-attention contract. No optional imports on startup."""
import os
from pathlib import Path


def add_arguments(parser):
    group = parser.add_argument_group('Experimental H3 Base attention (disabled by default)')
    group.add_argument('--h3_attention', choices=('legacy', 'spark', 'veda'), default='legacy')
    group.add_argument('--h3_veda_predictor', default=None, help='Published Veda predictor safetensors')
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
    if args.model != 'MiniMax-H3' or args.task_type != 't2va':
        raise ValueError('Experimental H3 attention initially supports T2VA only')
    if args.dtype != 'bf16' or args.atten != 'laser' or args.sp != 8:
        raise ValueError('Use BF16, SP8 and atten=laser (legacy components retain Laser)')
    if args.num_inference_steps != 51 or args.frames != 124:
        raise ValueError('Initial H3 attention validation requires 50 actual DiT steps / 124 native frames')
    expected = (2560, 1440) if getattr(args, 'h3_refiner', False) else (1344, 768)
    if (args.width, args.height) != expected or args.save_fps != 24:
        raise ValueError('Use 5s landscape 768P, or the existing 2K Refiner chain, at 24fps')
    if getattr(args, 'h3_refiner', False) and (type(args.h3_refiner_seconds) is not int or args.h3_refiner_seconds != 5):
        raise ValueError('Initial H3 attention supports five seconds only')
    if any(getattr(args, key, None) for key in ('keyframe_image', 'keyframe_last_image', 'reference_paths',
                                               'h3_turbo_sla_lora', 'lora_path_list', 'cache_dit')):
        raise ValueError('Initial H3 attention does not accept conditioning media, LoRA or DiT cache')
    if args.h3_attention == 'veda' and not Path(args.h3_veda_predictor or '').is_file():
        raise ValueError('Veda requires an existing h3_veda_predictor file')


def key(args):
    return getattr(args, 'h3_attention', 'legacy'), getattr(args, 'h3_veda_predictor', None)
