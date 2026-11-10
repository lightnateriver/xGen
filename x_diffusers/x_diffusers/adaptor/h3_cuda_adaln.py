"""Optional BF16 schedule tables on the existing CUDA H3 SP recipes."""
import json
import fnmatch
from pathlib import Path


def resolve_assets(args, backend, profile):
    """Resolve workflow-scoped paths without I/O or creating model assets."""
    mode = getattr(args, 'h3_cuda_adaln_mode', None) or 'auto'
    if mode not in ('auto', 'table', 'owner'):
        raise ValueError('Unknown CUDA AdaLN mode')
    root = getattr(args, 'h3_cuda_adaln_dir', None)
    if backend != 'cuda' or profile == 'none':
        if mode != 'auto' or root:
            raise ValueError('CUDA AdaLN modes require a CUDA H3 FP8 profile')
        return
    owned = getattr(args, '_h3_cuda_adaln_auto_paths', False)
    checkpoint = getattr(args, 'transformer_path', None)
    table = getattr(args, 'adaln_cache_file', None)
    if mode == 'owner':
        if owned:
            args.transformer_path = args.adaln_cache_file = None
        elif checkpoint or table or root:
            raise ValueError('CUDA AdaLN owner mode cannot combine with table assets')
        return
    active = mode == 'table' or bool(checkpoint or table or root) or (
        getattr(args, 'h3_cuda_preset', None) == 'sage-fp8-full-parallel' and profile == 'fp8-sage')
    if not active:
        return
    if getattr(args, 'h3_parallel_mode', 'sp') != 'sp':
        raise ValueError('CUDA AdaLN tables require native SP; use --h3_cuda_adaln_mode owner for TP/CP')
    args.h3_cuda_adaln_mode = 'table'
    if (checkpoint or table) and not owned:
        if root:
            raise ValueError('Choose h3_cuda_adaln_dir or explicit transformer_path/adaln_cache_file, not both')
        return
    variant = 'ref2va' if getattr(args, 'task_type', None) == 'ref2va' else 'fl2va'
    directory = Path(root) if root else Path(getattr(args, 'pretrained_model_name_or_path', 'weights/h3'))/'xgen-adaln'/variant
    steps = getattr(args, 'num_inference_steps', 50)
    if type(steps) is not int or steps < 1:
        raise ValueError('CUDA AdaLN table selection requires positive actual DiT steps')
    args.h3_cuda_adaln_dir = str(directory)
    args.transformer_path = str(directory/'transformer')
    args.adaln_cache_file = str(directory/'tables'/f'adaln_{variant}_bf16_{steps}step.safetensors')
    args._h3_cuda_adaln_auto_paths = True


def validate_options(args):
    checkpoint = getattr(args, 'transformer_path', None)
    table = getattr(args, 'adaln_cache_file', None)
    if not checkpoint and not table:
        return False
    if not checkpoint or not table:
        raise ValueError('CUDA FP8 AdaLN tables require both transformer_path and adaln_cache_file')
    if not Path(table).is_file() or not (Path(checkpoint)/'.xgen_adaln_source.json').is_file():
        raise ValueError('Missing prepared CUDA AdaLN assets: generate matched tables and a stripped checkpoint '
                         'with tools/adaln_cache/prepare_cuda_adaln.py (see docs/H3-CUDA-ADALN.md), '
                         'or explicitly select --h3_cuda_adaln_mode owner')
    task = getattr(args, 'task_type', None)
    variant = {'t2va': 'FL2VA', 'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}.get(task)
    if variant is None or getattr(args, 'h3_parallel_mode', 'sp') != 'sp':
        raise ValueError('CUDA FP8 AdaLN tables require H3 T2VA/FL2VA/Ref2VA native SP')
    from safetensors import safe_open
    from .h3_schedule import schedule_points

    schedule_points(args.num_inference_steps, table)
    with safe_open(table, framework='pt') as cache:
        metadata = cache.metadata() or {}
    manifest = json.loads((Path(checkpoint) / '.xgen_adaln_source.json').read_text())
    from x_base.adaln_cache import ADALN_DELETE_PATTERNS
    index = json.loads((Path(checkpoint) / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    if (len(manifest.get('removed_keys', [])) != 106
            or any(any(fnmatch.fnmatch(key, pattern) for pattern in ADALN_DELETE_PATTERNS) for key in index['weight_map'])):
        raise ValueError('CUDA AdaLN table mode requires a checkpoint with AdaLN weights physically removed')
    if (metadata.get('model_variant') != variant or manifest.get('model_variant') != variant
            or metadata.get('quantization', 'none') != 'none'
            or int(metadata.get('num_blocks', 0)) != 50
            or not manifest.get('adaln_weight_sha256')
            or metadata.get('adaln_weight_sha256') != manifest['adaln_weight_sha256']
            or any(float(metadata[name]) != float(manifest[name]) for name in ('video_shift', 'audio_shift'))):
        raise ValueError('AdaLN table must match the workflow, stripped checkpoint and scheduler')
    return True


def assert_weights_absent(model):
    from x_base.adaln_cache import ADALN_DELETE_PATTERNS
    if any(any(fnmatch.fnmatch(name, pattern) for pattern in ADALN_DELETE_PATTERNS)
           for name, _ in model.named_parameters()):
        raise ValueError('AdaLN projection/time-embedding weights remain in table-only mode')


def select_table(pipe, args, conditioning):
    """Switch only the table; retain the same stripped model and CUDA operators."""
    from x_base.adaln_cache import install_adaln_cache
    from safetensors import safe_open

    validate_options(args)
    model = pipe.transformer
    if str(Path(args.transformer_path).resolve()) != model._h3_cuda_adaln_checkpoint:
        raise ValueError('Restart the manager to change the stripped AdaLN checkpoint')
    with safe_open(args.adaln_cache_file, framework='pt') as source:
        metadata = source.metadata()
    if (float(pipe.scheduler.config.shift) != float(metadata['video_shift'])
            or float(pipe.audio_scheduler.config.shift) != float(metadata['audio_shift'])):
        raise ValueError('AdaLN table scheduler shifts do not match the loaded pipeline')
    cache = model._adaln_cache
    if (cache._source_file != str(Path(args.adaln_cache_file).resolve())
            or cache._conditioning_type != conditioning):
        install_adaln_cache(pipe, args.adaln_cache_file, conditioning_type=conditioning)
