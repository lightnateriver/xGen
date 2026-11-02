"""Opt-in adapter for existing offline/online managers. No service or request loop."""
from copy import deepcopy
import logging
import time

from .h3_refiner_config import model_key, validate_inputs, validate_settings

logger = logging.getLogger(__name__)


def prepare(manager, args):
    """Load once through the original loader/ability functions; borrow manager.pipe."""
    validate_settings(args)
    if getattr(manager, '_h3_refiner_failed', False):
        raise RuntimeError('H3 Refiner initialization failed; restart the worker before loading again')
    if getattr(manager, '_h3_refiner_key', None) is not None:
        if manager._h3_refiner_key != model_key(args):
            raise ValueError('H3 Refiner model/workflow/settings are fixed at startup; restart for changes')
        if manager.sr_pipe.failed:
            raise RuntimeError('H3 Refiner worker failed; restart before accepting more requests')
        return
    if manager.pipe is not None:
        raise ValueError('Enable H3 Refiner on a fresh manager; restart to change the backend')
    from .load_pipe import load_pipe, update_lora, update_pipe, pipe_to_device
    from ..framework.pipeline.h3_regenerate.pipeline import H3RefinerPipeline
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() != 8:
        raise ValueError('Initialize the existing XGEN distributed environment with eight ranks first')
    started = time.perf_counter()
    try:
        pipe = load_pipe(args)
        update_lora(pipe, args.model, None, args.lora_path_list,
                    fuse_lora=args.fuse_lora, weights_list=args.lora_scale_weight_list)
        update_pipe(pipe, args)
        pipe = pipe_to_device(pipe, args)
        refiner = H3RefinerPipeline(pipe, args)
    except Exception:
        manager._h3_refiner_failed = True
        logger.exception('H3_REFINER_INIT_FAILED retry=false restart_required=true')
        raise
    manager.pipe, manager.sr_pipe = pipe, refiner
    manager.init_args = deepcopy(args)
    manager.current_model = args.model
    manager._h3_refiner_key = model_key(args)
    refiner.startup_seconds = time.perf_counter() - started
    logger.warning('H3_REFINER_READY experimental=true sp=8 init_seconds=%.3f backend=%s',
                   refiner.startup_seconds, args.h3_refiner_loading)


def infer(manager, args):
    """Both existing managers call the same model path, using their original lifecycle."""
    validate_settings(args)
    validate_inputs(args)
    prepare(manager, args)
    return manager.sr_pipe(args)
