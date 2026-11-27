"""MiniMax-H3 text encoder TP mechanism for ComfyUI.

The application CLIP hook resolves environment/node settings and passes
``tp_size`` explicitly. The checkpoint is already truncated to 50 layers.
"""

import logging

import torch
from x_base import ParallelManager, enable_tp


logger = logging.getLogger("x_comfyui")

# --- Constants ---

# The H3 checkpoint has 50 layers (pre-truncated by ComfyUI config).
_TP_TARGET_LAYERS = 50

# Cached TP parallel manager (separate from SP manager)
_tp_parallel_manager = None


# --- Patch functions ---

def apply_tp_patch(text_encoder, *, tp_size=0):
    """Apply TP to a loaded text encoder model.

    Called from the CLIP loading hook with a resolved TP size.
    """
    if tp_size <= 1:
        logger.info(f"x_comfyui: Text Encoder TP disabled (tp_size={tp_size})")
        return text_encoder

    if not torch.distributed.is_initialized():
        logger.warning("x_comfyui: Text Encoder TP requires torch.distributed to be initialized")
        return text_encoder

    world_size = torch.distributed.get_world_size()
    dp_size = world_size // tp_size
    if dp_size < 1:
        logger.warning(
            f"x_comfyui: Text Encoder TP size {tp_size} > world_size {world_size}, skipping"
        )
        return text_encoder

    global _tp_parallel_manager
    _tp_parallel_manager = ParallelManager(
        dp_size=dp_size, cp_size=1, sp_size=1, tp_size=tp_size,
    )
    logger.info(f"x_comfyui: Text Encoder TP ParallelManager created (dp={dp_size}, tp={tp_size})")

    enable_tp(text_encoder, tp_size, target_layers=range(_TP_TARGET_LAYERS))
    logger.info(
        f"x_comfyui: enable_tp applied to MiniMax-H3 text encoder "
        f"(tp_size={tp_size}, target_layers=range({_TP_TARGET_LAYERS}))"
    )
    return text_encoder
