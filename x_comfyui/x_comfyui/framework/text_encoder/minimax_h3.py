"""NPU adaptation for the MiniMax-H3 text encoder in ComfyUI.

This feature only supports MiniMax-H3. The patch is applied at **import time**
— when ComfyUI loads x_comfyui as a custom node. The patch is controlled
by environment variables:

  X_TEXT_TP=8 — Text Encoder TP size (default 0 = disabled)

Patch applied:
  1. **Tensor Parallelism** — replaces nn.Linear with ColumnParallelLinear/
     RowParallelLinear for all 50 layers

The MiniMax-H3 checkpoint is pre-truncated to 50 layers by ComfyUI
(Qwen3VL_32BConfig: num_hidden_layers=50, final_norm=False, lm_head=False).
No runtime truncation is needed.

The NPUTurbo node provides runtime overrides for X_TEXT_TP.
See ``nodes.py`` for details.

Environment variables:
  X_TEXT_TP — int, TP size (default 0 = disabled)
"""

import logging

import torch
from x_base import ParallelManager, enable_tp

from x_comfyui.adaptor.env import get_text_tp

logger = logging.getLogger("x_comfyui")

# --- Constants ---

# The H3 checkpoint has 50 layers (pre-truncated by ComfyUI config).
_TP_TARGET_LAYERS = 50

# Cached TP parallel manager (separate from SP manager)
_tp_parallel_manager = None


# --- Env var readers ---

def _get_text_tp_size():
    return get_text_tp()


# --- Patch functions ---

def apply_tp_patch(text_encoder):
    """Apply TP to a loaded text encoder model.

    Called from the CLIP loading hook in __init__.py.
    """
    tp_size = _get_text_tp_size()
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
