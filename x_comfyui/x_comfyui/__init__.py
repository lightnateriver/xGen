"""x_comfyui — NPU acceleration adapter for ComfyUI.

Importing this package triggers:
  1. ``import x_base`` — auto-patches F.rms_norm to torch_npu.npu_rms_norm
  2. Attention mode setup — reads X_ATTENTION_MODE env var and sets the
     global flag in attention_manager (sage/laser/sdpa/rainfusion/sparse)
  3. Model class patches — patches ComfyUI's MiniMax-H3 Attention.forward,
     VAE Attention.forward, BaseModel.__init__, BaseModel.load_model_weights
  4. Text Encoder TP (X_TEXT_TP) for Qwen3-VL,
     applied via CLIP loading hook
  5. Node registration — exposes NODE_CLASS_MAPPINGS for ComfyUI

All patches are applied at import time.  No NPUTurbo node call is needed
for SP/FSDP/quant/VAE/attention — those are controlled by environment variables:

  X_SP_SIZE=8           — sequence parallel size (requires torchrun)
  X_FSDP=true           — enable FSDP (requires torchrun)
  X_QUANT=int8          — weight/activation quantization
  X_ADALN_CACHE_FILE=/path/to/cache.safetensors  — precomputed AdaLN cache file
  X_ADALN_TRANSFORMER_PATH=/path/to/stripped/transformer — adaln-stripped checkpoint (file or dir)
  X_TEXT_TP=2           — Text Encoder TP size for Qwen3-VL (default 0 = disabled)
  X_ROPE_FUSED=true     — enable fused RoPE kernel (default false)

The following can be **overridden at runtime** by the NPUTurbo node
(immediate effect):
  X_ATTENTION_MODE=sageattn  — attention backend (sdpa/sageattn/laserattn/rainfusion/sparse)
  X_PHAA_SPLIT=4        — PHAA overlap split number
  X_ROPE_FUSED          — fused RoPE kernel (enable/disable toggle)

The following can be **overridden at runtime** by the NPUTurbo node
(takes effect on next model load):
  X_QUANT                — weight/activation quantization (enable/disable toggle, fixed int8)
  X_ADALN_CACHE_FILE     — AdaLN cache file (enable/disable toggle + text input)
  X_ADALN_TRANSFORMER_PATH — adaln-stripped checkpoint (text input override)
  X_TEXT_TP              — text encoder TP (enable/disable toggle, size from env var)

When X_ADALN_CACHE_FILE is set, adaln weights are stripped from the state
dict during loading, saving ~26GB of memory.  If X_ADALN_TRANSFORMER_PATH
is also set, a pre-stripped checkpoint is loaded from that path instead of
filtering the original state dict.
conditioning_type is auto-detected from the cache file at load time —
no user-facing parameter is needed.
"""

import x_base  # noqa: F401 — triggers F.rms_norm auto-patch

# --- Set attention mode from env var at import time ---
import logging

logger = logging.getLogger("x_comfyui")

try:
    from .adaptor.env import get_attention_mode
    from x_base import attention_manager

    _mode = get_attention_mode()
    if _mode == "sageattn":
        attention_manager.enable_sage_attention()
    elif _mode == "laserattn":
        attention_manager.enable_laser_attention()
    elif _mode == "rainfusion":
        attention_manager.enable_rainfusion_attention()
    elif _mode == "sparse":
        attention_manager.enable_sparse_attention()
    # "sdpa" = default, no change
    if _mode != "sdpa":
        logger.info(f"x_comfyui: attention mode = {_mode} (from X_ATTENTION_MODE)")
except Exception as e:
    logger.warning(f"x_comfyui: failed to set attention mode — {e}")

# --- Set rope fused from env var at import time ---
try:
    from .adaptor.env import get_rope_fused
    from x_base.operator.rope import rope_manager
    if get_rope_fused():
        rope_manager.enable_rope_fused()
        logger.info("x_comfyui: rope fused = True (from X_ROPE_FUSED)")
except Exception as e:
    logger.warning(f"x_comfyui: failed to set rope fused — {e}")

# --- Patch ComfyUI model classes at import time ---
# These are class-level patches that affect all future model instances.
# If ComfyUI doesn't have MiniMax-H3 support, the import fails silently.

try:
    from .framework.transformer.minimax_h3 import patch_model_classes
    patch_model_classes()
except Exception as e:
    logger.warning(f"x_comfyui: transformer patches skipped — {e}")

try:
    from .framework.vae.minimax_h3 import patch_vae_classes
    patch_vae_classes()
except Exception as e:
    logger.warning(f"x_comfyui: VAE patches skipped — {e}")

# --- CLIP loading hook (TP applied when text encoder loads) ---
try:
    import comfy.sd
    from .framework.text_encoder.minimax_h3 import apply_tp_patch

    _original_clip_init = comfy.sd.CLIP.__init__

    def _patched_clip_init(self, *args, **kwargs):
        """Patched CLIP.__init__ — TP after model loads."""
        _original_clip_init(self, *args, **kwargs)

        text_encoder = getattr(self, "cond_stage_model", None)
        if text_encoder is None:
            return

        # TP — operates on the 50-layer text encoder
        try:
            apply_tp_patch(text_encoder)
        except Exception as exc:
            logger.warning(f"x_comfyui: Text Encoder TP failed — {exc}")

    comfy.sd.CLIP.__init__ = _patched_clip_init
    logger.info("x_comfyui: CLIP loading hook installed (TP)")
except ImportError:
    logger.info("x_comfyui: comfy.sd not available, CLIP loading hook skipped")
except Exception as e:
    logger.warning(f"x_comfyui: CLIP loading hook failed — {e}")

from .adaptor.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
