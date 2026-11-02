"""NPU environment configuration for x_comfyui.

Environment variables (set before launching ComfyUI or main_multi.py):

  X_SP_SIZE=8       — sequence parallel size (requires torchrun)
  X_FSDP=true       — enable FSDP (requires torchrun)
  X_PHAA_SPLIT=4    — PHAA overlap split number (requires torchrun)
  X_QUANT=int8      — weight/activation quantization
  X_ADALN_CACHE_FILE=/path/to/cache.safetensors  — precomputed AdaLN cache file
  X_ADALN_TRANSFORMER_PATH=/path/to/stripped/transformer  — adaln-stripped checkpoint (file or dir)
  X_ATTENTION_MODE=sageattn  — attention backend (sdpa/sageattn/laserattn/rainfusion/sparse)
  X_VAE_COPY_STREAM=1  — async D2H copy stream for VAE streaming decode (default 0)
  X_TEXT_TP=2  — Text Encoder TP size for Qwen3-VL (default 0 = disabled)
  X_ROPE_FUSED=true  — enable fused RoPE kernel (default false)

Init-time only (affect process group / model sharding, must be env vars):
  X_SP_SIZE, X_FSDP

Runtime-overridable by NPUTurbo (immediate effect):
  X_ATTENTION_MODE, X_PHAA_SPLIT, X_ROPE_FUSED

Runtime-overridable by NPUTurbo (takes effect on next model load):
  X_QUANT, X_ADALN_CACHE_FILE, X_ADALN_TRANSFORMER_PATH,
  X_TEXT_TP

The NPUTurbo node in ComfyUI is an optional runtime override — if placed
in the workflow, it overrides the env var defaults.  For attention mode,
PHAA the override takes effect immediately.  For quant
and AdaLN cache the override takes effect when the model is (re)loaded
(because these modify model weights/structure).

NPUTurbo exposes quant as a simple enable/disable toggle (fixed int8).
For adaln, it provides an enable/disable toggle plus text inputs for
adaln_cache_file and adaln_transformer_path — when non-empty, they
override the corresponding env vars at runtime (takes effect on next
model load).  When empty, they fall back to the env vars.

When X_ADALN_CACHE_FILE is set, adaln-related weights are stripped from
the state dict during model loading, saving ~26GB of memory.  If
X_ADALN_TRANSFORMER_PATH is also set, a pre-stripped checkpoint is loaded
from that path instead of filtering the original state dict.

conditioning_type is auto-detected from the cache file at load time:
if the cache contains constant timestep entries (0.999, 1.0), all are
pre-merged (equivalent to conditioning_type="both"); otherwise none are
pre-merged ("none").  No user-facing parameter is needed.
"""

import os

# ---------------------------------------------------------------------------
# Override globals — set by NPUTurbo node
# None  = use env var (default)
# True  = force enable
# False = force disable
# str   = use this value (for path-type overrides)
# ---------------------------------------------------------------------------
_QUANT_OVERRIDE = None                       # None / True / False
_ADALN_OVERRIDE = None                       # None / True / False
_ADALN_CACHE_FILE_OVERRIDE = None             # None / str
_ADALN_TRANSFORMER_PATH_OVERRIDE = None      # None / str
_TEXT_TP_OVERRIDE = None                     # None / True / False
_ROPE_FUSED_OVERRIDE = None                  # None / True / False

# Quant value from the previous prompt execution (for cache invalidation).
# Compared at the start of execute_async to detect quant changes.
_last_quant = None                           # None / "enable" / "disable"

# Text Encoder TP value from the previous prompt execution.
# Used by _patched_execute_async to detect changes between runs (same as _last_quant).
_last_text_tp = None                         # None / "enable" / "disable"


def set_quant_override(enable: bool):
    """Set quant override from NPUTurbo.

    Args:
        enable: True to force-enable quant (reads X_QUANT for mode),
                False to force-disable quant,
                None to use env var.
    """
    global _QUANT_OVERRIDE
    _QUANT_OVERRIDE = enable


def set_text_tp_override(enable: bool):
    """Set text encoder TP override from NPUTurbo.

    When enabled, reads X_TEXT_TP env var for the actual TP size.
    When disabled, forces TP size to 0 (disabled).

    Args:
        enable: True to use X_TEXT_TP env var value,
                False to force-disable TP,
                None to use env var directly.
    """
    global _TEXT_TP_OVERRIDE
    _TEXT_TP_OVERRIDE = enable


def set_rope_fused_override(enable: bool):
    """Set rope fused override from NPUTurbo.

    Args:
        enable: True to force-enable fused RoPE,
                False to force-disable fused RoPE,
                None to use env var.
    """
    global _ROPE_FUSED_OVERRIDE
    _ROPE_FUSED_OVERRIDE = enable


def set_adaln_override(enable: bool):
    """Set AdaLN cache override from NPUTurbo.

    Args:
        enable: True to force-enable AdaLN cache (reads X_ADALN_CACHE_FILE
                for path and X_ADALN_TRANSFORMER_PATH for stripped weights),
                False to force-disable AdaLN cache,
                None to use env var.
    """
    global _ADALN_OVERRIDE
    _ADALN_OVERRIDE = enable


def set_adaln_cache_file_override(path: str | None):
    """Set adaln cache file override from NPUTurbo.

    Args:
        path: cache file path string, or None/empty to fall back to
              X_ADALN_CACHE_FILE env var.
    """
    global _ADALN_CACHE_FILE_OVERRIDE
    _ADALN_CACHE_FILE_OVERRIDE = path if path else None


def set_adaln_transformer_path_override(path: str | None):
    """Set adaln transformer path override from NPUTurbo.

    Args:
        path: transformer path string (file or directory), or None/empty
              to fall back to X_ADALN_TRANSFORMER_PATH env var.
    """
    global _ADALN_TRANSFORMER_PATH_OVERRIDE
    _ADALN_TRANSFORMER_PATH_OVERRIDE = path if path else None


def get_last_quant():
    """Return the quant value from the previous prompt execution.

    Used by the execute_async patch to detect quant changes between runs.
    Returns None on the first run.
    """
    return _last_quant


def set_last_quant(value):
    """Store the quant value from the current prompt execution."""
    global _last_quant
    _last_quant = value


def get_last_text_tp():
    """Return the text_tp value from the previous prompt execution.

    Used by the execute_async patch to detect text_tp changes between runs.
    Returns None on the first run.
    """
    return _last_text_tp


def set_last_text_tp(value):
    """Store the text_tp value from the current prompt execution."""
    global _last_text_tp
    _last_text_tp = value


# ---------------------------------------------------------------------------
# Env var readers (check override first, fall back to env var)
# ---------------------------------------------------------------------------

def get_sp_size():
    """Sequence parallel size from X_SP_SIZE env var (default 1)."""
    return int(os.environ.get("X_SP_SIZE", "1"))


def get_fsdp_enabled():
    """Whether FSDP is enabled via X_FSDP env var."""
    return os.environ.get("X_FSDP", "").lower() == "true"


def get_phaa_split():
    """PHAA split number from X_PHAA_SPLIT env var (default 1)."""
    return int(os.environ.get("X_PHAA_SPLIT", "1"))


def get_quant():
    """Quantization mode (default '').

    Checks _QUANT_OVERRIDE first:
      - True  → returns 'int8' (force-enabled by NPUTurbo)
      - False → returns '' (force-disabled by NPUTurbo)
      - None  → falls back to X_QUANT env var

    Returns 'int8' if enabled, '' otherwise.
    """
    if _QUANT_OVERRIDE is not None:
        return "int8" if _QUANT_OVERRIDE else ""
    return os.environ.get("X_QUANT", "").lower()


def get_text_tp():
    """Text encoder TP size (default 0 = disabled).

    Checks _TEXT_TP_OVERRIDE first:
      - True  → returns X_TEXT_TP env var value; if X_TEXT_TP is not set,
               falls back to X_SP_SIZE (so NPUTurbo's text_tp=enable works
               without requiring a separate env var)
      - False → returns 0 (force-disabled by NPUTurbo)
      - None  → falls back to X_TEXT_TP env var

    Returns the TP size as int.
    """
    if _TEXT_TP_OVERRIDE is not None:
        if _TEXT_TP_OVERRIDE:
            tp_str = os.environ.get("X_TEXT_TP", "")
            if tp_str:
                return int(tp_str)
            # X_TEXT_TP not set — fall back to X_SP_SIZE as default TP size
            sp_size = int(os.environ.get("X_SP_SIZE", "1"))
            if sp_size > 1:
                return sp_size
            return 0
        return 0
    return int(os.environ.get("X_TEXT_TP", "0"))


def get_rope_fused():
    """Whether fused RoPE is enabled.

    Checks _ROPE_FUSED_OVERRIDE first:
      - True  → force-enabled by NPUTurbo
      - False → force-disabled by NPUTurbo
      - None  → falls back to X_ROPE_FUSED env var

    Returns True if enabled, False otherwise.
    """
    if _ROPE_FUSED_OVERRIDE is not None:
        return _ROPE_FUSED_OVERRIDE
    return os.environ.get("X_ROPE_FUSED", "").lower() == "true"


def get_adaln_cache_file():
    """Path to precomputed AdaLN cache file.

    Priority:
      1. _ADALN_OVERRIDE is False → None (force-disabled by NPUTurbo toggle)
      2. _ADALN_CACHE_FILE_OVERRIDE is not None → return it (from NPUTurbo text input)
      3. X_ADALN_CACHE_FILE env var

    Returns None if not set or disabled.
    """
    if _ADALN_OVERRIDE is False:
        return None  # force-disabled by NPUTurbo toggle
    if _ADALN_CACHE_FILE_OVERRIDE is not None:
        return _ADALN_CACHE_FILE_OVERRIDE  # path from NPUTurbo text input
    # True or None: use env var
    return os.environ.get("X_ADALN_CACHE_FILE", None)


def get_adaln_transformer_path():
    """Path to adaln-stripped transformer checkpoint (file or directory).

    Checks _ADALN_TRANSFORMER_PATH_OVERRIDE first:
      - str  → returns the override path (from NPUTurbo)
      - None → falls back to X_ADALN_TRANSFORMER_PATH env var

    When set, the stripped checkpoint is loaded at weight-loading time
    and used instead of the original state dict.  The stripped checkpoint
    must be in the same format as the original ComfyUI checkpoint, with
    adaln-related keys removed (adaln_proj, time_embedder,
    final_layer/norm_out).

    This is the ComfyUI equivalent of x_diffusers' --transformer_path.
    The precompute script's --strip_output produces a suitable checkpoint.

    Returns None if not set.
    """
    if _ADALN_TRANSFORMER_PATH_OVERRIDE is not None:
        return _ADALN_TRANSFORMER_PATH_OVERRIDE
    return os.environ.get("X_ADALN_TRANSFORMER_PATH", None)


def get_attention_mode():
    """Attention backend from X_ATTENTION_MODE env var (default 'sdpa').

    Returns one of: 'sdpa', 'sageattn', 'laserattn', 'rainfusion', 'sparse'.
    """
    return os.environ.get("X_ATTENTION_MODE", "sdpa").lower()


def get_vae_copy_stream():
    """Whether async D2H copy stream for VAE streaming decode is enabled.

    Reads X_VAE_COPY_STREAM env var (default "0" = disabled).
    """
    return os.environ.get("X_VAE_COPY_STREAM", "0") == "1"
