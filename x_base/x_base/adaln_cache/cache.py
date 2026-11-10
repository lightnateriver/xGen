"""
AdaLN Schedule Cache.

Caches the output of AdaLN projection layers (adaln_proj) keyed by timestep
schedule.  The projection output depends only on t_emb (which depends only on
timestep), not on latent content or prompt.  token_tags / adaln_indices only
affect index_select at forward time, not the projection computation itself.

File cache mode: loads precomputed tables from a safetensors file at init
time.  100% cache hit if sampler config matches.  Cache miss = config
mismatch = error (no fallback).

Enabled via CLI flag:
    --adaln_cache_file     # path to precomputed cache file (safetensors)

Currently targets MiniMax-H3 model structure (transformer_blocks,
time_embedder.linear_1/2, norm_out.linear).  Patterns can be parameterized
to support other AdaLN-based models in the future.
"""

import logging
import fnmatch
import os

import torch
import torch.nn as nn

from x_base.device import device_type

logger = logging.getLogger(__name__)

# Key patterns for AdaLN-related weights stripped in Step 2.
# Used by load_with_allowlist / validate_stripped_checkpoint to verify
# that only adaln keys are missing from a stripped checkpoint.
ADALN_DELETE_PATTERNS = [
    "transformer_blocks.*.adaln_proj.linear.weight",
    "transformer_blocks.*.adaln_proj.linear.bias",
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
    "norm_out.linear.weight",
    "norm_out.linear.bias",
]


class AdalnScheduleCache:
    """Cache AdaLN modulation tables keyed by timestep schedule.

    Usage:
        cache = AdalnScheduleCache(transformer, cache_file="cache.safetensors",
                                   conditioning_type="video")
        cache.maybe_install(timesteps)  # looks up precomputed table
    """

    # Map conditioning_type to constant timestep indices.
    # const_timesteps = [0.999, 1.0]: index 0 = video (keyframe_noise_aug),
    # index 1 = audio.
    _CONST_INDEX_MAP = {
        "none": [],
        "video": [0],
        "audio": [1],
        "both": [0, 1],
    }

    def __init__(self, model, cache_file: str | None = None,
                 conditioning_type: str = "none"):
        self.model = model
        self._tables = {}            # schedule_key -> built tables
        self._current_key = None
        self._current_table = None   # currently installed table dict
        self._file_cache_loaded = False
        self._const_table = None       # constant timestep table (0.999, 1.0)
        self._const_timesteps = None   # tensor of constant timesteps
        self._conditioning_type = conditioning_type
        self._source_file = os.path.realpath(cache_file) if cache_file else None
        self._device = device_type()

        if cache_file:
            self._load_file_cache(cache_file)

    # -- file cache loading ------------------------------------------------

    def _load_file_cache(self, path: str):
        """Load precomputed tables from a safetensors file.

        Populates _tables with schedule_key -> table mapping for all steps.
        After loading, maybe_install() does pure lookup (no computation).
        """
        from safetensors import safe_open

        f = safe_open(path, framework="pt")
        metadata = f.metadata()

        # MiniMaxH3Scheduler: num_steps sigmas (terminal 0 included),
        # but only num_eval_steps timesteps (timesteps = 1 - sigmas[:-1]).
        # For older tables, derive evaluations from the sigma point count.
        num_eval_steps = int(metadata.get("num_eval_steps", int(metadata["num_steps"]) - 1))
        num_blocks = int(metadata.get("num_blocks", len(self.model.transformer_blocks)))
        quantization = metadata.get("quantization", "none")
        # New CUDA-prepared assets include complete timestep batches for each
        # condition. Keep the existing split/merge format unchanged by default.
        conditioned = metadata.get("conditioning_tables")
        if conditioned and self._conditioning_type not in conditioned.split(','):
            raise ValueError(f"AdaLN table does not contain conditioning={self._conditioning_type}")
        prefix = f"{self._conditioning_type}_" if conditioned else ""

        step_entries = []  # list of (timesteps_tensor, table)
        for step_idx in range(num_eval_steps):
            ts_tensor = f.get_tensor(f"{prefix}step{step_idx}_timesteps").to(self._device)

            # Load block tables: chunk raw output into 6 modulation params
            block_tables = []
            for block_idx in range(num_blocks):
                raw = self._read_tensor(f, f"{prefix}step{step_idx}_block{block_idx}", quantization)
                # raw shape: [M, 96768] -> view(-1, 6*H) -> [M*3, 32256] -> chunk(6) -> 6 x [M*3, 5376]
                _H = raw.shape[-1] // (6 * 3)
                modulation = raw.view(-1, 6 * _H).chunk(6, dim=-1)
                block_tables.append(modulation)

            # Load final layer: chunk raw output into (shift, scale)
            final_raw = self._read_tensor(f, f"{prefix}step{step_idx}_final", quantization)
            # final_raw shape: [M, 10752] -> chunk(2, dim=-1) -> 2 x [M, 5376]
            final_shift, final_scale = final_raw.chunk(2, dim=-1)

            table = {
                "blocks": block_tables,
                "final": (final_shift, final_scale),
            }
            step_entries.append((ts_tensor, table))

        # Load constant timesteps table (condition rows: 0.999, 1.0)
        if "const_timesteps" in f.keys():
            const_ts = f.get_tensor("const_timesteps").to(self._device)
            const_block_tables = []
            for block_idx in range(num_blocks):
                raw = self._read_tensor(f, f"const_block{block_idx}", quantization)
                _H = raw.shape[-1] // (6 * 3)
                modulation = raw.view(-1, 6 * _H).chunk(6, dim=-1)
                const_block_tables.append(modulation)
            const_final_raw = self._read_tensor(f, "const_final", quantization)
            const_final_shift, const_final_scale = const_final_raw.chunk(2, dim=-1)
            self._const_table = {
                "blocks": const_block_tables,
                "final": (const_final_shift, const_final_scale),
            }
            self._const_timesteps = const_ts
        elif not conditioned:
            self._const_table = None
            self._const_timesteps = None
            logger.warning(
                "No constant timesteps in cache file. "
                "Condition row (0.999, 1.0) cache miss will error."
            )

        # Proactive pre-merge: merge step tables with selected constant entries
        # based on conditioning_type. After this, _tables keys are full
        # timestep schedules (step + constant), so maybe_install() is pure
        # lookup with zero runtime overhead.
        const_indices = [] if conditioned else self._CONST_INDEX_MAP.get(self._conditioning_type, [])
        if const_indices:
            if self._const_table is None:
                raise RuntimeError(
                    f"conditioning_type={self._conditioning_type} requires "
                    f"constant entries but cache file has none. "
                    f"Regenerate cache with updated precompute script."
                )
            const_sel = torch.tensor(const_indices, dtype=torch.long, device=self._device)
            const_ts_sel = self._const_timesteps[const_indices]
            for step_ts, step_table in step_entries:
                full_ts = torch.cat([step_ts, const_ts_sel])
                full_key = self._schedule_key(full_ts)
                merged = self._merge_table(step_table, const_sel)
                self._tables[full_key] = merged
        else:
            for step_ts, step_table in step_entries:
                key = self._schedule_key(step_ts)
                self._tables[key] = step_table

        self._file_cache_loaded = True
        _const = self._const_timesteps.tolist() if self._const_timesteps is not None else "none"
        logger.info(
            f"AdaLN cache loaded: {num_eval_steps} steps, {num_blocks} blocks, "
            f"{len(self._tables)} tables, conditioning={self._conditioning_type}, "
            f"quant={quantization}, const={_const}, "
            f"variant={metadata.get('model_variant', 'unknown')}"
        )

    def _read_tensor(self, safe_file, key: str, quantization: str) -> torch.Tensor:
        """Read a tensor from safetensors, dequantizing if int8.

        In bf16 mode: reads {key} directly.
        In int8 mode: reads {key}_int8 and {key}_scale, dequantizes to bf16.
        Result is moved to self._device immediately — no CPU intermediate.
        """
        if quantization == "int8":
            int8_data = safe_file.get_tensor(f"{key}_int8")
            scale = safe_file.get_tensor(f"{key}_scale")
            return (int8_data.to(torch.float32) * scale).to(torch.bfloat16).to(self._device)
        else:
            return safe_file.get_tensor(key).to(self._device)

    # -- proactive pre-merge -----------------------------------------------

    def _merge_table(self, step_table: dict, const_sel: torch.Tensor) -> dict:
        """Merge a per-step table with selected constant entries.

        Called during _load_file_cache to proactively build full tables.
        After pre-merge, maybe_install() is pure lookup — zero runtime overhead.

        Args:
            step_table: per-step table with 'blocks' and 'final'.
            const_sel: indices into const table rows to select (e.g. [0]
                for video, [1] for audio, [0,1] for both).

        Returns:
            Merged table with step rows + selected constant rows concatenated.
            Concatenation order: step rows first, then constant rows.
            (step timesteps < 0.999 < 1.0 in all standard configs.)
        """
        if self._const_table is None:
            raise RuntimeError("_merge_table called without const table")
        num_blocks = len(self._const_table["blocks"])
        merged_blocks = []
        for block_idx in range(num_blocks):
            step_mod = step_table["blocks"][block_idx]
            const_mod = self._const_table["blocks"][block_idx]
            # Expand timestep indices to row indices: each timestep has
            # 3 modality rows (video/audio/text) in the block table.
            row_sel = (
                const_sel.unsqueeze(1) * 3
                + torch.arange(3, device=const_sel.device)
            ).reshape(-1)
            selected_const = tuple(
                const_mod[j].index_select(0, row_sel)
                for j in range(6)
            )
            merged = tuple(
                torch.cat([step_mod[j], selected_const[j]], dim=0)
                for j in range(6)
            )
            merged_blocks.append(merged)

        step_final_shift, step_final_scale = step_table["final"]
        const_final_shift, const_final_scale = self._const_table["final"]
        merged_shift = torch.cat([
            step_final_shift,
            const_final_shift.index_select(0, const_sel),
        ], dim=0)
        merged_scale = torch.cat([
            step_final_scale,
            const_final_scale.index_select(0, const_sel),
        ], dim=0)

        return {
            "blocks": merged_blocks,
            "final": (merged_shift, merged_scale),
        }

    # -- key ---------------------------------------------------------------

    @staticmethod
    def _schedule_key(timesteps: torch.Tensor) -> bytes:
        """Float32 bit pattern of timesteps as hashable key."""
        ts_f32 = timesteps.to(torch.float32)
        return ts_f32.cpu().numpy().tobytes()

    # -- install -----------------------------------------------------------

    def maybe_install(self, timesteps: torch.Tensor):
        """Install the precomputed table for the given timesteps.

        Called before each denoise step.  Tables are pre-validated during
        _load_file_cache; a missing key here is a config bug (KeyError).
        Tables are already on the target device (loaded in _load_file_cache).
        """
        key = self._schedule_key(timesteps)
        if key == self._current_key:
            return
        self._current_table = self._tables[key]
        self._current_key = key

    # -- lookup ------------------------------------------------------------

    @property
    def has_current(self):
        """Whether a table is currently installed."""
        return self._current_table is not None

    def lookup_block(self, block_idx: int):
        """Return 6 tensors of [M*3, H] for block forward index_select."""
        return self._current_table["blocks"][block_idx]

    def lookup_final(self):
        """Return (shift, scale), each [M, H], for final layer index_select."""
        return self._current_table["final"]


def install_adaln_cache(pipe, cache_file: str | None = None,
                        conditioning_type: str = "none"):
    """Create AdalnScheduleCache and inject into transformer blocks.

    Called from load_pipe.py after pipeline construction.
    The caller (load_pipe.py) is responsible for gating on the CLI flag.

    Args:
        pipe: Pipeline object with .transformer attribute.
        cache_file: Path to precomputed safetensors cache file.
        conditioning_type: Type of conditioning input. Determines which
                    constant timesteps (0.999, 1.0) are pre-merged with
                    per-step tables. One of: "none", "video", "audio", "both".
    """
    transformer = pipe.transformer
    cache = AdalnScheduleCache(
        transformer, cache_file=cache_file,
        conditioning_type=conditioning_type,
    )

    # Inject cache and block index into each block's adaln_proj
    for i, block in enumerate(transformer.transformer_blocks):
        block.adaln_proj._block_idx = i
        block.adaln_proj._adaln_cache = cache

    # Inject cache into final layer
    transformer.norm_out._adaln_cache = cache

    # Inject cache into transformer itself so _mmh3_forward_with_sp
    # can call maybe_install(timestep) before the block loop.
    transformer._adaln_cache = cache
    if getattr(transformer, '_h3_cuda_adaln_table', False):
        transformer._h3_cuda_adaln_load_count += 1

    return cache


def load_with_allowlist(model, state_dict: dict, allowed_missing_patterns: list[str]):
    """Load state_dict into model with strict=True semantics, but allow
    specific keys to be missing (filled with zeros).

    Used in Step 2 to load a stripped checkpoint where adaln weights have
    been physically removed.  Missing adaln keys are filled with zeros;
    any other missing key or any extra key raises RuntimeError.

    Args:
        model: nn.Module to load into.
        state_dict: checkpoint state dict (may be missing adaln keys).
        allowed_missing_patterns: list of fnmatch patterns for keys that
            are allowed to be missing.

    Raises:
        RuntimeError: if there are unexpected missing keys or extra keys.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys     # model has, checkpoint doesn't
    extra = ckpt_keys - model_keys       # checkpoint has, model doesn't

    # 1. Extra keys -> error (strict=True semantics)
    if extra:
        raise RuntimeError(
            f"load_with_allowlist: unexpected keys in checkpoint (not in model):\n"
            f"{sorted(extra)}\n"
            f"Are you using an unstripped checkpoint with load_with_allowlist?"
        )

    # 2. Missing keys must all be in the allowlist
    unexpected_missing = [
        k for k in sorted(missing)
        if not any(fnmatch.fnmatch(k, p) for p in allowed_missing_patterns)
    ]
    if unexpected_missing:
        raise RuntimeError(
            f"load_with_allowlist: missing keys not in allowlist "
            f"(checkpoint may be corrupted or wrong model):\n"
            f"{unexpected_missing}"
        )

    # 3. Fill missing keys with zeros, then load with strict=True
    for key in missing:
        param = model.state_dict()[key]
        state_dict[key] = torch.zeros_like(param)

    model.load_state_dict(state_dict, strict=True)
    logger.info(
        f"load_with_allowlist: {len(missing)} keys filled with zeros "
        f"(all in allowlist), {len(ckpt_keys)} keys loaded from checkpoint"
    )


def _delete_linear_weights(linear, extra_attrs=()):
    """Delete weight/bias (and optional quant attrs) from a linear module.

    Returns the number of tensors freed.
    """
    freed = 0
    if hasattr(linear, 'weight') and linear.weight is not None:
        del linear.weight
        freed += 1
    if hasattr(linear, 'bias') and linear.bias is not None:
        del linear.bias
        freed += 1
    for attr in extra_attrs:
        if hasattr(linear, attr):
            delattr(linear, attr)
            freed += 1
    return freed


def release_adaln_weights(transformer):
    """Delete adaln-related weight tensors from model to free memory.

    Called after loading from a stripped checkpoint (transformer_adaln_cache).
    The adaln parameters are on meta device (not in checkpoint, not materialized)
    and are never used because the cache provides all modulation params at runtime.

    Safe to call even if some modules don't have the expected attributes.
    """
    freed = 0
    for block in transformer.transformer_blocks:
        adaln = getattr(block, 'adaln_proj', None)
        if adaln is not None and hasattr(adaln, 'linear'):
            freed += _delete_linear_weights(
                adaln.linear, extra_attrs=('weight_scale', 'weight_quantized'))

    time_embedder = getattr(transformer, 'time_embedder', None)
    if time_embedder is not None:
        for name in ('linear_1', 'linear_2'):
            lin = getattr(time_embedder, name, None)
            if lin is not None:
                freed += _delete_linear_weights(lin)

    norm_out = getattr(transformer, 'norm_out', None)
    if norm_out is not None and hasattr(norm_out, 'linear'):
        freed += _delete_linear_weights(norm_out.linear)

    logger.info(f"release_adaln_weights: freed {freed} weight tensors")
    torch.get_device_module(device_type()).empty_cache()


def validate_stripped_checkpoint(transformer, checkpoint_dir: str) -> bool:
    """Validate that a checkpoint directory is correctly stripped of adaln weights.

    Auto-detects the index file name (model.safetensors.index.json or
    diffusion_pytorch_model.safetensors.index.json).  Checks that only
    adaln-related keys are missing from the checkpoint.  Non-adaln missing
    keys or any extra keys raise RuntimeError.

    Args:
        transformer: The transformer model (to get expected state_dict keys).
        checkpoint_dir: Path to the transformer checkpoint directory.

    Returns:
        True if validation passes.

    Raises:
        FileNotFoundError: if no index file found at checkpoint_dir.
        RuntimeError: if non-adaln keys are missing or extra keys are present.
    """
    import json

    # Auto-detect index file name
    index_path = None
    for candidate in [
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ]:
        candidate_path = os.path.join(checkpoint_dir, candidate)
        if os.path.exists(candidate_path):
            index_path = candidate_path
            break
    if index_path is None:
        raise FileNotFoundError(
            f"validate_stripped_checkpoint: no safetensors index file found in "
            f"{checkpoint_dir}. Expected model.safetensors.index.json or "
            f"diffusion_pytorch_model.safetensors.index.json"
        )

    with open(index_path) as f:
        index = json.load(f)

    checkpoint_keys = set(index.get("weight_map", {}).keys())
    model_keys = set(transformer.state_dict().keys())

    missing = model_keys - checkpoint_keys
    extra = checkpoint_keys - model_keys

    if extra:
        raise RuntimeError(
            f"validate_stripped_checkpoint: extra keys in checkpoint (not in model):\n"
            f"{sorted(extra)}\n"
            f"Are you using an unstripped checkpoint?"
        )

    unexpected_missing = [
        k for k in sorted(missing)
        if not any(fnmatch.fnmatch(k, p) for p in ADALN_DELETE_PATTERNS)
    ]
    if unexpected_missing:
        raise RuntimeError(
            f"validate_stripped_checkpoint: missing keys not in adaln allowlist "
            f"(checkpoint may be corrupted or wrong model):\n{unexpected_missing}"
        )

    logger.info(
        f"validate_stripped_checkpoint: PASS -- {len(missing)} adaln keys missing "
        f"(expected), {len(checkpoint_keys)} keys present"
    )
    return True
