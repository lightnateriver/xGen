#!/usr/bin/env python3
"""Precompute AdaLN modulation parameters for MiniMax-H3.

Generates a safetensors cache file containing precomputed AdaLN outputs
for all denoising steps. At runtime, the cache is loaded and used to
skip adaln_proj computation.

Optionally strips adaln weights from the checkpoint (--strip_output),
producing a smaller model that must be used with the cache file.

The cache is keyed by timestep schedule (float32 bit pattern of unique
timesteps per step). Since the timestep schedule is deterministic for a
given sampler config, the cache achieves 100% hit rate when the config
matches.

Usage:
    # Cache + strip weights (recommended: one command, weights removed)
    python precompute_adaln_cache.py \
        --model_path ./weights/base/transformer \
        --steps 50 --video_shift 12.0 --audio_shift 3.0 \
        --output ./weights/base/adaln_cache/adaln_cache_50step.safetensors \
        --strip_output ./weights/base/transformer_adaln_cache

    # For Ref2VA:
    python precompute_adaln_cache.py \
        --model_path ./weights/base/transformer_ref \
        --steps 50 --video_shift 12.0 --audio_shift 3.0 \
        --output adaln_cache_ref2va_50step.safetensors \
        --model_variant Ref2VA
"""

import argparse
import json
import os
import shutil
import fnmatch
import logging

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# AdaLN-related keys to strip from checkpoint (diffusers naming, no remapping needed).
DELETE_PATTERNS = [
    "transformer_blocks.*.adaln_proj.linear.weight",
    "transformer_blocks.*.adaln_proj.linear.bias",
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
    "norm_out.linear.weight",
    "norm_out.linear.bias",
]


def load_model(transformer_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load MiniMax-H3 transformer model via ModularPipeline.

    Uses ModularPipeline.from_pretrained() + load_components() which
    handles sharded checkpoint loading and key remapping.

    The pipeline path is derived as the parent of transformer_path.
    It must contain modular_model_index.json (top-level repo, not
    the FL2VA/ or Ref2VA/ subfolder).

    Only the transformer is kept; the rest of the pipeline (VAE, text
    encoder, etc.) is deleted to free memory.
    """
    import gc
    from diffusers import ModularPipeline

    pipeline_path = os.path.dirname(transformer_path)
    pipe = ModularPipeline.from_pretrained(pipeline_path)
    pipe.load_components(dtype=dtype, pretrained_model_name_or_path=pipeline_path)

    model = pipe.transformer
    model.eval()

    del pipe
    gc.collect()

    logger.info(f"Loaded transformer from {transformer_path}")
    return model


def create_scheduler(shift: float, num_steps: int):
    """Create MiniMax-H3 scheduler with given shift and steps."""
    from diffusers import MiniMaxH3Scheduler
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(num_inference_steps=num_steps)
    return scheduler


def _matches_delete(key: str) -> bool:
    """Check if a state_dict key matches any DELETE_PATTERN."""
    for pattern in DELETE_PATTERNS:
        if fnmatch.fnmatch(key, pattern):
            return True
    return False


def strip_checkpoint(src_dir: str, dst_dir: str):
    """Strip adaln weights from checkpoint and write to dst_dir.

    Reads each safetensors shard, filters out adaln-related keys,
    writes stripped shards + new index.json + copies config files.

    Args:
        src_dir: Source checkpoint directory (model-*.safetensors + index.json)
        dst_dir: Output directory for stripped checkpoint
    """
    os.makedirs(dst_dir, exist_ok=True)

    # Auto-detect index file name (top-level uses diffusion_pytorch_model.*,
    # FL2VA subfolder uses model.*)
    index_path = None
    for candidate in [
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ]:
        candidate_path = os.path.join(src_dir, candidate)
        if os.path.exists(candidate_path):
            index_path = candidate_path
            break
    if index_path is None:
        raise FileNotFoundError(
            f"No safetensors index file found in {src_dir}. "
            f"Expected model.safetensors.index.json or "
            f"diffusion_pytorch_model.safetensors.index.json"
        )
    index_name = os.path.basename(index_path)
    logger.info(f"Using index file: {index_name}")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    old_total_size = index.get("metadata", {}).get("total_size", 0)

    # Partition keys by shard
    shard_keys = {}  # shard_file → {keep: [...], delete: [...]}
    for key, shard_file in weight_map.items():
        if shard_file not in shard_keys:
            shard_keys[shard_file] = {"keep": [], "delete": []}
        if _matches_delete(key):
            shard_keys[shard_file]["delete"].append(key)
        else:
            shard_keys[shard_file]["keep"].append(key)

    total_delete = sum(len(v["delete"]) for v in shard_keys.values())
    total_keep = sum(len(v["keep"]) for v in shard_keys.values())
    logger.info(f"Strip: {total_delete} keys to delete, {total_keep} to keep, {len(shard_keys)} shards")

    # Process each shard
    new_weight_map = {}
    new_total_size = 0

    for shard_file, keys in shard_keys.items():
        shard_path = os.path.join(src_dir, shard_file)
        logger.info(f"  {shard_file}: keep={len(keys['keep'])}, delete={len(keys['delete'])}")

        state_dict = load_file(shard_path)
        new_state_dict = {}
        for key in keys["keep"]:
            if key in state_dict:
                new_state_dict[key] = state_dict[key]
                new_weight_map[key] = shard_file
                new_total_size += state_dict[key].numel() * state_dict[key].element_size()

        del state_dict
        save_file(new_state_dict, os.path.join(dst_dir, shard_file))
        del new_state_dict

    # Write new index.json (preserve original index file name)
    with open(os.path.join(dst_dir, index_name), "w") as f:
        json.dump({"metadata": {"total_size": new_total_size}, "weight_map": new_weight_map}, f, indent=2)

    # Copy non-weight files (config.json, etc.)
    for fname in os.listdir(src_dir):
        if fname.endswith(".safetensors") or fname == index_name:
            continue
        src_path = os.path.join(src_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, fname))

    old_mb = old_total_size / (1024 * 1024)
    new_mb = new_total_size / (1024 * 1024)
    logger.info(
        f"Strip done: {old_mb:.0f} MB → {new_mb:.0f} MB (saved {old_mb - new_mb:.0f} MB, {(1 - new_mb / old_mb) * 100:.1f}%)")
    logger.info(f"Output: {dst_dir}")


def _compute_silu_temb(model, timesteps, device):
    """Compute SiLU(time_embedder(time_proj(timesteps)))."""
    t_emb = model.time_proj(timesteps.to(device))
    temb_dtype = model.time_embedder.linear_1.weight.dtype
    t_emb = t_emb.to(temb_dtype)
    temb = model.time_embedder(t_emb)
    return F.silu(temb)


def _linear_forward(linear, silu_temb):
    """Forward linear layer."""
    return linear(silu_temb.to(linear.weight.dtype))


def _store_output(tensors_to_save, key, raw_output):
    """Store output tensor."""
    tensors_to_save[key] = raw_output.cpu().contiguous()


def _precompute_adaln_for_timesteps(model, timesteps, device, prefix, tensors_to_save):
    """Precompute all AdaLN outputs for a set of timesteps.

    Computes time embedding, iterates transformer blocks, and final layer.
    Stores results in tensors_to_save with keys like f"{prefix}_block{j}" etc.
    """
    silu_temb = _compute_silu_temb(model, timesteps, device)

    for block_idx, block in enumerate(model.transformer_blocks):
        block.to(device)
        raw_output = _linear_forward(block.adaln_proj.linear, silu_temb)
        block.cpu()
        _store_output(tensors_to_save, f"{prefix}_block{block_idx}", raw_output)

    final_raw = _linear_forward(model.norm_out.linear, silu_temb)
    _store_output(tensors_to_save, f"{prefix}_final", final_raw)


def precompute_cache(
        model_path: str,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        output_path: str,
        model_variant: str = "FL2VA",
):
    """Precompute all AdaLN modulation parameters and save to safetensors.

    Single card + group offload. Model blocks stay on CPU, each block
    (~1.2GB) is moved to device one at a time for computation.

    MiniMaxH3Scheduler produces num_steps sigmas (terminal 0 included)
    but only num_steps-1 timesteps (timesteps = 1 - sigmas[:-1]).
    The shift's unique_consecutive may further reduce the count.
    The actual eval step count is min(len(video_ts), len(audio_ts)).

    Cache file format:
      Metadata:
        num_steps, num_eval_steps, video_shift, audio_shift,
        model_variant, num_blocks, has_const_entries
      Tensors per step (idx = 0..num_eval_steps-1):
        step{idx}_timesteps  → [M] float32 (unique step timesteps: video+audio)
        step{idx}_block{j}   → [M, 96768] bf16 (adaln_proj raw output, pre-chunk)
        step{idx}_final      → [M, 10752] bf16 (norm_out.linear raw output, pre-chunk)
      Constant entries (condition rows, stored once, not per-step):
        const_timesteps  → [2] float32 = [0.999, 1.0]
        const_block{j}   → [2, 96768] bf16
        const_final      → [2, 10752] bf16
    """
    # --- Determine device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- Load model ---
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model.eval()

    # Move non-block modules to device (group offload only manages transformer_blocks)
    model.time_embedder.to(device)
    model.norm_out.to(device)

    # --- Create dual schedulers ---
    video_scheduler = create_scheduler(video_shift, num_steps)
    audio_scheduler = create_scheduler(audio_shift, num_steps)
    num_eval_steps = min(len(video_scheduler.timesteps), len(audio_scheduler.timesteps))
    logger.info(f"Video scheduler: shift={video_shift}, {len(video_scheduler.timesteps)} timesteps")
    logger.info(f"Audio scheduler: shift={audio_shift}, {len(audio_scheduler.timesteps)} timesteps")
    logger.info(f"Eval steps: {num_eval_steps}")

    # --- Collect unique timesteps per step ---
    step_timesteps = []
    for step_idx in range(num_eval_steps):
        t_video = video_scheduler.timesteps[step_idx]
        t_audio = audio_scheduler.timesteps[step_idx]
        unique_ts = torch.unique(torch.stack([t_video, t_audio]))
        step_timesteps.append(unique_ts)
        logger.info(
            f"Step {step_idx}: video_t={t_video.item():.6f}, "
            f"audio_t={t_audio.item():.6f}, "
            f"unique_count={len(unique_ts)}"
        )

    # --- Precompute tables ---
    tensors_to_save = {}
    num_blocks = len(model.transformer_blocks)

    with torch.no_grad():
        for step_idx, unique_ts in enumerate(step_timesteps):
            logger.info(f"Precomputing step {step_idx}/{num_eval_steps} (M={len(unique_ts)})...")
            tensors_to_save[f"step{step_idx}_timesteps"] = unique_ts.cpu().to(torch.float32).contiguous()
            _precompute_adaln_for_timesteps(
                model, unique_ts, device, f"step{step_idx}", tensors_to_save
            )

        # --- Precompute constant timesteps (condition rows: 0.999, 1.0) ---
        CONST_NOISE_AUG = 0.999  # MINIMAX_H3_KEYFRAME_NOISE_AUG
        const_timesteps = torch.tensor([CONST_NOISE_AUG, 1.0], dtype=torch.float32)
        logger.info(f"Precomputing constant timesteps: {const_timesteps.tolist()}")
        tensors_to_save["const_timesteps"] = const_timesteps.cpu().contiguous()
        _precompute_adaln_for_timesteps(
            model, const_timesteps, device, "const", tensors_to_save
        )

    # --- Save to safetensors with metadata ---
    metadata = {
        "num_steps": str(num_steps),
        "num_eval_steps": str(num_eval_steps),
        "video_shift": str(video_shift),
        "audio_shift": str(audio_shift),
        "model_variant": model_variant,
        "num_blocks": str(num_blocks),
        "has_const_entries": "true",
    }
    logger.info(f"Saving cache to {output_path}...")
    logger.info(f"  {len(tensors_to_save)} tensors")
    logger.info(f"  metadata: {metadata}")
    save_file(tensors_to_save, output_path, metadata=metadata)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Cache file size: {size_mb:.1f} MB")
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute AdaLN cache for MiniMax-H3"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to transformer directory (e.g. ./weights/base/transformer). "
             "Parent must contain modular_model_index.json."
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of actual H3 Base denoising steps"
    )
    parser.add_argument(
        "--video_shift", type=float, default=12.0,
        help="Video scheduler shift (default: 12.0)"
    )
    parser.add_argument(
        "--audio_shift", type=float, default=3.0,
        help="Audio scheduler shift (default: 3.0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output safetensors cache file path. If provided, precompute AdaLN cache."
    )
    parser.add_argument(
        "--model_variant", type=str, default="FL2VA",
        choices=["FL2VA", "Ref2VA"],
        help="Model variant (default: FL2VA)"
    )
    parser.add_argument(
        "--strip_output", type=str, default=None,
        help="If provided, strip adaln weights from the checkpoint and "
             "write the stripped checkpoint to this directory. "
             "Produces a smaller model (~36GB from ~62GB) that requires "
             "--adaln_cache_file at inference time."
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be a positive integer")

    if not args.output and not args.strip_output:
        parser.error("at least one of --output or --strip_output is required")

    if args.output:
        precompute_cache(
            model_path=args.model_path,
            num_steps=args.steps + 1,
            video_shift=args.video_shift,
            audio_shift=args.audio_shift,
            output_path=args.output,
            model_variant=args.model_variant,
        )

    if args.strip_output:
        logger.info(f"Stripping adaln weights from {args.model_path} → {args.strip_output}")
        strip_checkpoint(args.model_path, args.strip_output)


if __name__ == "__main__":
    main()
