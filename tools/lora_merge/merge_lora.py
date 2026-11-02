#!/usr/bin/env python3
"""Merge ComfyUI-format LoRA into MiniMax-H3 diffusers weights (one-step).

This script reads a ComfyUI LoRA safetensors file and merges it directly into
the **top-level diffusers-format** checkpoint (``/data/models/minimax/transformer/``),
producing merged weights that are ready for ``ModularPipeline.from_pretrained()``.

One-step approach:
  - Base = top-level diffusers checkpoint (``diffusion_pytorch_model-*.safetensors``)
  - For each LoRA target, compute delta and **convert** to diffusers key layout:
      * QKV split:  ``qkv_proj`` delta [21504, 5376] → ``to_q`` / ``to_k`` / ``to_v``
      * SwiGLU swap: ``mlp.fc1`` delta rows [gate;value] → [value;gate] for ``ff.net.0.proj``
      * Key rename:  ``blocks.N`` → ``transformer_blocks.N``, etc.
  - Non-LoRA keys are copied as-is from the top-level checkpoint.

Three transformation traps:
  1. Key prefix: strip ``diffusion_model.`` from LoRA keys
  2. SwiGLU row swap: ComfyUI [gate;value] → diffusers [value;gate]
  3. LoRA scaling: delta = (alpha / rank) × (B @ A), NOT alpha × (B @ A)

Usage:
    python merge_lora.py \
        --lora-path /data/weights/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
        --base-dir /data/models/minimax/transformer \
        --output-dir /data/models/minimax/transformer_8step \
        [--verify]

Environment:
    CPU only. ~16GB RAM. No GPU/NPU needed.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from typing import Callable, Optional, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMFYUI_PREFIX = "diffusion_model."
SUFFIX_ALPHA = ".alpha"
SUFFIX_LORA_A = ".lora_A.weight"
SUFFIX_LORA_B = ".lora_B.weight"

# QKV split sizes (MiniMax-H3: hidden=5376, head_dim=128, num_heads=56)
QKV_OUT_DIM = 21504   # 3 × 7168
QKV_SPLIT = 7168      # 21504 / 3


# ---------------------------------------------------------------------------
# Step 0: Check original shard metadata
# ---------------------------------------------------------------------------

def check_shard_metadata(base_dir: str) -> Optional[dict]:
    """Read metadata from the first shard to ensure save_file consistency."""
    index_path = _find_index(base_dir)
    if not index_path:
        raise FileNotFoundError(f"No safetensors index found in {base_dir}")

    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    first_shard = os.path.join(base_dir, shard_files[0])

    with safe_open(first_shard, framework="pt") as f:
        metadata = f.metadata()

    logger.info("Original shard metadata: %s", metadata)
    return metadata


# ---------------------------------------------------------------------------
# Step 1: Build LoRA mapping
# ---------------------------------------------------------------------------

def build_lora_map(lora_path: str) -> dict[str, dict]:
    """Load LoRA safetensors and build {target_key: {alpha, lora_A, lora_B}}.

    Keys use FL2VA/ComfyUI naming (``blocks.N.attn.qkv_proj`` etc.).
    """
    lora_map: dict[str, dict] = {}

    with safe_open(lora_path, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            clean = k.replace(COMFYUI_PREFIX, "")

            if clean.endswith(SUFFIX_ALPHA):
                target = clean[: -len(SUFFIX_ALPHA)]
                lora_map.setdefault(target, {})["alpha"] = f.get_tensor(k).item()
            elif clean.endswith(SUFFIX_LORA_A):
                target = clean[: -len(SUFFIX_LORA_A)]
                lora_map.setdefault(target, {})["lora_A"] = f.get_tensor(k)
            elif clean.endswith(SUFFIX_LORA_B):
                target = clean[: -len(SUFFIX_LORA_B)]
                lora_map.setdefault(target, {})["lora_B"] = f.get_tensor(k)

    incomplete = [
        t for t, v in lora_map.items()
        if not all(k in v for k in ("alpha", "lora_A", "lora_B"))
    ]
    if incomplete:
        logger.warning("Incomplete LoRA targets: %s", incomplete)

    logger.info("Loaded %d LoRA targets", len(lora_map))
    return lora_map


# ---------------------------------------------------------------------------
# Step 2: Build FL2VA → diffusers key mapping
# ---------------------------------------------------------------------------

def build_key_mapping(lora_map: dict[str, dict]) -> dict[str, list[tuple[str, Optional[Callable]]]]:
    """Build mapping from FL2VA LoRA target key → list of (diffusers_key, transform_fn).

    transform_fn takes the delta tensor and returns the converted delta.
    None means no transform (direct copy).

    Mapping rules:
      blocks.N.attn.qkv_proj  → transformer_blocks.N.attn.to_q / to_k / to_v (split)
      blocks.N.attn.out_proj  → transformer_blocks.N.attn.to_out.0
      blocks.N.mlp.fc1         → transformer_blocks.N.ff.net.0.proj (SwiGLU swap)
      blocks.N.mlp.fc2         → transformer_blocks.N.ff.net.2
      token_refiner.blocks.N.* → token_refiner.refiner_blocks.N.*
    """
    mapping: dict[str, list[tuple[str, Optional[Callable]]]] = {}

    for fl2va_key in lora_map:
        # Determine block prefix
        if fl2va_key.startswith("token_refiner.blocks."):
            # token_refiner.blocks.N.xxx → token_refiner.refiner_blocks.N.xxx
            m = re.match(r"token_refiner\.blocks\.(\d+)\.(.+)", fl2va_key)
            if not m:
                continue
            block_idx = m.group(1)
            suffix = m.group(2)
            prefix = f"token_refiner.refiner_blocks.{block_idx}"
        elif fl2va_key.startswith("blocks."):
            m = re.match(r"blocks\.(\d+)\.(.+)", fl2va_key)
            if not m:
                continue
            block_idx = m.group(1)
            suffix = m.group(2)
            prefix = f"transformer_blocks.{block_idx}"
        else:
            continue

        if suffix == "attn.qkv_proj":
            # Split QKV delta into to_q, to_k, to_v
            def split_qkv(delta, _idx=block_idx, _prefix=prefix):
                return [
                    (f"{_prefix}.attn.to_q.weight", delta[0:QKV_SPLIT, :].clone()),
                    (f"{_prefix}.attn.to_k.weight", delta[QKV_SPLIT:2*QKV_SPLIT, :].clone()),
                    (f"{_prefix}.attn.to_v.weight", delta[2*QKV_SPLIT:, :].clone()),
                ]
            mapping[fl2va_key] = [("__split__", split_qkv)]

        elif suffix == "attn.out_proj":
            diffusers_key = f"{prefix}.attn.to_out.0.weight"
            mapping[fl2va_key] = [(diffusers_key, None)]

        elif suffix == "mlp.fc1":
            # SwiGLU: swap [gate;value] → [value;gate]
            def swap_swiglu(delta, _key=f"{prefix}.ff.net.0.proj.weight"):
                half = delta.shape[0] // 2
                swapped = torch.cat([delta[half:].clone(), delta[:half].clone()], dim=0)
                return [(_key, swapped)]
            mapping[fl2va_key] = [("__swap__", swap_swiglu)]

        elif suffix == "mlp.fc2":
            diffusers_key = f"{prefix}.ff.net.2.weight"
            mapping[fl2va_key] = [(diffusers_key, None)]

        else:
            logger.warning("Unknown LoRA target suffix: %s (key=%s)", suffix, fl2va_key)

    return mapping


# ---------------------------------------------------------------------------
# Step 3: Compute delta
# ---------------------------------------------------------------------------

def compute_delta(target_key: str, lora_info: dict) -> torch.Tensor:
    """Compute delta = (alpha / rank) × (B @ A).

    ComfyUI stores raw alpha; the effective scaling is alpha / rank
    (confirmed from ComfyUI source: comfy/weight_adapter/lora.py).
    No special QKV alpha handling — the same alpha/rank applies to all targets.

    No SwiGLU swap here — that's done in the key mapping step.
    """
    alpha = lora_info["alpha"]
    lora_A = lora_info["lora_A"]
    lora_B = lora_info["lora_B"]

    rank = lora_A.shape[0]  # lora_A: [rank, in_features]
    scaling = alpha / rank

    delta = scaling * (lora_B.to(torch.float32) @ lora_A.to(torch.float32))
    return delta.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_index(base_dir: str) -> Optional[str]:
    """Return path to the first existing safetensors index file."""
    for name in [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ]:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def _apply_delta(tensors: dict, key: str, delta_slice: torch.Tensor, label: str = ""):
    """Add delta_slice to tensors[key] in-place (float32 math, bfloat16 output)."""
    if key not in tensors:
        return
    base = tensors[key].to(torch.float32)
    tensors[key] = (base + delta_slice.to(torch.float32)).to(torch.bfloat16)
    logger.info("    %s: %s", label, key)


def _collect_diffusers_keys(key_mapping: dict) -> set[str]:
    """Collect all diffusers keys that will receive LoRA deltas."""
    keys = set()
    for targets in key_mapping.values():
        for tag, fn in targets:
            if tag in ("__split__", "__swap__"):
                dummy = torch.zeros(
                    QKV_OUT_DIM if tag == "__split__" else 2, 1, dtype=torch.bfloat16,
                )
                for dk, _ in fn(dummy):
                    keys.add(dk)
            else:
                keys.add(tag)
    return keys


# ---------------------------------------------------------------------------
# Step 5: Process one shard
# ---------------------------------------------------------------------------

def merge_shard(
    shard_filename: str,
    base_dir: str,
    output_dir: str,
    lora_map: dict[str, dict],
    key_mapping: dict[str, list],
    shard_metadata: Optional[dict],
) -> dict:
    """Load top-level shard, merge LoRA deltas, save to output.

    For each tensor in the shard:
      - Check if its key matches a LoRA target (via key_mapping)
      - If yes: compute delta, apply transform, add to base
      - If no: copy as-is
    """
    shard_path = os.path.join(base_dir, shard_filename)
    output_path = os.path.join(output_dir, shard_filename)

    # Load all tensors from this shard
    tensors = {}
    with safe_open(shard_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Single pass: for each LoRA target, compute delta and apply
    for fl2va_key, targets in key_mapping.items():
        if fl2va_key not in lora_map:
            continue
        delta = compute_delta(fl2va_key, lora_map[fl2va_key])

        for tag, fn in targets:
            if tag in ("__split__", "__swap__"):
                label = "QKV split" if tag == "__split__" else "SwiGLU swap"
                for dk, dv in fn(delta):
                    _apply_delta(tensors, dk, dv, label)
            elif fn is not None:
                result = fn(delta)
                if isinstance(result, list):
                    for dk, dv in result:
                        _apply_delta(tensors, dk, dv, "Transform")
            else:
                _apply_delta(tensors, tag, delta, "Direct")

    # Save
    save_file(
        tensors,
        output_path,
        metadata=shard_metadata or {"format": "pt"},
    )

    del tensors
    logger.info("  Saved %s", shard_filename)
    return {}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_output(
    base_dir: str,
    output_dir: str,
    lora_map: dict[str, dict],
) -> bool:
    """Post-merge verification: delta range + NaN/Inf checks on LoRA keys."""
    ok = True

    index_path = _find_index(base_dir)
    if not index_path:
        logger.error("Index not found for verification")
        return False

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    key_mapping = build_key_mapping(lora_map)
    lora_keys = _collect_diffusers_keys(key_mapping)

    delta_stats = []
    for shard_file in sorted(set(weight_map.values())):
        shard_keys = [k for k, v in weight_map.items() if v == shard_file and k in lora_keys]
        if not shard_keys:
            continue

        with safe_open(os.path.join(base_dir, shard_file), framework="pt") as bf, \
             safe_open(os.path.join(output_dir, shard_file), framework="pt") as of:
            for key in shard_keys:
                delta = (of.get_tensor(key).to(torch.float32)
                         - bf.get_tensor(key).to(torch.float32))
                has_nan = torch.isnan(delta).any().item()
                has_inf = torch.isinf(delta).any().item()
                delta_stats.append({
                    "key": key,
                    "abs_max": delta.abs().max().item(),
                    "abs_mean": delta.abs().mean().item(),
                    "has_nan": has_nan, "has_inf": has_inf,
                })
                if has_nan or has_inf:
                    logger.error("  NaN/Inf in delta for %s", key)
                    ok = False

    logger.info("Verification: %d LoRA deltas checked", len(delta_stats))
    if delta_stats:
        logger.info(
            "Delta stats: abs_max [%.6f, %.6f], abs_mean [%.6f, %.6f]",
            min(s["abs_max"] for s in delta_stats), max(s["abs_max"] for s in delta_stats),
            min(s["abs_mean"] for s in delta_stats), max(s["abs_mean"] for s in delta_stats),
        )

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge ComfyUI LoRA into MiniMax-H3 diffusers weights (one-step)",
    )
    parser.add_argument(
        "--lora-path", type=str, required=True,
        help="Path to ComfyUI LoRA safetensors file",
    )
    parser.add_argument(
        "--base-dir", type=str, required=True,
        help="Path to top-level diffusers transformer directory "
             "(containing diffusion_pytorch_model-*.safetensors + index.json)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for merged weights",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run post-merge verification",
    )
    args = parser.parse_args()

    if not os.path.exists(args.lora_path):
        logger.error("LoRA file not found: %s", args.lora_path)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 0: Check shard metadata
    logger.info("Step 0: Checking original shard metadata...")
    shard_metadata = check_shard_metadata(args.base_dir)

    # Step 1: Build LoRA mapping
    logger.info("Step 1: Building LoRA mapping...")
    lora_map = build_lora_map(args.lora_path)
    if not lora_map:
        logger.error("No LoRA targets found.")
        sys.exit(1)

    # Step 2: Build key mapping (FL2VA → diffusers)
    logger.info("Step 2: Building key mapping...")
    key_mapping = build_key_mapping(lora_map)
    logger.info("  %d FL2VA keys mapped to diffusers keys", len(key_mapping))

    # Step 3: Copy meta files
    logger.info("Step 3: Copying meta files...")
    for fname in os.listdir(args.base_dir):
        if fname.endswith(".safetensors") or fname == ".gitattributes":
            continue
        src = os.path.join(args.base_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))
            logger.info("  Copied %s", fname)

    # Step 4: Get shard list
    index_path = _find_index(args.base_dir)
    if not index_path:
        logger.error("Index not found")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    logger.info("Step 4: Found %d shards to process", len(shard_files))

    # Step 5: Merge each shard
    logger.info("Step 5: Merging shards...")
    for i, shard_file in enumerate(shard_files, 1):
        logger.info("  [%d/%d] %s", i, len(shard_files), shard_file)
        merge_shard(
            shard_filename=shard_file,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            lora_map=lora_map,
            key_mapping=key_mapping,
            shard_metadata=shard_metadata,
        )

    # Step 6: Verify (optional)
    if args.verify:
        logger.info("Step 6: Running verification...")
        ok = verify_output(args.base_dir, args.output_dir, lora_map)
        if ok:
            logger.info("  ✅ Verification PASSED")
        else:
            logger.error("  ❌ Verification FAILED")
            sys.exit(1)
    else:
        logger.info("Step 6: Skipped verification (use --verify to enable)")

    logger.info("Done! Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
