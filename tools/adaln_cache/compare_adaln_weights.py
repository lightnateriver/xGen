#!/usr/bin/env python3
"""Compare adaln-related weights between two MiniMax-H3 transformer checkpoints.

Verifies whether transformer (fl2va) and transformer_ref (ref2va) share
identical adaln weights. If they do, a single cache file suffices for both.

Usage:
    python compare_adaln_weights.py \
        --path_a /data/models/minimax/transformer \
        --path_b /data/models/minimax/transformer_ref
"""

import argparse
import fnmatch
import json
import os
import sys

import torch
from safetensors import safe_open


# AdaLN-related key patterns (same as DELETE_PATTERNS in precompute script).
ADALN_PATTERNS = [
    "transformer_blocks.*.adaln_proj.linear.weight",
    "transformer_blocks.*.adaln_proj.linear.bias",
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
    "norm_out.linear.weight",
    "norm_out.linear.bias",
]


def is_adaln_key(key: str) -> bool:
    return any(fnmatch.fnmatch(key, pat) for pat in ADALN_PATTERNS)


def load_keys(path: str) -> dict[str, str]:
    """Map key -> filepath for all safetensors files in a directory."""
    index_json = os.path.join(path, "model_index.json")
    safetensors_index = os.path.join(path, "model.safetensors.index.json")

    files = []
    if os.path.isfile(safetensors_index):
        with open(safetensors_index) as f:
            idx = json.load(f)
        files = sorted(set(idx["weight_map"].values()))
    else:
        files = sorted(
            f for f in os.listdir(path)
            if f.endswith(".safetensors")
        )

    key_to_file = {}
    for fname in files:
        fpath = os.path.join(path, fname)
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                key_to_file[key] = fpath
    return key_to_file


def load_tensor(filepath: str, key: str) -> torch.Tensor:
    with safe_open(filepath, framework="pt") as f:
        return f.get_tensor(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_a", required=True, help="transformer checkpoint dir")
    parser.add_argument("--path_b", required=True, help="transformer_ref checkpoint dir")
    args = parser.parse_args()

    print(f"Loading key lists from:\n  A: {args.path_a}\n  B: {args.path_b}")

    keys_a = load_keys(args.path_a)
    keys_b = load_keys(args.path_b)

    adaln_keys_a = {k for k in keys_a if is_adaln_key(k)}
    adaln_keys_b = {k for k in keys_b if is_adaln_key(k)}

    print(f"\nAdaLN keys in A: {len(adaln_keys_a)}")
    print(f"AdaLN keys in B: {len(adaln_keys_b)}")

    only_a = adaln_keys_a - adaln_keys_b
    only_b = adaln_keys_b - adaln_keys_a
    common = adaln_keys_a & adaln_keys_b

    if only_a:
        print(f"\nKeys only in A ({len(only_a)}):")
        for k in sorted(only_a):
            print(f"  {k}")
    if only_b:
        print(f"\nKeys only in B ({len(only_b)}):")
        for k in sorted(only_b):
            print(f"  {k}")

    print(f"\nCommon adaln keys: {len(common)}")
    if not common:
        print("No common adaln keys — checkpoints have different structure.")
        sys.exit(1)

    # Compare values
    identical = 0
    different = 0
    different_keys = []

    for key in sorted(common):
        tensor_a = load_tensor(keys_a[key], key)
        tensor_b = load_tensor(keys_b[key], key)

        if tensor_a.shape != tensor_b.shape:
            print(f"  [SHAPE DIFF] {key}: A={list(tensor_a.shape)} B={list(tensor_b.shape)}")
            different += 1
            different_keys.append(key)
            continue

        if torch.equal(tensor_a, tensor_b):
            identical += 1
        else:
            diff = (tensor_a.float() - tensor_b.float()).abs()
            print(f"  [VALUE DIFF] {key}: max_abs={diff.max().item():.6e} "
                  f"mean_abs={diff.mean().item():.6e}")
            different += 1
            different_keys.append(key)

    print(f"\n{'='*60}")
    print(f"Summary: {identical} identical, {different} different (out of {len(common)} common)")
    print(f"{'='*60}")

    if different == 0:
        print("\n✅ ALL adaln weights are IDENTICAL.")
        print("   A single adaln_cache file works for both transformer and transformer_ref.")
    else:
        print(f"\n❌ {different} adaln weights DIFFER between the two checkpoints.")
        print("   Separate adaln_cache files are needed for fl2va and ref2va.")
        print("   Different keys:")
        for k in different_keys:
            print(f"     {k}")


if __name__ == "__main__":
    main()
