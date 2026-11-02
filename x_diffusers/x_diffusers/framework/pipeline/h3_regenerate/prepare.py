"""CPU LoRA merge and NPU AdaLN precomputation; never modify source weights."""
import argparse
import gc
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .vendor import adapters
from .vendor.sigmas import SCHEDULES
from .vendor import geometry
from .assets import digest, model_digests, validate_refresh
from .state import atomic_json

CURVE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"


@torch.inference_mode()
def prepare(weights, output, refresh_cache=False):
    import torch_npu
    from diffusers import MiniMaxH3Transformer3DModel
    weights, output = Path(weights), Path(output)
    if refresh_cache and not output.is_dir():
        raise ValueError("Cache refresh requires an existing prepared directory")
    if not refresh_cache and output.exists():
        raise FileExistsError(output)
    if digest(weights / adapters.CURVE_TABLE) != CURVE_SHA256:
        raise ValueError("Unexpected Sharp curve table checksum")
    torch.set_num_threads(16)
    fingerprint = {name: digest(weights / name) for name in
                   (adapters.LORA_FILE, adapters.CURVE_FILE, adapters.CURVE_TABLE)}
    base_digests = model_digests(weights / "base_model" / "transformer")
    previous = validate_refresh(output, fingerprint, base_digests) if refresh_cache else None
    pairs = adapters.load_lora(weights / adapters.LORA_FILE)
    if refresh_cache and any(any(part in name for part in ("adaln", "time_embedder", "norm_out")) for name in pairs):
        raise ValueError("Cache refresh requires re-merging LoRA affecting time modulation")
    if not refresh_cache:
        output.mkdir(parents=True)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    print("LOAD_BASE", flush=True)
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(weights / "base_model"), subfolder="transformer",
        torch_dtype=torch.bfloat16, local_files_only=True).eval()
    for index, (name, (a, b)) in enumerate(pairs.items()):
        if refresh_cache:
            continue
        layer = model.get_submodule(name)
        assert tuple(layer.weight.shape) == (b.shape[0], a.shape[1])
        for start in range(0, b.shape[0], 256):
            end = start + 256
            fused = layer.weight[start:end].float()
            fused.addmm_(b[start:end], a)
            layer.weight[start:end].copy_(fused.to(layer.weight.dtype))
        if index % 50 == 0:
            print("MERGED", index, len(pairs), flush=True)
    del pairs
    gc.collect()
    # attach validates all branch dimensions. Its ordinary LoRA pairs are already merged.
    pairs, curve = adapters.attach(model, weights)
    del pairs
    curve.strength = 1.0
    model.time_proj.to("npu")
    model.time_embedder.to("npu")
    sigmas = SCHEDULES[3]
    times, embeddings, tables = [], [], {}
    for i, sigma in enumerate(sigmas[:-1]):
        layout = geometry.build_t2va_layout(torch.ones(1, dtype=torch.long),
                                            geometry.MiniMaxH3Geometry(1, 2, 2, 1))
        t, _ = geometry.build_row_timestep_plan(layout, video_sigma=sigma, audio_sigma=0., keyframe_noise_aug=1.)
        t = t.to("npu")
        times.append(t)
        tables[f"step{i}_timesteps"] = t.cpu()
        embeddings.append(F.silu(model.time_embedder(model.time_proj(t).to(torch.bfloat16))))
    for j, block in enumerate(model.transformer_blocks):
        linear = block.adaln_proj.linear.to("npu")
        for i, (t, emb) in enumerate(zip(times, embeddings)):
            curve.pre(model, (), {"timestep": t})
            actual = linear(emb).detach()
            # Explicitly verify Sharp is included before storing the cache.
            expected = linear.base(emb.to(linear.base.weight.dtype))
            delta = F.linear(F.linear(curve.coords, linear.curve_a), linear.curve_b)
            expected = expected + delta.to(expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            tables[f"step{i}_block{j}"] = actual.cpu().contiguous()
        block.adaln_proj.linear = linear.base.cpu()
        print("CACHED_BLOCK", j, flush=True)
    model.norm_out.linear.to("npu")
    for i, emb in enumerate(embeddings):
        tables[f"step{i}_final"] = model.norm_out.linear(emb).cpu().contiguous()
    model.cpu()
    model._forward_pre_hooks.clear()
    from x_base.adaln_cache import release_adaln_weights
    release_adaln_weights(model)
    output_model = output / "transformer"
    if not refresh_cache:
        model.save_pretrained(str(output_model), max_shard_size="4GB")
    # Immutable cache first; the manifest atomically selects a complete version.
    # A failed refresh cannot invalidate the previously selected cache.
    with tempfile.TemporaryDirectory(prefix=".adaln-", dir=output) as temporary:
        cache = Path(temporary) / "cache.safetensors"
        save_file(tables, str(cache), metadata={
            "num_steps": "4", "num_eval_steps": "3", "num_blocks": "50",
            "quantization": "none", "model_variant": "Open-H3-Regenerate-2K",
            "sharp_included": "true"})
        cache_hash = digest(cache)
        cache_name = f"adaln_cache.{cache_hash}.safetensors" if refresh_cache else "adaln_cache.safetensors"
        cache.replace(output / cache_name)
    record = dict(source=str(weights.resolve()), adapters_sha256=fingerprint,
                  sigmas=list(sigmas), dtype="bfloat16", sharp_included=True,
                  curve_cache_exact_check=True,
                  source_base_sha256=base_digests,
                  transformer_sha256=previous["transformer_sha256"] if previous else model_digests(output_model),
                  cache_file=cache_name,
                  upstream_commit="781e8341fb3b29d3aea5d50265772e9619340b66",
                  cache_sha256=cache_hash)
    atomic_json(output / "manifest.json", record)
    print("PREPARED", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--refresh-cache", action="store_true", help="Rebuild cache without repeating unaffected weight merges")
    a = p.parse_args()
    prepare(a.weights, a.output, a.refresh_cache)
