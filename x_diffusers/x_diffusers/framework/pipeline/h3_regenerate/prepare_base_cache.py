"""Prepare the exact Ref2VA AdaLN schedule for twenty or fifty DiT evaluations."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


@torch.inference_mode()
def main():
    from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-steps", type=int, choices=(20, 50), default=20)
    p.add_argument("--model-variant", choices=("Ref2VA", "T2VA"), default="Ref2VA")
    p.add_argument("--stripped-output", help="New directory for the same model with cached AdaLN weights removed")
    a = p.parse_args()
    if a.stripped_output:
        Path(a.stripped_output).mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    schedulers = [MiniMaxH3Scheduler(shift=s) for s in (12., 3.)]
    for scheduler in schedulers:
        scheduler.set_timesteps(num_inference_steps=a.eval_steps+1)
        assert len(scheduler.timesteps) == a.eval_steps
    schedules = [torch.unique(torch.stack([s.timesteps[i] for s in schedulers])).float() for i in range(a.eval_steps)]
    schedules.append(torch.tensor([.999, 1.], dtype=torch.float32))
    model = MiniMaxH3Transformer3DModel.from_pretrained(a.source, torch_dtype=torch.bfloat16,
                                                      local_files_only=True).eval()
    model.time_embedder.to("npu")
    embeddings = [F.silu(model.time_embedder(model.time_proj(t.to("npu")).bfloat16())) for t in schedules]
    tensors = {}
    names = [f"step{i}" for i in range(a.eval_steps)] + ["const"]
    for name, t in zip(names, schedules):
        tensors[name + "_timesteps"] = t.contiguous()
    for i, block in enumerate(model.transformer_blocks):
        linear = block.adaln_proj.linear.to("npu")
        for name, emb in zip(names, embeddings):
            tensors[f"{name}_block{i}"] = linear(emb).cpu().contiguous()
        linear.cpu()
        print("CACHED_BASE_BLOCK", i, flush=True)
    model.norm_out.linear.to("npu")
    for name, emb in zip(names, embeddings):
        tensors[name + "_final"] = model.norm_out.linear(emb).cpu().contiguous()
    metadata = dict(num_steps=str(a.eval_steps+1), num_eval_steps=str(a.eval_steps), video_shift="12.0", audio_shift="3.0",
                    model_variant=a.model_variant, num_blocks="50", has_const_entries="true", quantization="none")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, a.output, metadata=metadata)
    Path(a.output + ".json").write_text(json.dumps(metadata, indent=2))
    if a.stripped_output:
        from x_base.adaln_cache import release_adaln_weights
        release_adaln_weights(model)
        model.save_pretrained(a.stripped_output, safe_serialization=True, max_shard_size="3GB")
    print(f"BASE_{a.eval_steps}_EVAL_CACHE_READY", flush=True)


if __name__ == "__main__":
    main()
