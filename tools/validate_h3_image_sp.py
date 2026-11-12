"""Compare all PDD heads to upstream on packed, uneven sequences (CPU or NPU)."""
import argparse
import copy
import json
import os

import torch
import torch.distributed as dist
_cpu_sdpa = torch.nn.functional.scaled_dot_product_attention
_cpu_rms_norm = torch.nn.functional.rms_norm
from diffusers import MiniMaxH3ImageTransformer2DModel
from x_diffusers.framework.transformer.minimax_h3_image import enable_h3_image_sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--local-input", action="store_true")
    args = parser.parse_args()
    device = args.device
    if device == "cpu":
        # Optional NPU adapters redirect SDPA process-wide; retain the real CPU oracle here.
        torch.nn.functional.scaled_dot_product_attention = _cpu_sdpa
        torch.nn.functional.rms_norm = _cpu_rms_norm
    torch.set_num_threads(2)
    if device == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        torch.npu.set_compile_mode(jit_compile=False)
    elif device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend={"cpu": "gloo", "npu": "hccl", "cuda": "nccl"}[device])
    world = dist.get_world_size()
    torch.manual_seed(20261111)
    dense = MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=8, attention_head_dim=16, hidden_size=32, num_layers=2,
        num_refiner_layers=1, ffn_dim=64, in_channels=2, text_dim=16,
        freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2,
        num_pdd_heads=6,
    ).eval()
    if args.dtype == "bf16":
        dense.bfloat16()
        # Match the real checkpoint's mixed precision, including all PDD heads.
        for module in (dense.proj_in, dense.time_embedder, dense.pdd_proj_out, dense.rope):
            module.float()
    dense.to(device)
    parallel = copy.deepcopy(dense)
    enable_h3_image_sp(parallel, world)
    if args.local_input:
        from x_diffusers.framework.transformer.h3_image_optimizations import (
            install_image_refiner_sp, install_image_condition_cache, install_image_local_input)
        install_image_refiner_sp(parallel)
        install_image_condition_cache(parallel)
        install_image_local_input(parallel)
    results = []
    try:
        with torch.inference_mode():
            # Change length across requests; include interleaved text/reference/target rows.
            for length in ((13, 16, 19, 12) if args.local_input else (13, 16, 19)):
                if args.local_input:
                    from x_diffusers.framework.transformer.h3_image_optimizations import clear_image_condition_cache
                    clear_image_condition_cache(parallel)
                text_rows = list(range(10)) if args.local_input and length == 12 else [1, 4, 7]
                text = torch.tensor(text_rows, device=device)
                image = torch.tensor([i for i in range(length) if i not in text_rows], device=device)
                tags = torch.zeros(length, device=device, dtype=torch.long)
                tags[text] = 1
                times = torch.zeros(length, device=device, dtype=torch.long)
                times[image[:2]] = 1
                values = dict(
                    hidden_states=torch.randn(1, len(image), 8, device=device),
                    encoder_hidden_states=torch.randn(1, len(text_rows), 16, device=device),
                    timestep=torch.tensor([0.75, 0.999], device=device),
                    timestep_indices=times, token_tags=tags,
                    position_ids=torch.arange(length * 3, device=device).reshape(length, 3),
                    image_indices=image, text_indices=text, return_dict=False,
                )
                for head in range(6):
                    expected = dense(**values, pdd_step_index=head)[0]
                    actual = parallel(**values, pdd_step_index=head)[0]
                    tolerance = 5e-3 if args.dtype == "bf16" else 2e-5
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    results.append(dict(length=length, head=head,
                                        max_abs=(actual-expected).abs().max().item()))
        if dist.get_rank() == 0:
            print(json.dumps(dict(device=device, dtype=args.dtype, world_size=world,
                                  comparisons=results, passed=True)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
