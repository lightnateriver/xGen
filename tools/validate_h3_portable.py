"""Multi-rank CUDA/NPU BF16 component gate, random weights, no model assets.

XGEN_DEVICE=npu torchrun --standalone --nproc-per-node=8 tools/validate_h3_portable.py --device npu --output /tmp/h3-gate
This validates collective/layout math, not trained-model video quality.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def compare(expected, actual):
    a, b = expected.float(), actual.float()
    result = dict(max_abs=(a-b).abs().max().item(),
                  relative_rmse=((a-b).square().mean()/a.square().mean().clamp_min(1e-12)).sqrt().item(),
                  finite=bool(torch.isfinite(b).all()))
    assert result['finite'] and result['relative_rmse'] < .01, result
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'npu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'npu':
        import torch_npu  # noqa: F401
    import x_diffusers  # noqa: F401
    accelerator = torch.get_device_module(args.device)
    accelerator.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl' if args.device == 'npu' else 'nccl')
    assert dist.get_world_size() in (2, 4, 8)
    device = torch.device(args.device, int(os.environ['LOCAL_RANK']))
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefiner
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from x_diffusers.adaptor.h3_input_refiner import install_input_refiner
    from x_diffusers.adaptor.h3_vision_sp import install_vision_sp
    from x_diffusers.adaptor.h3_cuda_refiner import install_ff_chunking
    from x_base import get_pad, set_pad
    torch.manual_seed(42)
    reference = MiniMaxH3TokenRefiner(128, 8, 32, 256, 2, 1e-5, 1e-5, 1e-5).bfloat16().eval().to(device)
    model = SimpleNamespace(token_refiner=deepcopy(reference),
        parallel_manager=SimpleNamespace(sp_group=dist.group.WORLD, tp_size=1))
    install_input_refiner(model, 'custom', sp=True, attention='legacy', ffn='bf16')
    install_ff_chunking(model, 128)
    records = []
    for n in (3, 33, 259):
        x = torch.randn(1, n, 128, dtype=torch.bfloat16, device=device)
        previous = set_pad('mmh3_seq', 101, dist.group.WORLD)
        expected, actual = reference(x), model.token_refiner(x)
        assert previous == get_pad('mmh3_seq')
        records.append(dict(tokens=n, **compare(expected, actual)))
    config = Qwen3VLVisionConfig(hidden_size=128, num_heads=8, depth=2, intermediate_size=256,
        out_hidden_size=128, patch_size=4, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=64, deepstack_visual_indexes=[0])
    config._attn_implementation = 'sdpa'
    visual = Qwen3VLVisionModel(config).bfloat16().eval().to(device)
    parallel = deepcopy(visual)
    install_vision_sp(parallel, lambda: dist.group.WORLD)
    grid = torch.tensor([[1, 8, 8], [1, 6, 6]], device=device)
    pixels = torch.randn(100, 3*2*4*4, device=device, dtype=torch.bfloat16)
    expected, actual = visual(pixels, grid), parallel(pixels, grid)
    vision = dict(hidden=compare(expected.last_hidden_state, actual.last_hidden_state),
                  pooled=compare(expected.pooler_output, actual.pooler_output),
                  deep=[compare(a, b) for a, b in zip(expected.deepstack_features, actual.deepstack_features)])
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed', rank=dist.get_rank(), device=str(device), input_refiner=records,
                  vision=vision, refiner_metrics=model.token_refiner._h3_sp_metrics,
                  vision_metrics=parallel._h3_sp_metrics, scope='synthetic BF16 component math only')
    (args.output/f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, indent=2))
    dist.barrier()
    if dist.get_rank() == 0:
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
