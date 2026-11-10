"""CUDA/NPU trained Audio VAE parity and encode->decode staging gate.

Requires an Audio VAE checkpoint only. Does not launch the video DiT.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def comparison(expected, actual):
    a, b = expected.float(), actual.float()
    assert a.shape == b.shape and torch.isfinite(b).all()
    error = (a-b).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-8)
    result = dict(shape=list(a.shape), relative_rmse=error.item(), max_abs=(a-b).abs().max().item())
    assert result['relative_rmse'] < .001, result
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'npu'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'npu':
        import torch_npu  # noqa: F401
    import x_diffusers  # noqa: F401
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from x_diffusers.adaptor.h3_features import configure_components
    accelerator = torch.get_device_module(args.device)
    if args.device == 'npu':
        accelerator.config.allow_internal_format = False
        accelerator.set_compile_mode(jit_compile=False)
    accelerator.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl' if args.device == 'npu' else 'nccl')
    device = torch.device(args.device, int(os.environ['LOCAL_RANK']))
    native = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
    parallel = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False)
    pipe = SimpleNamespace(audio_vae=parallel, vae=SimpleNamespace(decode=lambda *a, **kw: None),
        transformer=SimpleNamespace(parallel_manager=SimpleNamespace(
            sp_group=dist.group.WORLD, sp_size=dist.get_world_size())))
    configure_components(pipe, SimpleNamespace(h3_cuda_audio_encoder_parallel=True,
        h3_cuda_audio_decoder_parallel='long'), device)
    torch.manual_seed(42)
    wave = torch.randn(2, 1, 160000, device=device)*.03
    expected = native.encode(wave, return_dict=False)[0].mode()
    actual = parallel.encode(wave, return_dict=False)[0].mode()
    encoder = comparison(expected, actual)
    assert all(p.device.type == 'cpu' for p in parallel.parameters())
    records = []
    for frames in (203, 405, 603):
        latent = torch.randn(2, 32, frames, device=device)*.1
        expected = native.decode(latent, return_dict=False)[0]
        actual = parallel.decode(latent, return_dict=False)[0]
        assert all(p.device.type == 'cpu' for p in parallel.parameters())
        records.append(dict(latents=frames, **comparison(expected, actual)))
    report = dict(status='passed', rank=dist.get_rank(), world=dist.get_world_size(), device=str(device),
        encoder=encoder, decoder=records, metrics=parallel._h3_parallel_metrics,
        scope='Audio VAE component parity and CPU return, not generated-video quality')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, indent=2))
    dist.barrier()
    if dist.get_rank() == 0:
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
