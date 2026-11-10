"""Trained video VAE tile-encoder parity and staging on CUDA/NPU.

Uses only the VAE checkpoint, not the video DiT.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'npu'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quant-decoder', action='store_true', help='NPU INT8 decoder plus pipeline.to regression')
    args = parser.parse_args()
    if args.device == 'npu':
        import torch_npu  # noqa: F401
        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)
    import x_diffusers  # noqa: F401
    from diffusers import AutoencoderKLMiniMaxH3, DiffusionPipeline
    from x_diffusers.adaptor.h3_condition_parallel import install_video_encoder
    accelerator = torch.get_device_module(args.device)
    accelerator.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl' if args.device == 'npu' else 'nccl')
    device = torch.device(args.device, int(os.environ['LOCAL_RANK']))
    models = [AutoencoderKLMiniMaxH3.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(device) for _ in range(2)]
    for model in models:
        model.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
            tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    native, parallel = models
    if args.quant_decoder:
        assert args.device == 'npu'
        from types import SimpleNamespace
        from x_base import matmul_manager
        from x_diffusers.adaptor.h3_features import configure_components
        for model in models:
            model.decoder.bfloat16()
            for layer in model.decoder.modules():
                if isinstance(layer, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.RMSNorm)):
                    layer.float()
            matmul_manager.enable_vae_linear_quant(model, dtype='int8')
        native.to(device)
        class VAEPipeline(DiffusionPipeline):
            def __init__(self, vae):
                super().__init__()
                self.register_modules(vae=vae)
        pipe = VAEPipeline(parallel)
        pipe.transformer = SimpleNamespace(parallel_manager=SimpleNamespace(
            sp_group=dist.group.WORLD, sp_size=dist.get_world_size()))
        configure_components(pipe, SimpleNamespace(h3_cuda_vae_encoder_parallel=True), device)
        pipe.to(device)
        assert all(p.device.type == args.device for p in parallel.decoder.parameters())
    else:
        install_video_encoder(parallel, lambda: dist.group.WORLD, device)
    records = []
    for frames, size in ((1, 512), (1, 256), (5, 512)):
        torch.manual_seed(frames+size)
        pixels = torch.randn(1, 3, frames, size, size, device=device, dtype=torch.bfloat16)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            expected = native.encode(pixels, return_dict=False)[0].mode()
            actual = parallel.encode(pixels, return_dict=False)[0].mode()
        assert expected.shape == actual.shape and torch.isfinite(actual).all()
        a, b = expected.float(), actual.float()
        error = ((a-b).square().mean()/a.square().mean().clamp_min(1e-12)).sqrt().item()
        records.append(dict(frames=frames, size=size, shape=list(a.shape),
                            max_abs=(a-b).abs().max().item(), relative_rmse=error))
        assert error < .01, records[-1]
        assert all(p.device.type == 'cpu' for p in parallel.encoder.parameters())
        if args.quant_decoder:
            # Image encodes have only one temporal latent. Exercise one valid
            # complete decoder clip instead of the generation-only streaming
            # route, which requires a full temporal chunk.
            count = native.tokens_chunk_size + native.token_overlap
            clip_reference = expected[:, :, -1:].repeat(1, 1, count, 1, 1)
            clip = actual[:, :, -1:].repeat(1, 1, count, 1, 1)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                decoded_reference = native._decode_clip(clip_reference)
                decoded = parallel._decode_clip(clip)
            a, b = decoded_reference.float(), decoded.float()
            error = ((a-b).square().mean()/a.square().mean().clamp_min(1e-12)).sqrt().item()
            records[-1]['decode_relative_rmse'] = error
            assert error < .01 and torch.isfinite(b).all(), records[-1]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed', rank=dist.get_rank(), world=dist.get_world_size(), device=str(device),
        cases=records, quant_decoder=args.quant_decoder, metrics=parallel._h3_encoder_parallel_metrics,
        scope='VAE encoder BF16 parity, no DiT or generated-video quality claim')
    (args.output/f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, indent=2))
    dist.barrier()
    if dist.get_rank() == 0:
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
