"""Assemble and run the upstream MiniMax-H3-Image modular pipeline."""

import functools
import json
from pathlib import Path

import torch
import torch.distributed as dist


def validate_checkpoint(model_root):
    root = Path(model_root)
    for name in ("modular_model_index.json", "transformer_diffusers/config.json", "vae_diffusers/config.json",
                 "text_encoder/config.json", "pdd_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Missing local H3 Image component: {root / name}")
    pdd = json.loads((root / "pdd_config.json").read_text())
    if pdd.get("num_inference_steps") != 6 or len(pdd.get("sigmas", [])) != 7:
        raise ValueError("Expected the six-step H3 Image PDD checkpoint")


def install_image_rgb8_output(pipe):
    """Transport the released quantized pixels as RGB8, avoiding a float32 D2H round trip."""
    from PIL import Image
    processor = pipe.image_processor
    original = processor.postprocess
    @functools.wraps(original)
    def postprocess(image, output_type="pil", do_denormalize=None):
        if image.device.type != "npu" or output_type != "pil" or processor.config.do_normalize or (do_denormalize is not None and any(do_denormalize)):
            return original(image, output_type=output_type, do_denormalize=do_denormalize)
        # DecodeStep already applies the checkpoint's truncating quantizer before this call.
        pixels = (image * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [Image.fromarray(item) for item in pixels]
    processor.postprocess = postprocess


def _configure_dit(pipe, options, *, root, device):
    from ..transformer.h3_image_optimizations import (
        fuse_image_qkv,
        install_image_adaln_table,
        install_image_dit_tp,
        install_image_ff_chunk,
        install_image_refiner_sp,
        install_image_swiglu,
        load_image_adaln_table,
        quantize_image_blocks,
        save_image_adaln_table,
    )
    from ..transformer.minimax_h3_image import enable_h3_image_sp

    transformer = pipe.transformer
    adaln_table = getattr(options, "image_adaln_table_file", None)
    save_adaln_table = getattr(options, "image_adaln_table_save", None)
    if adaln_table:
        load_image_adaln_table(
            adaln_table, root, transformer, pipe.config.sample_sigmas, pipe.reference_noise_aug, device
        )
    elif getattr(options, "image_adaln_table", False) or save_adaln_table:
        install_image_adaln_table(transformer, pipe.config.sample_sigmas, pipe.reference_noise_aug, device)
        if save_adaln_table and (not dist.is_initialized() or dist.get_rank() == 0):
            save_image_adaln_table(
                save_adaln_table, root, transformer, pipe.config.sample_sigmas, pipe.reference_noise_aug
            )
    if getattr(options, "image_fused_qkv", False):
        fuse_image_qkv(transformer)
    if getattr(options, "image_int8", False):
        if device.type != "npu":
            raise ValueError("Image INT8 requires NPU")
        quantize_image_blocks(transformer)
    dit_tp = getattr(options, "image_dit_tp", 1)
    enable_h3_image_sp(transformer, options.sp, dit_tp)
    if dit_tp > 1:
        install_image_dit_tp(transformer, dist.group.WORLD)
    if getattr(options, "image_refiner_sp", False) and options.sp > 1:
        install_image_refiner_sp(transformer)
    if getattr(options, "image_fused_rope", False):
        for block in transformer.transformer_blocks:
            block.attn._xgen_image_fused_rope = True
    dense_tail = getattr(options, "image_laser_dense_tail", 0)
    if dense_tail:
        for block in transformer.transformer_blocks[-dense_tail:]:
            block.attn._xgen_image_dense_attention = True
    if getattr(options, "image_ff_chunk", 0):
        install_image_ff_chunk(transformer, options.image_ff_chunk)
    if getattr(options, "image_swiglu", "none") != "none":
        install_image_swiglu(transformer, options.image_swiglu)


def _configure_text_encoder(pipe, options, *, device, world, low_cpu_mem_usage):
    from diffusers.hooks import apply_group_offloading
    from x_base import ParallelManager, enable_tp

    if options.text_encoder_tp > 1:
        pipe._xgen_image_tp_manager = ParallelManager(1, 1, 1, tp_size=options.text_encoder_tp)
        enable_tp(pipe.text_encoder, options.text_encoder_tp, target_layers=range(50))
    if getattr(options, "image_text_int8", False):
        from ..transformer.h3_image_optimizations import quantize_image_text_encoder
        quantize_image_text_encoder(pipe.text_encoder)
    if getattr(options, "image_text_resident", False) or options.text_encoder_tp > 1:
        pipe.text_encoder.to(device)
    else:
        apply_group_offloading(pipe.text_encoder, onload_device=device, offload_device=torch.device("cpu"),
                               offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=low_cpu_mem_usage)
    if getattr(options, "image_vision_sp", False) and world > 1:
        from ..transformer.h3_vision_sp import install_vision_sp
        install_vision_sp(pipe.text_encoder.model.visual, lambda: dist.group.WORLD)


def _place_dit(transformer, options, *, device, low_cpu_mem_usage):
    from ..transformer.h3_image_optimizations import install_image_condition_cache, install_image_local_input

    # A BF16 DiT alone exceeds a 64 GiB 910B; stage one block at a time.
    if getattr(options, "image_resident", False):
        transformer.to(device)
    else:
        transformer.enable_group_offload(
            onload_device=device, offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=low_cpu_mem_usage)
    if getattr(options, "image_condition_cache", False):
        install_image_condition_cache(transformer)
    if getattr(options, "image_local_input", False):
        install_image_local_input(transformer)


@torch.inference_mode(False)
@torch.no_grad()
def load_pipeline(model_root, options, *, device, world):
    """Load checkpoint components and install the selected xGen device policies."""
    try:
        from diffusers import MiniMaxH3ImageModularPipeline
    except ImportError as error:
        raise ImportError("H3 Image needs the pinned optional Diffusers source in requirements-h3-image.txt") from error
    from ..transformer.h3_text_encoder import truncate_text_encoder

    validate_checkpoint(model_root)
    root = str(Path(model_root).resolve())
    pipe = MiniMaxH3ImageModularPipeline.from_pretrained(root, local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=root, local_files_only=True)
    truncate_text_encoder(pipe, num_keep=50)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    low_cpu_mem_usage = not getattr(options, "image_pinned_cpu_offload", False)
    _configure_dit(pipe, options, root=root, device=device)
    _configure_text_encoder(pipe, options, device=device, world=world, low_cpu_mem_usage=low_cpu_mem_usage)
    _place_dit(pipe.transformer, options, device=device, low_cpu_mem_usage=low_cpu_mem_usage)

    # The upstream VAE keeps its required FP32 operations and latent normalization.
    pipe.vae.to(device=device)
    vae_parallel = getattr(options, "image_vae_parallel", False) and world > 1
    vae_decode_batch = getattr(options, "image_vae_decode_batch_size", 1)
    if vae_parallel or vae_decode_batch > 1:
        from ..vae.minimax_h3_image import install_image_vae_parallel
        install_image_vae_parallel(
            pipe.vae,
            dist.group.WORLD if vae_parallel else None,
            gather_once=getattr(options, "image_vae_gather_once", False),
            decode_batch_size=vae_decode_batch,
        )
    if getattr(options, "image_rgb8_output", False):
        install_image_rgb8_output(pipe)
    return pipe


@torch.inference_mode()
def generate_images(pipe, *, prompt, width, height, min_generation_area, generator, references=()):
    """Run an in-memory image request with condition caches scoped to this call."""
    from ..transformer.h3_image_optimizations import clear_image_condition_cache

    kwargs = dict(prompt=prompt, width=width, height=height,
                  min_generation_area=min_generation_area, output="images", generator=generator)
    if references:
        kwargs["image"] = references
    clear_image_condition_cache(pipe.transformer)
    try:
        return pipe(**kwargs)
    finally:
        clear_image_condition_cache(pipe.transformer)
