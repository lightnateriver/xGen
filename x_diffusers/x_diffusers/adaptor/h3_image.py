"""MiniMax-H3-Image loading and inference, separate from audio/video recipes."""
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from x_base.device import device_type


def validate_options(args, world_size):
    dit_tp = getattr(args, "image_dit_tp", 1)
    if world_size not in (1, 2, 4) or args.sp * dit_tp != world_size or (args.sp > 1 and dit_tp > 1):
        raise ValueError("Use pure SP1/SP2/SP4 or pure DiT TP2/TP4")
    if args.text_encoder_tp not in (1, world_size):
        raise ValueError("text_encoder_tp must be 1 or the distributed world size")
    if args.num_inference_steps != 6:
        raise ValueError("MiniMax-H3-Image has exactly six PDD sampling steps")
    if args.width <= 0 or args.height <= 0 or args.width * args.height > 20 * 1024**2:
        raise ValueError("Image dimensions must be positive and at most 20 megapixels")
    if len(args.image_path_list or []) > 16:
        raise ValueError("MiniMax-H3-Image supports at most 16 ordered reference images")
    if not 1024 <= args.min_generation_area <= 20 * 1024**2:
        raise ValueError("min_generation_area must be between 1024 pixels and 20 megapixels")
    for path in args.image_path_list or []:
        if not Path(path).is_file():
            raise ValueError(f"Reference image does not exist: {path}")
    root = Path(args.pretrained_model_name_or_path)
    for name in ("modular_model_index.json", "transformer_diffusers/config.json", "vae_diffusers/config.json",
                 "text_encoder/config.json", "pdd_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Missing local H3 Image component: {root / name}")
    pdd = json.loads((root / "pdd_config.json").read_text())
    if pdd.get("num_inference_steps") != 6 or len(pdd.get("sigmas", [])) != 7:
        raise ValueError("Expected the six-step H3 Image PDD checkpoint")


@torch.inference_mode(False)
@torch.no_grad()
def load_pipeline(args):
    try:
        from diffusers import MiniMaxH3ImageModularPipeline
    except ImportError as error:
        raise ImportError("H3 Image needs the pinned optional Diffusers source in requirements-h3-image.txt") from error
    from diffusers.hooks import apply_group_offloading
    from x_base import ParallelManager, enable_tp
    from ..framework.transformer.minimax_h3_image import enable_h3_image_sp
    from .load_pipe import _truncate_text_encoder

    backend = device_type()
    device = torch.device(backend, torch.get_device_module(backend).current_device())
    world = dist.get_world_size() if dist.is_initialized() else 1
    validate_options(args, world)
    root = str(Path(args.pretrained_model_name_or_path).resolve())
    pipe = MiniMaxH3ImageModularPipeline.from_pretrained(root, local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=root, local_files_only=True)
    _truncate_text_encoder(pipe, num_keep=50)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    from ..framework.transformer.h3_image_optimizations import (
        install_image_adaln_table, install_image_condition_cache, quantize_image_blocks, fuse_image_qkv)
    if getattr(args, "image_adaln_table", False):
        install_image_adaln_table(pipe.transformer, pipe.config.sample_sigmas, pipe.reference_noise_aug, device)
    if getattr(args, "image_fused_qkv", False):
        fuse_image_qkv(pipe.transformer)
    if getattr(args, "image_int8", False):
        if backend != "npu":
            raise ValueError("Image INT8 requires NPU")
        quantize_image_blocks(pipe.transformer)
    enable_h3_image_sp(pipe.transformer, args.sp, getattr(args, "image_dit_tp", 1))
    if getattr(args, "image_dit_tp", 1) > 1:
        from ..framework.transformer.h3_image_optimizations import install_image_dit_tp
        install_image_dit_tp(pipe.transformer, dist.group.WORLD)
    if getattr(args, "image_refiner_sp", False) and args.sp > 1:
        from ..framework.transformer.h3_image_optimizations import install_image_refiner_sp
        install_image_refiner_sp(pipe.transformer)
    if getattr(args, "image_fused_rope", False):
        for block in pipe.transformer.transformer_blocks:
            block.attn._xgen_image_fused_rope = True
    dense_tail = getattr(args, "image_laser_dense_tail", 0)
    if dense_tail:
        for block in pipe.transformer.transformer_blocks[-dense_tail:]:
            block.attn._xgen_image_dense_attention = True
    if getattr(args, "image_ff_chunk", 0):
        from ..framework.transformer.h3_image_optimizations import install_image_ff_chunk
        install_image_ff_chunk(pipe.transformer, args.image_ff_chunk)
    if getattr(args, "image_swiglu", "none") != "none":
        from ..framework.transformer.h3_image_optimizations import install_image_swiglu
        install_image_swiglu(pipe.transformer, args.image_swiglu)
    if args.text_encoder_tp > 1:
        pipe._xgen_image_tp_manager = ParallelManager(1, 1, 1, tp_size=args.text_encoder_tp)
        enable_tp(pipe.text_encoder, args.text_encoder_tp, target_layers=range(50))
        pipe.text_encoder.to(device)
    else:
        apply_group_offloading(pipe.text_encoder, onload_device=device, offload_device=torch.device("cpu"),
                               offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True)
    if getattr(args, "image_vision_sp", False) and world > 1:
        from .h3_vision_sp import install_vision_sp
        install_vision_sp(pipe.text_encoder.model.visual, lambda: dist.group.WORLD)
    # A BF16 DiT alone exceeds a 64 GiB 910B; stage one block at a time.
    if getattr(args, "image_resident", False):
        pipe.transformer.to(device)
    else:
        pipe.transformer.enable_group_offload(
            onload_device=device, offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True)
    if getattr(args, "image_condition_cache", False):
        install_image_condition_cache(pipe.transformer)
    if getattr(args, "image_local_input", False):
        from ..framework.transformer.h3_image_optimizations import install_image_local_input
        install_image_local_input(pipe.transformer)
    # The upstream VAE keeps its required FP32 operations and latent normalization.
    pipe.vae.to(device=device)
    if getattr(args, "image_vae_parallel", False) and world > 1:
        from ..framework.transformer.h3_image_optimizations import install_image_vae_parallel
        install_image_vae_parallel(pipe.vae, dist.group.WORLD)
    if getattr(args, "image_rgb8_output", False):
        from ..framework.transformer.h3_image_optimizations import install_image_rgb8_output
        install_image_rgb8_output(pipe)
    return pipe


@torch.inference_mode()
def run_image(args, pipe=None):
    from PIL import Image
    from x_base import attention_manager

    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    validate_options(args, world)
    backend = device_type()
    device_module = torch.get_device_module(backend)
    if args.atten == "laser":
        if backend != "npu":
            raise ValueError("Laser attention requires an Ascend NPU")
        from x_base.operator.attention import LA_AVAILABLE
        if not LA_AVAILABLE:
            raise RuntimeError("Laser attention was requested but MindIE-SD is unavailable")
        attention_manager.enable_laser_attention()
    else:
        attention_manager.disable_laser_attention()
    start = time.perf_counter()
    pipeline_reused = pipe is not None
    if pipe is None:
        pipe = load_pipeline(args)
    device_module.synchronize()
    initialization_s = time.perf_counter() - start
    references = []
    for path in args.image_path_list or []:
        with Image.open(path) as image:
            references.append(image.convert("RGB"))
    kwargs = dict(prompt=args.prompt, width=args.width, height=args.height,
                  min_generation_area=args.min_generation_area, output="images")
    if references:
        kwargs["image"] = references
    records = []
    for i in range(args.warmup + args.repeat):
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)
        device_module.synchronize()
        begin = time.perf_counter()
        from ..framework.transformer.h3_image_optimizations import clear_image_condition_cache
        clear_image_condition_cache(pipe.transformer)
        try:
            images = pipe(**kwargs)
        finally:
            clear_image_condition_cache(pipe.transformer)
        device_module.synchronize()
        elapsed = torch.tensor(time.perf_counter() - begin, device=backend)
        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if i >= args.warmup:
            output = Path(args.save_path)
            if args.repeat > 1:
                output = output.with_name(f"{output.stem}-{i-args.warmup:02d}{output.suffix}")
            if rank == 0:
                output.parent.mkdir(parents=True, exist_ok=True)
                images[0].save(output)
            records.append(dict(inference_s=elapsed.item(), output=str(output)))
    peak = torch.tensor(device_module.max_memory_allocated(), dtype=torch.int64, device=backend)
    if dist.is_initialized():
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    result = dict(model="MiniMax-H3-Image", workflow="edit" if references else "t2i",
                  width=args.width, height=args.height, seed=args.seed, steps=6,
                  sp=args.sp, text_encoder_tp=args.text_encoder_tp, attention=args.atten,
                  sequence_length=getattr(pipe.transformer, "_xgen_image_sp_state", {}).get("sequence_length"),
                  pipeline_reused=pipeline_reused,
                  initialization_s=initialization_s, warmup=args.warmup, records=records,
                  optimization_options={name: getattr(args, name, None) for name in (
                      "image_adaln_table", "image_resident", "image_int8", "image_dit_tp", "image_fused_qkv",
                      "image_condition_cache", "image_refiner_sp", "image_vision_sp", "image_vae_parallel",
                      "image_ff_chunk", "image_swiglu", "image_fused_rope", "image_laser_dense_tail", "image_rgb8_output", "image_local_input")},
                  peak_allocated_bytes=peak.item())
    if rank == 0:
        Path(args.save_path).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return result
