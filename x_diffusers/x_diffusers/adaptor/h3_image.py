"""H3-Image file runner and option adapter for the native pipeline."""
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from x_base.device import device_type

from ..framework.pipeline import minimax_h3_image


def resolve_image_text_options(args, backend):
    """Default Image decoder W8A8 on NPU, preserving explicit precision choices."""
    if getattr(args, "image_text_int8", None) is None:
        args.image_text_int8 = backend == "npu"
    if args.image_text_int8 and backend != "npu":
        raise ValueError("Image text INT8 requires NPU")


def _validate_request_options(args, world_size):
    dit_tp = getattr(args, "image_dit_tp", 1)
    if world_size not in (1, 2, 4) or args.sp * dit_tp != world_size or (args.sp > 1 and dit_tp > 1):
        raise ValueError("Use pure SP1/SP2/SP4 or pure DiT TP2/TP4")
    if args.text_encoder_tp not in (1, world_size):
        raise ValueError("text_encoder_tp must be 1 or the distributed world size")
    text_int8 = getattr(args, "image_text_int8", False)
    text_resident = getattr(args, "image_text_resident", False)
    if (text_int8 or text_resident) and args.text_encoder_tp != world_size:
        raise ValueError("Image text quantization/residency requires text_encoder_tp equal to world size")
    if args.num_inference_steps != 6:
        raise ValueError("MiniMax-H3-Image has exactly six PDD sampling steps")
    if getattr(args, "image_vae_decode_batch_size", 1) < 1:
        raise ValueError("VAE decode batch size must be at least one")
    if getattr(args, "image_vae_gather_once", False) and (world_size == 1 or not getattr(args, "image_vae_parallel", False)):
        raise ValueError("--image_vae_gather_once requires --image_vae_parallel on multiple devices")
    if getattr(args, "image_adaln_table_file", None) and getattr(args, "image_adaln_table_save", None):
        raise ValueError("Choose either reading or saving an AdaLN table")
    if args.width <= 0 or args.height <= 0 or args.width * args.height > 20 * 1024**2:
        raise ValueError("Image dimensions must be positive and at most 20 megapixels")
    if len(args.image_path_list or []) > 16:
        raise ValueError("MiniMax-H3-Image supports at most 16 ordered reference images")
    if not 1024 <= args.min_generation_area <= 20 * 1024**2:
        raise ValueError("min_generation_area must be between 1024 pixels and 20 megapixels")
    for path in args.image_path_list or []:
        if not Path(path).is_file():
            raise ValueError(f"Reference image does not exist: {path}")


def validate_options(args, world_size):
    _validate_request_options(args, world_size)
    minimax_h3_image.validate_checkpoint(args.pretrained_model_name_or_path)


def load_pipeline(args):
    """Validate the adapter options and assemble a reusable H3-Image pipeline."""
    backend = device_type()
    resolve_image_text_options(args, backend)
    world = dist.get_world_size() if dist.is_initialized() else 1
    validate_options(args, world)
    device = torch.device(backend, torch.get_device_module(backend).current_device())
    return minimax_h3_image.load_pipeline(args.pretrained_model_name_or_path, args, device=device, world=world)


@torch.inference_mode()
def run_image(args, pipe=None):
    from PIL import Image
    from x_base import attention_manager

    backend = device_type()
    resolve_image_text_options(args, backend)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    validate_options(args, world)
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
    records = []
    for i in range(args.warmup + args.repeat):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        device_module.synchronize()
        begin = time.perf_counter()
        images = minimax_h3_image.generate_images(
            pipe, prompt=args.prompt, width=args.width, height=args.height,
            min_generation_area=args.min_generation_area, generator=generator, references=references,
        )
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
                      "image_ff_chunk", "image_swiglu", "image_fused_rope", "image_laser_dense_tail", "image_rgb8_output",
                      "image_local_input", "image_vae_gather_once", "image_vae_decode_batch_size",
                      "image_adaln_table_file", "image_adaln_table_save", "image_pinned_cpu_offload",
                      "image_text_int8", "image_text_resident")},
                  peak_allocated_bytes=peak.item())
    if rank == 0:
        Path(args.save_path).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return result
