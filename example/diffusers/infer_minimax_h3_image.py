"""Run fixed-six-step H3 Image T2I/edit with xGen SP4 and text TP4."""
import argparse
import os
import json
from pathlib import Path
import time


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image_path_list", nargs="*", default=[])
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--min_generation_area", type=int, default=4194304)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp", type=int, default=4)
    parser.add_argument("--text_encoder_tp", type=int, default=4)
    parser.add_argument("--atten", choices=("none", "laser"), default="none")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--image_dit_tp", type=int, default=1)
    parser.add_argument("--image_ff_chunk", type=int, default=0)
    parser.add_argument("--image_laser_dense_tail", type=int, default=0)
    parser.add_argument("--image_vae_decode_batch_size", type=int, default=1,
                        help="Maximum VAE decoder tiles per device per call")
    parser.add_argument("--image_vae_gather_once", action="store_true",
                        help="Gather all VAE tiles at once; requires --image_vae_parallel and multiple devices")
    parser.add_argument("--image_adaln_table_file")
    parser.add_argument("--image_adaln_table_save")
    parser.add_argument("--image_pinned_cpu_offload", action="store_true")
    parser.add_argument("--image_swiglu", choices=("none", "native", "triton"), default="none")
    parser.add_argument("--image_text_int8", action=argparse.BooleanOptionalAction, default=None,
                        help="Qwen decoder W8A8: defaults on for NPU and off for CUDA; --no-image_text_int8 uses BF16")
    for name in ("image_adaln_table", "image_resident", "image_int8", "image_fused_qkv", "image_condition_cache", "image_vae_parallel", "image_vision_sp", "image_refiner_sp", "image_fused_rope", "image_rgb8_output", "image_local_input", "image_text_resident"):
        parser.add_argument("--" + name, action="store_true")
    return parser


def run_image(args, pipe=None):
    """Run the file-based CLI workflow, including warmup, timing and outputs."""
    from x_base.device import device_type
    from x_diffusers.adaptor.h3_runtime import generate_h3_images
    from x_diffusers.adaptor.validation import resolve_image_text_options
    from x_diffusers.adaptor.validation import validate_h3_image_request
    import torch
    import torch.distributed as dist
    with torch.inference_mode():
        from PIL import Image

        backend = device_type()
        resolve_image_text_options(args, backend)
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        validate_h3_image_request(args, world)
        device_module = torch.get_device_module(backend)
        from x_diffusers.adaptor.load_pipe import configure_h3_image_attention

        configure_h3_image_attention(args, backend)
        start = time.perf_counter()
        pipeline_reused = pipe is not None
        if pipe is None:
            from x_diffusers.adaptor.load_pipe import load_pipe

            pipe = load_pipe(args, model="MiniMax-H3-Image")
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
            images = generate_h3_images(
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")

    import torch
    import torch.distributed as dist
    from x_base.device import device_type

    backend = device_type()
    module = torch.get_device_module(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    module.set_device(local_rank)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if backend == "npu":
        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="hccl" if backend == "npu" else "nccl")
    try:
        run_image(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
