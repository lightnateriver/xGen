"""Run fixed-six-step H3 Image T2I/edit with xGen SP4 and text TP4."""
import argparse
import os


def main():
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
    parser.add_argument("--image_swiglu", choices=("none", "native", "triton"), default="none")
    for name in ("image_adaln_table", "image_resident", "image_int8", "image_fused_qkv", "image_condition_cache", "image_vae_parallel", "image_vision_sp", "image_refiner_sp", "image_fused_rope", "image_rgb8_output", "image_local_input"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")

    import torch
    import torch.distributed as dist
    from x_base.device import device_type
    from x_diffusers.adaptor.h3_image import run_image

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
