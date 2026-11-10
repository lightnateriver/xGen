import argparse


def parse_args(namespace=None, mode="infer", *, bootstrap_environment=True):
    parser = argparse.ArgumentParser(description="AscendX-Video inference script")

    parser = add_network_args(parser)
    parser = add_inference_args(parser)
    parser = add_wan_vace_args(parser)
    parser = add_minimax_h3_args(parser)
    from .h3_refiner_config import add_arguments
    parser = add_arguments(parser)
    from .h3_attention_config import add_arguments as add_h3_attention_arguments
    parser = add_h3_attention_arguments(parser)
    parser = add_abilities_args(parser)
    if mode == "benchmark":
        parser = add_benchmark_args(parser)

    from .h3_features import add_aliases
    add_aliases(parser)
    args = parser.parse_args(namespace=namespace)
    if bootstrap_environment and (getattr(args, 'h3_cuda_preset', None) not in (None, 'none')
            or (args.model == 'MiniMax-H3' and args.h3_cuda_precision == 'auto')):
        from x_base.device import device_type
        from .h3_cuda_options import select_default_recipe, resolve_options, commit_environment
        backend = device_type()
        select_default_recipe(args, backend)
        if getattr(args, 'h3_cuda_preset', None) not in (None, 'none'):
            resolve_options(args, backend)
            commit_environment(args)  # Allocator/NCCL options precede init_env().

    return args


def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="AscendX-Video network args")
    group.add_argument("--model", type=str, default="Wan2.2-T2V-A14B",
                       help="Model for inference.")
    group.add_argument("--pretrained_model_name_or_path", type=str, default="Wan-AI/Wan2.2-T2V-14B-Diffusers",
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    group.add_argument("--task_type", type=str, default="t2v", choices=["t2v", "i2v", "t2i", "t2va", "fl2va", "ref2va"],
                       help="Whether to open i2v mode.")
    group.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")
    group.add_argument(
        "--prompt", type=str, default="A cat and a dog baking a cake together",
        help="The text prompt for video generation."
    )
    group.add_argument("--negative_prompt", type=str, default="Briqht tones, overexposed, static",
                       help="The negative text prompt for video generation.")
    group.add_argument("--height", type=int, default=480, help="The resolution for the generated video height.")
    group.add_argument("--width", type=int, default=832, help="The resolution for the generated video width.")
    group.add_argument("--frames", type=int, default=81, help="The length of the generated video.")
    group.add_argument("--num_inference_steps", type=int, default=40, help="The number of steps for sampling.")
    group.add_argument("--true_cfg_scale", type=float, default=4.0, help="Guidance scale as defined in [Classifier-Free Diffusion Guidance]Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`")
    group.add_argument("--guidance_scale", type=float, default=None, help="The scale factor for the guidance term.")
    group.add_argument("--guidance_scale_2", type=float, default=None,
                       help="The scale factor for the guidance_scale_2 term.")
    group.add_argument("--flow_shift", type=float, default=3.0, help="Shift factor for flow matching schedulers.")
    group.add_argument("--seed", type=int, default=42,
                       help="The random seed for generating video, if None, we init a random seed.")
    group.add_argument("--save_fps", type=int, default=16, help="The frame of per sec for export videos.")
    group.add_argument("--save_path", type=str, default="./output.mp4",
                       help="Path to save the generated video.")
    group.add_argument("--i2v_image_path", type=str, default="",
                       help="The reference image for video generation")
    group.add_argument("--i2v_image_preprocess", type=str, default="wan", choices=["wan", "raw"],
                       help="Whether to resize the original image and change the video resolution accordingly.")
    group.add_argument("--batch_test_input_dir", type=str, default="", help="Directory containing input data for batch testing")
    group.add_argument("--batch_test_mode", type=str, default="normal", choices=["normal", "speed", "accuracy"], help="Test mode: normal (regular batch), speed (speed benchmarking), accuracy (accuracy evaluation)")
    group.add_argument("--batch_test_resolutions", type=str, default="", help="List of resolutions for batch testing, format like '832x480,1280x720'")
    group.add_argument("--batch_test_frames", type=str, default="", help="List of frame counts for batch testing, format like '81,49,17'")
    group.add_argument("--batch_test_targets", type=str, default="", help="List of acceleration targets for batch testing, format like 'none,encoder,decoder,encoder_and_decoder'")
    group.add_argument("--batch_test_sample_size", type=int, default=0, help="Number of videos randomly sampled for testing; 0 means testing the full dataset")
    group.add_argument("--batch_test_report_dir", type=str, default="", help="Directory where the test reports will be saved")
    return parser


def add_wan_vace_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Wan2.1 VACE args")
    group.add_argument("--vace_task", type=str, default='',
                       choices=["t2v", "i2v", "v2lf", "flf2v", "random2v", "inpaint", "outpaint", "openpose", "iwri"],
                       help="The vace task to run.")
    group.add_argument("--image_path", type=str, default="", help="The image for vace video generation")
    group.add_argument("--video_path", type=str, default=None,
                       help="The path of the videos to be processed, separated by commas if there are multiple.")
    group.add_argument("--first_frame_path", type=str, default="",
                       help="The first reference frame for video generation.")
    group.add_argument("--last_frame_path", type=str, default="", help="The last reference frame for video generation.")
    group.add_argument("--image_path_list", nargs='+', default=None, help="The image path list for the random2v task.")
    group.add_argument("--frame_indices", type=int, nargs='+', default=[0, 45, 70],
                       help="Random indices of some frames for the random2v task.")
    group.add_argument("--directions", type=str, nargs='+', default=["left", "right"],
                       help="The directions for the outpaint task.")
    group.add_argument("--expand_ratio", type=float, default=0.25, help="The expand ratio for the outpaint task.")
    return parser


def add_minimax_h3_args(parser: argparse.ArgumentParser):
    from .h3_cuda_options import add_arguments as add_cuda_options
    add_cuda_options(parser)
    from .h3_parallel_config import add_arguments
    add_arguments(parser)
    group = parser.add_argument_group(title="MiniMax-H3 args")
    group.add_argument("--h3_cuda_precision", choices=["auto", "none", "fp8-bf16", "fp8-sage"], default="auto",
                       help="CUDA H3 Base defaults to FP8 Linear/BF16 attention and communication; fp8-sage selects CUDA recipe 2 (visual quality review pending). Auto preserves NPU, other models and explicit legacy recipes. Use none for the legacy CUDA path.")
    group.add_argument("--h3_cuda_linear_tp", type=int, choices=(1, 2), default=1,
                       help="Experimental CUDA H3: TP2 linear projections with unchanged SP8 attention; BF16 pair communication")
    group.add_argument("--h3_cuda_local_output", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA FP8 SP8 memory option: project each sequence shard before gathering FP32 predictions. Default keeps the original full-hidden gather. TP+CP already uses local output.")
    group.add_argument("--h3_cuda_conditioning_offload", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA FP8 Ref2VA memory option: stage DiT on CPU during text/vision conditioning, then restore unchanged weights before DiT. Default off; incurs D2H/H2D transfers.")
    group.add_argument("--h3_cuda_refiner_ff_chunk_size", type=int, default=0,
                       help="Experimental CUDA FP8 Ref2VA input-refiner FFN token chunk size (128..32768, multiple of 128). Zero preserves the original FFN; 8192 bounds temporaries without changing attention or weights.")
    group.add_argument("--h3_cuda_input_refiner", choices=["none", "cudnn-sp", "cudnn-sp-fp8"], default=None,
                       help="Experimental CUDA FP8 Ref2VA SP8 input Token Refiner: BF16 cuDNN attention and Ulysses; optional FP8 FFN. Default preserves the replicated original refiner. Does not alter main DiT attention or communication.")
    group.add_argument("--h3_cuda_precompute_conditioning", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA FP8 Ref2VA SP8: reuse context projection and Token Refiner output within one managed request; release before VAE and on errors. Default off, no cross-request cache.")
    group.add_argument("--h3_cuda_vision_sp", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA Ref2VA SP8 BF16 Qwen3-VL visual blocks; preserves image boundaries and DeepStack outputs. Default off.")
    group.add_argument("--h3_cuda_vae_encoder_parallel", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA Ref2VA SP8 native spatial encoder tile distribution. Default off; encoder weights staged per encode call.")
    group.add_argument("--h3_cuda_audio_encoder_parallel", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA Ref2VA SP8 FP32 audio convolution partition with full native causal projection. Default off.")
    group.add_argument("--h3_cuda_audio_decoder_parallel", choices=("none", "long"), default=None,
                       help="Experimental CUDA Ref2VA SP8 FP32 audio decode: long enables validated 10/15 second latent shapes; other shapes retain original decode.")
    group.add_argument("--h3_cuda_reference_offload", action=argparse.BooleanOptionalAction, default=None,
                       help="Experimental CUDA FP8 Ref2VA: keep DiT on CPU through reference VAE encoding, restore before first DiT. Requires conditioning_offload; default keeps its original restore-after-Qwen boundary.")
    group.add_argument("--h3_turbo8", action="store_true",
                       help="Experimental LightX2V 768p preset with prepared Base assets")
    group.add_argument("--h3_turbo_sla_lora", type=str, default=None,
                       help="Native LightX2V FL2VA 768p Turbo-SLA safetensors. Requires 4 steps, BF16, "
                            "atten=none and full-KV Ulysses. Uses a portable BF16 sparse reference backend.")
    group.add_argument("--keyframe_image", type=str, default="",
                       help="First keyframe image path for the fl2va task.")
    group.add_argument("--keyframe_last_image", type=str, default="",
                       help="Last keyframe image path for the fl2va_last_frame task.")
    group.add_argument("--reference_paths", nargs='+', default=None,
                       help="Reference media paths for the ref2va task (images, videos, or audio files).")
    group.add_argument("--h3_reference_image_short_edge", type=int, default=None,
                       help="Ref2VA Base image short edge: 32..2048, multiple of 32; default preserves 2048")
    group.add_argument("--audio_save_path", type=str, default="",
                       help="Path to save the generated audio. If empty, derived from --save_path.")
    group.add_argument("--rope_table_cache", action="store_true", default=False,
                       help="Enable RoPE table cache: reuse cos/sin frequency tables "
                            "across denoise steps when position_ids are unchanged. "
                            "Only effective for MiniMax-H3 model.")
    return parser


def add_benchmark_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Benchmarking args")
    group.add_argument("--config", type=str, required=True)
    return parser


def add_abilities_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference abilities")

    group.add_argument("--sp", type=int, default=1, help="Sequence parallel number.")
    group.add_argument("--ulysses_degree", type=int, default=None, help="Ulysses sequence parallel number.")
    group.add_argument("--ring_degree", type=int, default=None, help="Ring sequence parallel number.")
    group.add_argument("--phaa_num", type=int, default=0, help="PHAA number.")
    group.add_argument("--turbo_mode", type=str, default="default", choices=["default", "faiz", "next_faiz"],
                       help="Use different turbo mode inference.")
    group.add_argument("--vae_lightning", type=str, default="none",
                   choices=["none", "decoder", "encoder", "encoder_and_decoder"],
                   help="Non-H3 Light VAE target: none/decoder/encoder/encoder_and_decoder. MiniMax-H3 uses --h3_vae_backend lightning instead.")
    group.add_argument("--vae_output_dir", type=str, default="", help="Codec output result directory name. If not specified, the results will not be saved. Currently this only applies to the encoder.")
    group.add_argument("--h3_vae_backend", choices=["legacy", "lightning", "native-stream"], default=None,
                       help="H3 decode backend: recommended CUDA FP8 profiles default to native-stream; other offline paths default to legacy. Explicit selection overrides the recipe.")
    group.add_argument("--h3_vae_model_path", default=None, help="Native MiniMax-H3 video_vae directory for the native-stream decoder.")
    group.add_argument("--h3_vae_trust_remote_code", action="store_true", help="Trust Python code in the selected native H3 VAE checkpoint.")
    group.add_argument("--vae_pad_mode", type=str, default="all_sides_valid", choices=["none", "tail_only", "all_sides", "all_sides_valid"], help="Padding strategy when VAE parallel acceleration is enabled.")
    group.add_argument("--vae_pad_content", type=str, default="constant", choices=["constant", "reflect", "replicate"], help="The padding value when VAE parallel acceleration is enabled and `vae_pad_mode = all_sides`.")
    group.add_argument("--vae_pad_latent_size", type=int, default=8, help="Latent space padding size when VAE parallel acceleration is enabled.")
    group.add_argument("--vae_use_blend", action="store_true", default=False, help="Whether to perform linear blending on overlapping boundaries when VAE parallel acceleration is enabled.")
    group.add_argument("--enable_vae_time_count", action="store_true", default=False, help="Whether to run repeatedly and calculate the average runtime of the VAE Encoder/Decoder. Enabling this option will make the end-to-end generation time inaccurate.")

    group.add_argument("--frame_interpolation", action="store_true", default=False, help="Frame interpolation without using sr model.")
    group.add_argument("--frame_interpolation_sr", action="store_true", default=False,
                       help="Frame interpolation using sr model.")
    group.add_argument("--frame_model_path", type=str, default='./weights/IFRNet_S_Vimeo90K.pth', help="Frame interpolation Model Path.")
    group.add_argument("--x", action="store_true", default=False, help="Use x model")
    group.add_argument("--x_model_path", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x",
                       help="Path to x model")
    group.add_argument("--x_model_path_2", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x_2",
                       help="Path to x model")
    group.add_argument("--joint", action="store_true", default=False, help="Use joint inference")
    group.add_argument("--ten_second", action="store_true", default=False, help="Generate ten second video")
    group.add_argument("--ten_second_model_id_t2v", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                       help="Wan2.2_t2v_path")
    group.add_argument("--ten_second_model_path", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x",
                       help="Wan2.2_t2v_distill_path")
    group.add_argument("--ten_second_model_path_2", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x_2",
                       help="Wan2.2_t2v_distill_path_2")
    group.add_argument("--pusa_lora", type=str, default="Wan-AI/pytorch_lora_weights_high.safetensors",
                       help="pusa_lora")
    group.add_argument("--pusa_lora2", type=str, default="Wan-AI/pytorch_lora_weights_low.safetensors",
                       help="pusa_lora2")
    group.add_argument("--joint_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers_joint",
                       help="Path to joint model")
    parser.add_argument('--fsdp',
                        type=str,
                        nargs='?',
                        const='all',
                        choices=['all', 'text_encoder', 'transformer'],
                        help='Enable Fully Sharded Data Parallel (FSDP). '
                             'The default is the all model, with options for text_encoder and transformer models')
    group.add_argument("--atten",
                       choices=['none', 'sage', 'laser', 'rainfusion', 'sparse', 'mxfp4', 'sparge'],
                       default='none',
                       help="Attention type: none (base), sage (A8W8), laser, rainfusion, sparse, mxfp4")
    group.add_argument("--rainfusion_ratio", default=0.0, type=float,
                       help="The ratio of skip timesteps of rainfusion attention."
                            "If num_inference_steps is 40 and rainfusion_ratio is 0.375, the skip_timesteps is 15.")
    group.add_argument("--rainfusion_sparsity", default=0.8, type=float,
                       help="Sparsity ratio for rainfusion rf_v2 attention. Higher = more blocks dropped = faster. Range [0, 1).")
    group.add_argument("--ada_sparsity", default=0.7, type=float,
                       help="Proportion of invalid/uncomputed elements in the attention matrix")
    group.add_argument("--matmul_quant",
                       choices=['none', 'int8', 'int4', 'mxfp4'],
                       default='none',
                       help="Matmul quantization type: int8, int4, or mxfp4")
    group.add_argument("--conv3d_w8a8", action="store_true", default=False,
                       help="Enable conv3d quantization (A8W8)")
    group.add_argument("--vae_matmul_quant", type=str, default='none',
                       choices=['none', 'int8', 'int4', 'mxfp4'],
                       help="VAE decoder Linear quantization dtype (default: none)")
    group.add_argument("--rope_fused", action="store_true", default=False,
                       help="Enable fused rope operator")
    group.add_argument("--adaln_cache_file", type=str, default=None,
                       help="Path to precomputed AdaLN cache file (safetensors). "
                            "Enables AdaLN schedule cache: loads all tables at init time, "
                            "skips adaln_proj computation during inference. Cache miss = error.")
    group.add_argument("--adaln_offload", action="store_true", default=False,
                       help="NOT YET IMPLEMENTED — AdaLN weight offload (no effect, kept for future use)")
    group.add_argument("--transformer_path", type=str, default=None,
                       help="Path to a custom transformer checkpoint directory (containing "
                            "safetensors index file). When set, transformer weights are loaded "
                            "from this directory instead of the default pretrained_model_name_or_path/transformer/. "
                            "Can be used alone (full checkpoint, adaln computed on-the-fly) or together "
                            "with --adaln_cache_file (pre-stripped checkpoint + cache for memory optimization).")
    group.add_argument("--h3_nvfp4_checkpoint", type=str, default=None,
                       help="Experimental CUDA FL2VA path to a FastVideo pruned NVFP4 safetensors file.")
    group.add_argument("--h3_nvfp4_resident", action="store_true",
                       help="Keep the packed CUDA H3 DiT on device instead of streaming blocks from CPU.")
    group.add_argument("--h3_nvfp4_compile", action="store_true",
                       help="Experimentally compile repeated resident CUDA H3 DiT blocks.")
    group.add_argument("--h3_cuda_vae_sp", action="store_true",
                       help="Experimentally use existing H3 VAE sequence parallel decode on CUDA.")
    group.add_argument("--lora_path_list", nargs='+', default=None,
                       help="List of path to load lora weights.")
    group.add_argument("--lora_scale_weight_list", nargs='+', default=None,
                       help="List of lora scale weights.")
    group.add_argument("--lora_transformer_list", nargs='+', default=None, type=int, choices=[0, 1, 2],
                       help="The list of which transformer need load lora, 0 means all transformer, 1 means transformer_1, 2 means transformer_2.")
    group.add_argument("--fuse_lora", action="store_true", default=False,
                       help="Fuse lora weights.")
    group.add_argument("--server", action="store_true", default=False, help="Use API server")
    group.add_argument("--inf_vram_blocks_num", type=int, default=0,
                       help="if > 0, enable Inifinity VRAM and sets the number of blocks to keep in VRAM. 0 disables offloading")
    group.add_argument("--seedvr2_model_dir", type=str, default="../weights/SeedVR2")
    group.add_argument("--seedvr2_model_name", type=str, default="seedvr2_ema_7b_fp16.safetensors")
    group.add_argument("--adopt_sr", action="store_true", default=False, help="if adopt super-resolution")
    group.add_argument("--resolution", type=int, default=1080, choices=[1080, 720],
                       help="generate video with specified resolution.")
    group.add_argument("--ada_brighten", action="store_true", default=False,
                       help="Adaptively adjusts and restores input image brightness for video generation")
    group.add_argument("--save_memory", action="store_true", default=False,
                       help="apply save memory for offloadmanager")
    group.add_argument("--cfg_parallel_size", default=1, type=int, choices=[1, 2],
                       help="Number of cfg parallel, 1 means not use cfg parallel",
                       )
    group.add_argument("--text_encoder_tp", type=int, default=1,
                       help="Tensor parallel size for text encoder (e.g. Qwen3-VL). "
                            "Splits text encoder weights across NPU cards. "
                            "Conflicts with --fsdp text_encoder/all.",
                       )
    group.add_argument("--cache_dit", action="store_true", default=False,
                       help="use cache_dit",
                       )
    return parser
