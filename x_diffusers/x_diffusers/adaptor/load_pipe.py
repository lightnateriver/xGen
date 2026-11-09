import os

import torch
import torch.distributed as dist
import x_diffusers.framework

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanImageToVideoPipeline, WanVACEPipeline, WanTransformer3DModel, \
    CogVideoXPipeline, HunyuanVideoImageToVideoPipeline, HunyuanVideoPipeline
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.models.attention_dispatch import AttentionBackendName
from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel
from diffusers.utils import logging

from x_base import fsdp_init, turbo_on_pipe, cache_dit_on_pipe, attention_manager, matmul_manager, \
    rope_manager, phaa_on_pipe, OffloadManager, OffloadManager_For_Save_Memory, offload_config_manager
from x_base.device import device_type
from x_base.config import QWEN_LORA_SCHEDULER_CONFIG
from .distillation import update_wan_distillation_pipe_params, load_distillation_model
from ..framework.pipeline import WanImageToVideoPipelineJoint, WanVideoToVideoPipeline
from ..framework.schedulers.pusa_schedulers import FlowMatchEulerDiscreteSchedulerPusa
from ..framework.pipeline.registry import get_pipeline_cls_by_hf_class_name, is_registered_mappings

WAN_22_FLAG = "Wan2.2"
VAE_TILING = os.getenv("VAE_TILING", "False") == "True"
logger = logging.get_logger("infer")

try:
    from seedvr.inference import SeedVR2_Pipe

    logger.info("Plugin import seedvr successful!")
except ImportError as e1:
    # 捕获所有导入相关的异常（模块不存在、类不存在等）
    logger.info(f"Plugin import seedvr2 failed: {str(e1)}")

SUPPORTED_MODEL = [
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B",
    "Wan2.1-T2V-1.3B",
    "Wan2.2-T2V-A14B",
    "Wan2.2-I2V-A14B",
    "Wan2.1-VACE-14B",
    "Wan2.1-VACE-1.3B",
    "CogVideoX-5b",
    "HunyuanVideo-T2V-13B",
    "MiniMax-H3",
]


def load_wan_pipe(args):
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32)
    pipe_params = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "vae": vae,
        "torch_dtype": torch.bfloat16
    }
    pipe_params = update_wan_distillation_pipe_params(pipe_params, args)

    logger.info(f"building {args.model}'s pipeline...")
    pipe = None

    if args.joint:
        subfolder_name = "transformer_2" if "Wan2.2" in args.joint_model_path else "transformer"
        print(subfolder_name, flush=True)
        transformer_3 = WanTransformer3DModel.from_pretrained(args.joint_model_path, subfolder=subfolder_name)
        transformer_3.to(dtype=torch.bfloat16)
        pipe_params["transformer_3"] = transformer_3
        pipe = WanImageToVideoPipelineJoint.from_pretrained(**pipe_params)
        pipe.scheduler.config.flow_shift = args.flow_shift
    elif "VACE" in args.model:
        pipe = WanVACEPipeline.from_pretrained(**pipe_params)
    elif "T2V" in args.model or "t2v" == args.task_type or "T2I" in args.model or "t2i" == args.task_type:
        pipe = WanPipeline.from_pretrained(**pipe_params)
    elif "I2V" in args.model or "i2v" == args.task_type:
        pipe = WanImageToVideoPipeline.from_pretrained(**pipe_params)
    else:
        logger.error(f"Unknown task_type: {args.task_type}!")

    if "Wan2.1" in args.model and not args.joint:
        pipe.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=args.flow_shift)

    #if not args.ten_second:
    pipe.to(dtype=torch.bfloat16)
    return pipe


def load_v2v_pipe(args):
    logger.info("building v2v pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32)

    v2v_transformer = load_distillation_model(args.ten_second_model_id_t2v, "transformer", args.model,
                                              args.ten_second_model_path)
    v2v_transformer_2 = load_distillation_model(args.ten_second_model_id_t2v, "transformer_2", args.model,
                                                args.ten_second_model_path_2)
    v2v = WanVideoToVideoPipeline.from_pretrained(
        args.ten_second_model_id_t2v,
        vae=vae,  # shared VAE
        transformer=v2v_transformer,  # isolated
        transformer_2=v2v_transformer_2,
        torch_dtype=torch.bfloat16,
    )

    v2v.load_lora_weights(
        args.pusa_lora,
        adapter_name="lightning",
    )
    v2v.load_lora_weights(
        args.pusa_lora2,
        adapter_name="lightning_2",
        load_into_transformer_2=True,
    )
    v2v.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.5, 1.4])
    noise_scheduler = FlowMatchEulerDiscreteSchedulerPusa(shift=3.0, sigma_min=0.0, extra_one_step=True)
    v2v.scheduler = noise_scheduler

    if args.conv3d_w8a8:
        v2v.vae = matmul_manager.enable_vae_conv3d_quant(v2v.vae)

    return v2v


def load_seedvr2_pipe(args):
    logger.info("building seedvr2 pipeline...")
    pipe = SeedVR2_Pipe(args)
    return pipe


def load_cogvideo_pipe(args):
    logger.info("building cogvideo t2v pipeline...")
    if "T2V" in args.model or "t2v" == args.task_type:
        pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    else:
        raise Exception(f"Unsupported model, supported models are: {SUPPORTED_MODEL}")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def load_hunyuanvideo_pipe(args):
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    if "T2V" in args.model or "t2v" == args.task_type:
        logger.info("building hunyuan t2v pipeline...")
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=torch.float16)
    else:
        logger.info("building hunyuan i2v pipeline...")
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=torch.float16)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def load_minimax_h3_pipe(args):
    """Load a MiniMax-H3 ModularPipeline.

    MiniMax-H3 is integrated as Modular Diffusers blocks only — there is no
    ``DiffusionPipeline`` half.  The ``t2va`` / ``fl2va`` tasks load through
    ``ModularPipeline.from_pretrained``; the ``ref2va`` task loads through
    ``ModularPipeline.from_pretrained(path, workflow="ref2va")``.  Both share one repository.

    Memory strategy: group offloading (block-level streaming between CPU and NPU)
    instead of FSDP, following the official MiniMax-H3 example.
    """
    from diffusers.hooks import apply_group_offloading

    dtype = to_dtype(args.dtype)
    task_type = getattr(args, "task_type", "t2va")
    packed_checkpoint = getattr(args, "h3_nvfp4_checkpoint", None)
    packed_resident = getattr(args, "h3_nvfp4_resident", False)
    packed_compile = getattr(args, "h3_nvfp4_compile", False)
    cuda_fusions = ()
    if device_type() == "cuda":
        from .h3_cuda import get_dit_fusions

        cuda_fusions = get_dit_fusions(packed_resident, packed_compile)
    if packed_compile and not packed_resident:
        raise ValueError("--h3_nvfp4_compile requires --h3_nvfp4_resident")
    if packed_resident and not packed_checkpoint:
        raise ValueError("--h3_nvfp4_resident requires --h3_nvfp4_checkpoint")
    if packed_checkpoint and (
        device_type() != "cuda" or task_type != "fl2va" or dtype != torch.bfloat16
        or args.matmul_quant != "none" or any(getattr(args, name, None) for name in (
            "transformer_path", "adaln_cache_file", "h3_turbo8",
            "h3_turbo_sla_lora", "lora_path_list", "fsdp",
        ))
    ):
        raise ValueError("H3 NVFP4 requires CUDA BF16 FL2VA without extra quantization, LoRA, AdaLN cache or FSDP")

    if task_type == "ref2va":
        logger.info("building minimax-h3 ref2va modular pipeline...")
        pipe = ModularPipeline.from_pretrained(args.pretrained_model_name_or_path, workflow="ref2va")
    elif getattr(args, "h3_refiner", False):
        pipe = ModularPipeline.from_pretrained(args.pretrained_model_name_or_path, workflow=task_type)
    else:
        logger.info("building minimax-h3 t2va/fl2va modular pipeline...")
        pipe = ModularPipeline.from_pretrained(args.pretrained_model_name_or_path)

    # Load transformer from a custom checkpoint directory (full or pre-stripped).
    # from_pretrained is effectively strict=False: missing keys (e.g. adaln in
    # pre-stripped checkpoints) only produce warnings, and with low_cpu_mem_usage
    # they stay on meta device (zero memory).
    transformer_path = getattr(args, 'transformer_path', None)
    if packed_checkpoint:
        from .h3_nvfp4 import load_packed_h3

        config = MiniMaxH3Transformer3DModel.load_config(
            args.pretrained_model_name_or_path, subfolder="transformer"
        )
        logger.info("Loading experimental H3 FL2VA NVFP4 transformer")
        transformer = load_packed_h3(MiniMaxH3Transformer3DModel, config, packed_checkpoint)
        pipe.update_components(transformer=transformer)
        pipe.load_components(dtype=dtype, pretrained_model_name_or_path=args.pretrained_model_name_or_path)
    elif transformer_path:
        logger.info(f"Loading transformer from: {transformer_path}")
        transformer_key = "transformer_ref" if task_type == "ref2va" else "transformer"
        pipe.load_components(
            dtype=dtype,
            pretrained_model_name_or_path={
                transformer_key: transformer_path,
                "default": args.pretrained_model_name_or_path,
            },
            subfolder={
                transformer_key: "",  # transformer_path is the direct path, no subfolder
            },
        )
    else:
        pipe.load_components(dtype=dtype, pretrained_model_name_or_path=args.pretrained_model_name_or_path)

    # origin/main: workflow="ref2va" only loads transformer_ref, pipe.transformer is None.
    # Set alias so operator_ability / pipe_to_device / install_adaln_cache etc. can use
    # pipe.transformer uniformly. Sync after each replacement (L399, L755) keeps them aligned.
    if task_type == "ref2va" and hasattr(pipe, "transformer_ref"):
        pipe.transformer = pipe.transformer_ref

    # AdaLN cache handling — decoupled from transformer_path.
    #
    #   transformer_path + cache_file (Case 3): pre-stripped checkpoint, release
    #       adaln weights (no-op on meta device), install cache. Optimal memory.
    #   cache_file only (Case 4): full checkpoint loaded, adaln weights are real
    #       and waste ~26GB. Do NOT release (would break model). Cache still works
    #       at runtime, but memory optimization is not in effect. Warn the user.
    #   transformer_path only (Case 2): no cache, adaln computed on-the-fly.
    #   neither (Case 1): default path, adaln on-the-fly.
    cache_file = getattr(args, 'adaln_cache_file', None)
    if cache_file:
        if transformer_path:
            # Case 3: pre-stripped checkpoint — release adaln weights (meta device, no-op)
            from x_base.adaln_cache import validate_stripped_checkpoint, release_adaln_weights

            try:
                validate_stripped_checkpoint(pipe.transformer, transformer_path)
            except FileNotFoundError:
                logger.warning(
                    f"transformer_path: index.json not found at {transformer_path}, "
                    f"skipping validation. Ensure the checkpoint is correctly stripped."
                )

            logger.info("Releasing adaln weights (pre-stripped checkpoint mode)...")
            release_adaln_weights(pipe.transformer)
        else:
            # Case 4: full checkpoint — adaln weights loaded but wasted
            logger.warning(
                "--adaln_cache_file used without --transformer_path: "
                "full checkpoint loaded with adaln weights (~26GB wasted). "
                "Cache will work at runtime, but memory optimization is not in effect. "
                "For optimal memory, use --transformer_path with a pre-stripped checkpoint."
            )

    # Truncate text_encoder to first 50 layers — before any NPU transfer.
    # H3 only uses hidden_states[50] (layer 49 output, pre-norm).
    # Layers 50-63 weights never enter NPU.
    from .h3_turbo_sla import configure_pipe
    configure_pipe(pipe, args)
    if getattr(args, "h3_turbo8", False):
        from .h3_turbo8 import configure
        configure(pipe)
    _truncate_text_encoder(pipe, num_keep=50)

    # Freeze models — no gradients needed for inference
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Group offloading: stream blocks between CPU and NPU to avoid OOM
    # low_cpu_mem_usage=True to skip pin_memory() which causes NPU host memory OOM
    backend = device_type()
    onload_device = torch.device(backend, torch.get_device_module(backend).current_device())
    offload = dict(
        onload_device=onload_device,
        offload_device=torch.device("cpu"),
        use_stream=True,
        low_cpu_mem_usage=True,
    )
    # Skip text_encoder group offloading when TP is enabled —
    # enable_tp() in parallelism_ability will replace nn.Linear with
    # ColumnParallelLinear/RowParallelLinear, and group offloading must
    # be applied AFTER that replacement to wrap the new TP layers.
    if getattr(args, "h3_cuda_precision", "none") != "none":
        from .h3_cuda_precision import prepare_transformer

        prepare_transformer(pipe, args, onload_device)
    elif packed_resident:
        logger.info("Keeping packed H3 DiT resident on %s", onload_device)
        pipe.transformer.to(onload_device)
        if cuda_fusions:
            from .h3_cuda import install_dit_fusions

            install_dit_fusions(pipe.transformer, cuda_fusions)
        if packed_compile:
            from x_base.operator.h3_nvfp4 import H3NVFP4Linear

            for block in pipe.transformer.transformer_blocks:
                # Keep stream choreography and packed custom kernels outside Dynamo.
                block.attn.forward = torch.compiler.disable(block.attn.forward)
                for module in block.ff.modules():
                    if isinstance(module, H3NVFP4Linear):
                        module.forward = torch.compiler.disable(module.forward)
                block.compile(dynamic=False, options={"triton.cudagraphs": False})
            logger.info("Regional compilation enabled for %d H3 DiT blocks",
                        len(pipe.transformer.transformer_blocks))
    elif args.matmul_quant == 'none' and not getattr(args, 'h3_refiner', False):
        pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    if args.text_encoder_tp <= 1:
        apply_group_offloading(pipe.text_encoder, offload_type="leaf_level", **offload)

    # VAEs are small enough to stay on device
    if getattr(args, "h3_cuda_precision", "none") != "none":
        from .h3_cuda_precision import prepare_vaes

        prepare_vaes(pipe, args, onload_device)
    else:
        pipe.vae.to(onload_device)
        pipe.audio_vae.to(onload_device)

    if backend == "cuda":
        return pipe

    # VAE decoder to bf16, keep norm weights in fp32
    # NPU aclnnRmsNorm accepts BF16 input + FP32 weight → BF16 output natively,
    # so .float()/.to(dtype) casts were removed from the SP forward path.
    # Everything (Linear, RoPE, etc.) runs in bf16.
    # post_quant_conv stays in FP32 — the latent z from the pipeline is FP32,
    # and autocast handles the dtype mismatch at proj_in (BF16 weight).
    # WeightQuantLinearModule.forward also casts FP32→BF16 as a safety net.
    pipe.vae.decoder.to(torch.bfloat16)
    for module in pipe.vae.decoder.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.RMSNorm)):
            module.to(torch.float32)

    return pipe


def _truncate_text_encoder(pipe, num_keep=50):
    """截断 Qwen3-VL text_encoder 到前 num_keep 层。

    在 load_components() 之后、group offloading 之前调用。
    层 50-63 的权重从不进入 NPU。

    H3 只取 hidden_states[50] (layer 49 输出, pre-norm)，
    后 14 层的权重和计算完全浪费。

    三步:
      1. 删 layers[num_keep:] — ModuleList 只剩 num_keep 层, forward 只跑这些层
      2. norm → Identity — last_hidden_state = layer (num_keep-1) pre-norm 输出
      3. 拦截 output_hidden_states=False — 避免 materialize hidden state tuple
    """
    import torch.nn as nn
    import types

    te = pipe.text_encoder
    # Qwen3-VL structure: text_encoder.model.language_model.layers
    #   Qwen3VLForConditionalGeneration
    #     └── .model = Qwen3VLModel
    #           └── .language_model = Qwen3VLTextModel
    #                 ├── .layers = nn.ModuleList([...])
    #                 ├── .norm = Qwen3VLTextRMSNorm
    #                 └── .rotary_emb
    layers = te.model.language_model.layers
    total_layers = len(layers)

    # 1. 删除后 (total - num_keep) 层 — CPU 内存直接释放，不经过 NPU
    for i in range(total_layers - 1, num_keep - 1, -1):
        del layers[i]

    # 2. final RMSNorm → Identity
    #    H3 要 pre-norm 输出; norm=Identity 后 last_hidden_state = layer (num_keep-1) 输出
    te.model.language_model.norm = nn.Identity()

    # 3. 拦截 output_hidden_states=False，避免 materialize hidden state tuple
    _orig_forward = te.model.forward

    def _patched_forward(self, *args, **kwargs):
        kwargs['output_hidden_states'] = False
        output = _orig_forward(*args, **kwargs)

        class _TruncatedOutput:
            def __init__(self, orig):
                self.last_hidden_state = orig.last_hidden_state
                # hidden_states[num_keep] = last_hidden_state (pre-norm, 因为 norm=Identity)
                self.hidden_states = (None,) * num_keep + (orig.last_hidden_state,)

        return _TruncatedOutput(output)

    te.model.forward = types.MethodType(_patched_forward, te.model)

    # 4. 绕过 get_qwen3vl_prompt_embeds 的 num_layers 验证
    #    forward 遍历 self.layers ModuleList (num_keep 个), 不读 config
    te.config.text_config.num_hidden_layers = num_keep + 1

    logger.info(f"Truncated text_encoder to {num_keep} layers (removed {total_layers - num_keep})")


def to_dtype(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def load_pipe(args):
    backend = device_type()
    from .h3_parallel_config import H3ParallelConfig
    from .h3_cuda_precision import resolve
    resolve(args, backend)
    H3ParallelConfig.from_args(args, backend, dist.get_world_size() if dist.is_initialized() else 1)
    precision = getattr(args, "h3_cuda_precision", "auto")
    if precision != "none":
        from .h3_cuda_precision import resolve, validate

        precision = resolve(args, backend)
        if precision != "none":
            validate(args, backend)
    from .h3_cuda_options import commit_environment
    commit_environment(args)
    if (backend != "cuda" or args.model != "MiniMax-H3") and any((
        getattr(args, "h3_nvfp4_checkpoint", None),
        getattr(args, "h3_nvfp4_resident", False),
        getattr(args, "h3_nvfp4_compile", False),
    )):
        raise ValueError("H3 NVFP4 options require CUDA MiniMax-H3")
    if getattr(args, "h3_cuda_vae_sp", False) and (
        backend != "cuda" or args.model != "MiniMax-H3" or args.sp < 2
    ):
        raise ValueError("--h3_cuda_vae_sp requires CUDA MiniMax-H3 and SP > 1")
    if backend == "cuda":
        if args.text_encoder_tp == 8 and (not dist.is_initialized() or dist.get_world_size() != 8):
            raise ValueError("CUDA H3 text encoder TP8 requires eight distributed ranks")
        unsupported = (
            args.model != "MiniMax-H3", args.atten not in ("none", "sage", "sparge"), args.matmul_quant != "none",
            args.vae_matmul_quant != "none", args.vae_lightning != "none", args.rope_fused,
            args.text_encoder_tp not in (1, 8), args.inf_vram_blocks_num, args.fsdp,
            getattr(args, "h3_attention", "legacy") != "legacy",
        )
        if any(unsupported):
            raise ValueError("CUDA currently supports MiniMax-H3 with SDPA, SageAttention or SpargeAttention, "
                             "group offload, and without NPU-specific optimizations")
    if getattr(args, "h3_turbo8", False):
        from .h3_turbo8 import validate, validate_assets
        validate(args)
        validate_assets(args)
    if getattr(args, 'h3_attention', 'legacy') != 'legacy':
        from .h3_attention_config import validate
        validate(args)
    from .h3_turbo_sla import validate_args
    validate_args(args)
    # 查看是否在自定义pipeline注册
    if is_registered_mappings(args.pretrained_model_name_or_path):
        return get_pipeline_cls_by_hf_class_name(args.pretrained_model_name_or_path, to_dtype(args.dtype))

    # 是否是支持的其他模型
    if args.model not in SUPPORTED_MODEL:
        raise Exception(f"Unsupported model, supported models are: {SUPPORTED_MODEL}")
    if "Wan" in args.model:
        pipe = load_wan_pipe(args)
    elif "CogVideoX" in args.model:
        pipe = load_cogvideo_pipe(args)
    elif "HunyuanVideo" in args.model:
        pipe = load_hunyuanvideo_pipe(args)
    elif "MiniMax-H3" in args.model:
        pipe = load_minimax_h3_pipe(args)
    else:
        pipe = None
    return pipe


def apply_npu_attention(pipe):
    """
    设置自定义 npu attention backend
    """
    for attr_name in dir(pipe):
        if attr_name.startswith("transformer"):
            attr_value = getattr(pipe, attr_name)
            if isinstance(attr_value, torch.nn.Module):
                attr_value.set_attention_backend(AttentionBackendName._NATIVE_NPU)


def _apply_attention_backend(args):
    if device_type() == 'cuda':
        attention_manager.configure_cuda_attention(args.atten)
    elif args.atten == 'sparge':
        raise ValueError('SpargeAttention requires CUDA')
    elif args.atten == 'sage':
        attention_manager.enable_sage_attention()
    elif args.atten == 'laser':
        attention_manager.enable_laser_attention()
    elif args.atten == 'rainfusion':
        attention_manager.enable_rainfusion_attention()
    elif args.atten == 'sparse':
        attention_manager.enable_sparse_attention()
    elif args.atten == 'mxfp4':
        attention_manager.enable_mxfp4_attention()


def _apply_matmul_quant(pipe, args):
    if args.matmul_quant != 'none':
        dtype = args.matmul_quant
        pipe.transformer = matmul_manager.enable_online_dynamic_quant(pipe.transformer, dtype=dtype)
        # Sync alias: quant replaced pipe.transformer, keep transformer_ref aligned
        if hasattr(pipe, "transformer_ref"):
            pipe.transformer_ref = pipe.transformer
        if WAN_22_FLAG in args.model:
            pipe.transformer_2 = matmul_manager.enable_online_dynamic_quant(pipe.transformer_2, dtype=dtype)
        if args.joint:
            pipe.transformer_3 = matmul_manager.enable_online_dynamic_quant(pipe.transformer_3, dtype=dtype)

    if args.conv3d_w8a8:
        pipe.vae = matmul_manager.enable_vae_conv3d_quant(pipe.vae)
    if args.vae_matmul_quant != 'none':
        pipe.vae = matmul_manager.enable_vae_linear_quant(pipe.vae, dtype=args.vae_matmul_quant)
    if args.rope_fused:
        rope_manager.enable_rope_fused()


def _apply_adaln_cache(pipe, args):
    from x_base.adaln_cache import install_adaln_cache

    if args.task_type == "ref2va" and getattr(args, "reference_paths", None):
        from .h3_references import reference_conditioning

        conditioning_type = reference_conditioning(args)
        logger.info("Determined conditioning_type=%s for adaln cache", conditioning_type)
        install_adaln_cache(pipe, args.adaln_cache_file, conditioning_type=conditioning_type)
        return

    # Determine conditioning type from args to pre-merge constant
    # timesteps (0.999=video, 1.0=audio) at cache load time.
    _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".ogg"}
    _IMAGE_VIDEO_EXTS = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp",
        ".mp4", ".avi", ".mov", ".webm", ".mkv",
    }
    has_video = bool(
        getattr(args, 'keyframe_image', '') or
        getattr(args, 'keyframe_last_image', '')
    )
    has_audio = False
    ref_paths = getattr(args, 'reference_paths', None)
    if ref_paths:
        for path in ref_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in _IMAGE_VIDEO_EXTS:
                has_video = True
            elif ext in _AUDIO_EXTS:
                has_audio = True
    conditioning_type = (
        "both" if (has_video and has_audio) else
        "video" if has_video else
        "audio" if has_audio else
        "none"
    )
    logger.info(f"Determined conditioning_type={conditioning_type} for adaln cache")

    install_adaln_cache(
        pipe,
        cache_file=getattr(args, 'adaln_cache_file', None),
        conditioning_type=conditioning_type,
    )


def operator_ability(pipe, args):
    # 设置自定义attention backend
    if device_type() == "npu":
        apply_npu_attention(pipe)
    _apply_attention_backend(args)
    _apply_matmul_quant(pipe, args)


def parallelism_ability(pipe, args):
    is_contain_transformer_2 = True if WAN_22_FLAG in args.model else False
    is_minimax_h3 = "MiniMax-H3" in args.model

    # SP for DIT (transformer) — independent of TP
    if getattr(args, 'h3_parallel_mode', 'sp') in ('tp_cp', 'tp_cp_shared', 'tp_sp'):
        from .h3_parallel import attach_pipeline
        attach_pipeline(pipe, args)
        if args.sp > 1 and args.phaa_num > 0:
            phaa_on_pipe(pipe, phaa_split_num=args.phaa_num)
    elif args.sp > 1:
        pipe.enable_sp(args.sp, args.ulysses_degree, args.ring_degree,
                       enable_transformer_2=is_contain_transformer_2,
                       enable_transformer_3=args.joint)
        if args.phaa_num > 0:
            phaa_on_pipe(pipe, is_contain_transformer_2, transformer_3=args.joint, phaa_split_num=args.phaa_num)

    if is_minimax_h3 and device_type() == "cuda":
        from .h3_cuda import configure_parallelism

        configure_parallelism(pipe, args)

    # TP for text_encoder (Qwen3-VL) — independent of SP
    # Creates a separate ParallelManager with its own 2D mesh (dp, tp).
    # ParallelManager.__init__ creates the TP group and writes it to
    # tensor_parallelism.state via set_tp_group().
    if args.text_encoder_tp > 1 and is_minimax_h3:
        from x_base import ParallelManager
        tp_size = args.text_encoder_tp
        dp_size = dist.get_world_size() // tp_size
        pipe._tp_parallel_manager = ParallelManager(
            dp_size=dp_size, cp_size=1, sp_size=1, tp_size=tp_size
        )

    if args.vae_lightning != "none":
        pipe.enable_vae_lightning(return_output=args.adopt_sr or args.ten_second or args.vae_lightning=="encoder")
    elif VAE_TILING:
        pipe.vae.enable_tiling()
    if args.fsdp is not None:
        if args.fsdp not in ['all', 'text_encoder', 'transformer']:
            logger.warning(f"Unsupported fsdp: {args.fsdp},fsdp should be 'all' or 'text_encoder' or 'transformer'!")
        else:
            te_model_type = "llama" if "HunyuanVideo" in args.model else ("qwen3vl" if is_minimax_h3 else "t5")
            is_need_dtype = args.matmul_quant == 'none'
            for p in pipe.transformer.parameters():
                p.requires_grad = False
            if is_contain_transformer_2:
                for p in pipe.transformer_2.parameters():
                    p.requires_grad = False
            fsdp_init(pipe, args, is_need_dtype, te_model_type, transformer_2=is_contain_transformer_2)

    # TP 启用
    # 前置条件: ParallelManager 已创建 TP group (通过 enable_sp 或直接创建)
    # TP 只作用于 text_encoder (Qwen3-VL)，不作用于 transformer (DIT)
    if args.text_encoder_tp > 1:
        from x_base import enable_tp
        if is_minimax_h3:
            # MiniMax-H3: TP 目标是 text_encoder (Qwen3-VL)，只对前 50 层做 TP
            enable_tp(pipe.text_encoder, args.text_encoder_tp, target_layers=range(50))
            if getattr(args, "h3_cuda_precision", "none") != "none":
                from .h3_cuda_precision import prepare_text_tp

                prepare_text_tp(pipe)
            # TP 替换后重新 apply group offloading (load_minimax_h3_pipe 中跳过了)
            if not hasattr(pipe.text_encoder, '_tp_offload_applied'):
                from diffusers.hooks import apply_group_offloading
                backend = device_type()
                onload_device = torch.device(backend, torch.get_device_module(backend).current_device())
                apply_group_offloading(pipe.text_encoder, offload_type="leaf_level",
                                       onload_device=onload_device, offload_device=torch.device("cpu"),
                                       use_stream=True, low_cpu_mem_usage=True)
                pipe.text_encoder._tp_offload_applied = True
        else:
            logger.warning(
                f"text_encoder_tp={args.text_encoder_tp} is only supported for MiniMax-H3, skipping."
            )
        logger.info(f"Text encoder tensor parallelism enabled: tp_size={args.text_encoder_tp}")


def general_ability(pipe, args):
    if args.turbo_mode != "default":
        turbo_on_pipe(pipe, args)
    elif args.cache_dit:
        cache_dit_on_pipe(pipe, args)
    if getattr(args, 'adaln_cache_file', None):
        _apply_adaln_cache(pipe, args)

def _check_feature_conflicts(args):
    if args.fuse_lora and args.matmul_quant != 'none':
        args.fuse_lora = False
        logger.warning("Conflict between fuse lora and matmul quant! Disable fuse_lora!")
    if args.joint:
        if not (args.x and ("Wan2.2" in args.model) and ("I2V" in args.model or "i2v" == args.task_type)):
            args.joint = False
            logger.warning("Jointinfer only support Wan2.2_distill_i2v! Disable jointinfer!")
    if args.ada_brighten:
        if not ("I2V" in args.model or "i2v" == args.task_type):
            args.ada_brighten = False
            logger.warning("Ada_brighten only support i2v! Disable ada_brighten!")


def _check_parallelism_conflicts(args):
    # text_encoder_tp 与 FSDP(text_encoder/all) 冲突 — 两者都作用于 text_encoder
    # FSDP(transformer) 不冲突 — TP 作用于 text_encoder，FSDP 作用于 transformer
    if args.text_encoder_tp > 1 and args.fsdp in ('text_encoder', 'all'):
        logger.warning(
            "text_encoder_tp conflicts with --fsdp text_encoder/all (both target text_encoder)! "
            "Disabling FSDP for text_encoder."
        )
        if args.fsdp == 'all':
            args.fsdp = 'transformer'
        else:
            args.fsdp = None

    # transformer_path only supports MiniMax-H3
    if getattr(args, 'transformer_path', None) and "MiniMax-H3" not in args.model:
        args.transformer_path = None
        logger.warning("--transformer_path only supports MiniMax-H3, disabling.")


def check_args_conflict(args):
    if args.vae_lightning != "none" and args.sp == 1:
        args.vae_lightning = "none"
        logger.warning("VAE lightning need sp>1!")
    if args.task_type == "t2i" and args.frames != 1:
        args.frames = 1
        logger.warning("task t2i only support 1 frames!")
    _check_feature_conflicts(args)
    _check_parallelism_conflicts(args)


def update_pipe(pipe, args):
    check_args_conflict(args)
    operator_ability(pipe, args)
    general_ability(pipe, args)
    parallelism_ability(pipe, args)
    if getattr(args, 'h3_attention', 'legacy') != 'legacy':
        from ..framework.transformer.h3_attention import install
        install(pipe.transformer, args)
    return pipe


def load_lora(
        pipe,
        model,
        lora_path_list,
        weights_list=None,
        lora_transformer_list=None
):
    if weights_list is None:
        weights_list = [1.0 / len(lora_path_list) for _ in lora_path_list]
    else:
        try:
            weights_list = [float(weights) for weights in weights_list]
        except Exception:
            logger.error("Invalid lora_scale_weight_list, it must be a float list!")
            weights_list = [1.0 / len(lora_path_list) for _ in lora_path_list]

    if lora_transformer_list is None:
        lora_transformer_list = [0 for _ in lora_path_list]

    adapter_name_list = [f"lora_{i}" for i in range(len(lora_path_list))]
    success_adapter_name_list = []
    success_weights_list = []

    # Check LoRA capability: prefer pipe (LoraLoaderMixin), fallback to pipe.transformer (PeftAdapterMixin).
    has_pipe_lora = hasattr(pipe, "load_lora_weights")
    has_transformer_lora = hasattr(getattr(pipe, "transformer", None), "load_lora_adapter")
    if not has_pipe_lora and not has_transformer_lora:
        logger.warning(f"Neither pipe nor pipe.transformer supports LoRA loading, skipping lora. "
                       f"pipe type={type(pipe).__name__}")
        return

    for i, lora_path in enumerate(lora_path_list):
        try:
            if has_pipe_lora:
                if lora_transformer_list[i] == 0 or lora_transformer_list[i] == 1:
                    pipe.load_lora_weights(lora_path, adapter_name=adapter_name_list[i])
                if (lora_transformer_list[i] == 0 or lora_transformer_list[i] == 2) and WAN_22_FLAG in model:
                    pipe.load_lora_weights(lora_path, adapter_name=adapter_name_list[i], load_into_transformer_2=True)
            else:
                # pipe.transformer has PeftAdapterMixin but not LoraLoaderMixin.
                # prefix=None because LoRA keys don't have a "transformer." prefix.
                from safetensors.torch import load_file as _load_safetensors
                _sd = _load_safetensors(lora_path)

                # Strip ".default." adapter name from keys if present.
                # PEFT's _insert_adapter_name_into_state_dict expects keys WITHOUT adapter name,
                # e.g. "lora_A.weight" not "lora_A.default.weight". If the file has ".default.",
                # PEFT would produce "lora_A.lora_0.default.weight" which doesn't match the
                # model's "lora_A.lora_0.weight", causing all weights to be silently skipped.
                if any('.default.' in k for k in _sd.keys()):
                    _sd = {
                        k.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.'): v
                        for k, v in _sd.items()
                    }

                pipe.transformer.load_lora_adapter(
                    _sd, adapter_name=adapter_name_list[i], prefix=None
                )
            success_adapter_name_list.append(adapter_name_list[i])
            success_weights_list.append(weights_list[i])
        except Exception as e:
            logger.error(f"Load lora_weights {lora_path} FAILED. Error: {type(e).__name__}:{e}")

    if not success_adapter_name_list:
        return
    if has_pipe_lora:
        pipe.set_adapters(success_adapter_name_list, success_weights_list)
    else:
        pipe.transformer.set_adapters(success_adapter_name_list, success_weights_list)


def update_lora(
        pipe,
        model,
        init_lora_path_list,
        lora_path_list,
        fuse_lora=False,
        weights_list=None,
        lora_transformer_list=None
):
    if getattr(pipe, "_h3_turbo_sla_path", None):
        if lora_path_list or fuse_lora:
            raise ValueError("Turbo-SLA does not accept ordinary LoRA overlays or fusion.")
        return
    has_pipe_lora = hasattr(pipe, "load_lora_weights")
    has_transformer_lora = hasattr(getattr(pipe, "transformer", None), "load_lora_adapter")
    if not has_pipe_lora and not has_transformer_lora:
        logger.warning(f"Neither pipe nor pipe.transformer supports LoRA, skipping lora update. "
                       f"pipe type={type(pipe).__name__}")
        return

    if init_lora_path_list != lora_path_list:
        if init_lora_path_list:
            logger.info(f"unload lora weights")
            if has_pipe_lora:
                pipe.unload_lora_weights()
            else:
                pipe.transformer.unload_lora()
        if lora_path_list is not None:
            logger.info(f"load lora weights from {lora_path_list}")
            load_lora(
                pipe,
                model,
                lora_path_list,
                weights_list=weights_list,
                lora_transformer_list=lora_transformer_list
            )
    if lora_path_list is not None and fuse_lora:
        logger.info("fuse lora...")
        if has_pipe_lora:
            pipe.fuse_lora()
        else:
            pipe.transformer.fuse_lora()


def update_scheduler(
        pipe,
        model,
        lora_path_list
):
    if lora_path_list and "qwen" in model.lower():
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(QWEN_LORA_SCHEDULER_CONFIG)
        logger.info(f"update scheduler use config:{QWEN_LORA_SCHEDULER_CONFIG}")
        pipe.scheduler = scheduler


def transformer_vram(transformer, inf_vram_blocks_num=0, save_memory=False):
    group = dist.group.WORLD

    # 根据transformer实例自动读取配置或者block_name
    block_names = offload_config_manager.get_block_name(transformer)
    block_module = getattr(transformer, block_names)
    logger.info(f"apply offload for transformer")

    for name, module in transformer.named_children():
        if name != block_names:
            module.to("npu")

    for name, param in transformer.named_parameters(recurse=False):
        if name != block_names:
            param.data = param.data.to("npu", non_blocking=True)
    if save_memory:
        offloader = OffloadManager_For_Save_Memory(
            transformer,
            module_groups={block_names: block_module},
            keep_n={block_names: int(inf_vram_blocks_num)},
            device=torch.cuda.current_device(),
            dist_group=group,
            sync_at_layer=True if dist.is_initialized() else False,
        )
    else:
        offloader = OffloadManager(
            transformer,
            module_groups={block_names: block_module},
            keep_n={block_names: int(inf_vram_blocks_num)},
            device=torch.cuda.current_device(),
            dist_group=group,
            sync_at_layer=True if dist.is_initialized() else False,
        )
    offloader.enable()
    return transformer


def _is_fsdp(module):
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as _FSDP
    return isinstance(module, _FSDP)


def _pipe_to_device_vram_offload(pipe, args, is_contain_transformer_2):
    npu = torch.device("npu")
    cpu = torch.device("cpu")
    # 2) 所有需要在npu常驻的，pipeline组件，先搬运到npu

    if hasattr(pipe, "vae") and isinstance(pipe.vae, torch.nn.Module):
        pipe.vae.to(npu)

    if hasattr(pipe, "text_encoder") and isinstance(pipe.text_encoder, torch.nn.Module) and not _is_fsdp(pipe.text_encoder):
        pipe.text_encoder.to(npu)

    if hasattr(pipe, "image_encoder") and isinstance(pipe.image_encoder, torch.nn.Module) and not _is_fsdp(pipe.image_encoder):
        pipe.image_encoder.to(npu)

    if hasattr(pipe, "image_processor") and isinstance(pipe.image_processor, torch.nn.Module):
        pipe.image_processor.to(npu)

    pipe.transformer = transformer_vram(pipe.transformer, inf_vram_blocks_num=args.inf_vram_blocks_num,
                                        save_memory=args.save_memory)
    # Sync alias: VRAM offload replaced pipe.transformer, keep transformer_ref aligned
    if hasattr(pipe, "transformer_ref"):
        pipe.transformer_ref = pipe.transformer
    if is_contain_transformer_2:
        pipe.transformer_2 = transformer_vram(pipe.transformer_2, inf_vram_blocks_num=args.inf_vram_blocks_num,
                                              save_memory=args.save_memory)
    if hasattr(pipe, "transformer_3"):
        pipe.transformer_3 = transformer_vram(pipe.transformer_3, inf_vram_blocks_num=args.inf_vram_blocks_num,
                                              save_memory=args.save_memory)


def _pipe_to_device_full(pipe, args):
    # 临时摘掉 FSDP-wrapped 模块，避免 pipe.to() 触发 FSDP _apply 导致 device error
    _fsdp_modules = {}
    for name in ["text_encoder", "image_encoder", "transformer", "transformer_2", "transformer_3"]:
        module = getattr(pipe, name, None)
        if module is not None and _is_fsdp(module):
            _fsdp_modules[name] = module
            setattr(pipe, name, None)
    try:
        pipe.to(device_type())
    finally:
        for name, module in _fsdp_modules.items():
            setattr(pipe, name, module)


def pipe_to_device(pipe, args):
    is_contain_transformer_2 = WAN_22_FLAG in args.model

    if int(args.inf_vram_blocks_num) > 0:
        _pipe_to_device_vram_offload(pipe, args, is_contain_transformer_2)
    else:
        _pipe_to_device_full(pipe, args)

    if getattr(args, "h3_cuda_precision", "none") != "none":
        from .h3_cuda_precision import offload_audio

        offload_audio(pipe.audio_vae)

    return pipe
