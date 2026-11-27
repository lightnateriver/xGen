from copy import deepcopy
from shutil import which
import time
from .validation import validate_h3_image_checkpoint


from x_diffusers.adaptor.validation import get_h3_dit_fusions
from x_diffusers.adaptor.validation import check_args_conflict
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from x_diffusers.framework import initialize_framework

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
from x_diffusers.framework.h3_encoders import truncate_text_encoder
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
    "MiniMax-H3-Image",
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


def configure_h3_npu_fusions(pipe, args, backend):
    """Apply the validated NPU fusion selection through the model's public hook."""
    from .validation import validate_h3_npu_fusions

    features = validate_h3_npu_fusions(args, backend)
    if args.model == "MiniMax-H3":
        configure = getattr(pipe.transformer, "configure_fused_ops", None)
        if features or getattr(pipe.transformer, "_h3_fused_ops", ()):
            if configure is None:
                raise ValueError("Selected H3 transformer does not support NPU fusions")
            configure(features)


def configure_h3_vae_backend(pipe, args, device):
    """Do nothing for the default path; load one CPU master for an Omni worker."""
    from x_diffusers.adaptor.validation import resolve_vae_backend

    selected = resolve_vae_backend(args)
    if selected == "legacy":
        return
    pipe._h3_vae_backend_name = selected
    if selected == "lightning":
        return
    installed = getattr(pipe, "_h3_vae_decode_backend", None)
    if installed is not None:
        if str(installed.path) != str(args.h3_vae_model_path):
            raise ValueError(
                "Restart the pipeline to change the native H3 VAE checkpoint"
            )
        if bool(getattr(installed, "torch_compile_enabled", False)) != bool(
            getattr(args, "h3_vae_torch_compile", False)
        ):
            raise ValueError("Restart the pipeline to change H3 VAE Torch Compiler")
        return
    from x_diffusers.framework.vae.minimax_h3 import OmniH3Decoder

    pipe._h3_vae_decode_backend = OmniH3Decoder(
        args.h3_vae_model_path,
        device,
        trust_remote_code=args.h3_vae_trust_remote_code,
        fp16_linears=True,
        torch_compile=bool(getattr(args, "h3_vae_torch_compile", False)),
    )
    pipe._h3_vae_torch_compile = bool(getattr(args, "h3_vae_torch_compile", False))
    logger.info(
        "Experimental H3 VAE backend=native-stream; native tile parallelism, asynchronous RGB8 transfer and MP4 encoding"
    )


def configure_h3_turbo8(pipe):
    """Assemble the experimental LightX2V 768P eight-evaluation schedule."""
    from types import MethodType
    from .validation import h3_schedule_points
    from x_diffusers.framework.lora.minimax_h3 import training_euler_step

    eval_steps = 8
    points = h3_schedule_points(eval_steps)

    for name, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
        scheduler = getattr(pipe, name)
        scheduler.set_shift(shift)
        scheduler.set_timesteps(points, device="cpu")
        if len(scheduler.timesteps) != eval_steps or len(scheduler.sigmas) != points:
            raise RuntimeError("Turbo8 scheduler must produce eight DiT evaluations")
        scheduler.step = MethodType(training_euler_step, scheduler)
    logger.warning("H3_TURBO8 experimental=true nfe=8 video_shift=6 audio_shift=3")


def configure_h3_turbo_sla(pipe, args):
    """Install the opt-in LoRA and schedule before offloading and parallel hooks."""
    if not getattr(args, "h3_turbo_sla_lora", None):
        return
    from types import MethodType
    from safetensors.torch import load_file
    from x_diffusers.framework.lora.minimax_h3 import install_lora
    from x_diffusers.framework.lora.minimax_h3 import training_euler_step

    for name, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
        scheduler = getattr(pipe, name, None)
        if scheduler is None or type(scheduler).__name__ != "MiniMaxH3Scheduler":
            raise TypeError(f"Turbo-SLA requires Diffusers MiniMaxH3Scheduler for {name}.")
        scheduler.set_shift(shift)
        scheduler.step = MethodType(training_euler_step, scheduler)
    count = install_lora(pipe.transformer, load_file(args.h3_turbo_sla_lora, device="cpu"),
                         has_adaln_cache=bool(args.adaln_cache_file))
    # Keep sparse policy local to H3 attention, never global to the encoder/VAEs.
    for block in pipe.transformer.transformer_blocks:
        block.attn._h3_turbo_sla = True
    for block in pipe.transformer.token_refiner.refiner_blocks:
        block.attn._h3_turbo_sla = True
    pipe._h3_turbo_sla_path = os.path.abspath(args.h3_turbo_sla_lora)
    pipe._h3_turbo_sla_pairs = count
    logger.warning("Turbo-SLA uses the portable selected-block BF16 reference backend; Sage2 INT8 parity and NPU speed are unqualified.")


def configure_h3_parallelism(pipe, args):
    omni_decode = getattr(args, "h3_vae_backend", "legacy") == "native-stream"
    cuda_vae_tiles = os.environ.get("XGEN_H3_CUDA_VAE_TILES") == "1"
    if getattr(args, "h3_cuda_precision", "none") in ("fp8-bf16", "fp8-sage"):
        cuda_vae_tiles = True
    if (
        not omni_decode
        and cuda_vae_tiles
        and not getattr(args, "h3_cuda_vae_sp", False)
    ):
        raise ValueError("XGEN_H3_CUDA_VAE_TILES requires --h3_cuda_vae_sp")
    if not omni_decode and getattr(args, "h3_cuda_vae_sp", False):
        manager = getattr(
            pipe, "_h3_vae_parallel_manager", pipe.transformer.parallel_manager
        )
        from x_diffusers.framework.vae.minimax_h3 import install_vae_parallel

        install_vae_parallel(pipe.vae, manager, tile_parallel=cuda_vae_tiles)
        if cuda_vae_tiles:
            logger.info("Experimental CUDA H3 spatial VAE tile parallelism enabled")
        logger.info("Experimental CUDA H3 VAE SP enabled, sp_size=%s", manager.sp_size)

    fp8_phaa = os.environ.get("XGEN_PHAA_FP8") == "1"
    h3_parallel = getattr(pipe.transformer, "_h3_parallel_config", None)
    if getattr(h3_parallel, "mode", None) == "tp_cp_shared":
        logger.info(
            "H3 shared TP8/CP8 %s sequence exchange; QKV all-to-all fused with TP head layout; no standalone PHAA transport",
            h3_parallel.tp_comm,
        )
        return
    if h3_parallel is not None:
        fp8_phaa = h3_parallel.cp_comm == "fp8"
    async_exchange = os.environ.get("XGEN_PHAA_ASYNC_EXCHANGE") == "1"
    direct_return = os.environ.get("XGEN_PHAA_DIRECT_RETURN") == "1"
    fp8_row_tiling = os.environ.get("XGEN_PHAA_FP8_ROW_TILING") == "1"
    fused_return_pack = os.environ.get("XGEN_PHAA_FUSED_RETURN_PACK") == "1"
    direct_gather = os.environ.get("XGEN_PHAA_DIRECT_GATHER") == "1"
    if direct_return and not async_exchange:
        raise ValueError("XGEN_PHAA_DIRECT_RETURN requires XGEN_PHAA_ASYNC_EXCHANGE=1")
    if fused_return_pack and not direct_return:
        raise ValueError(
            "XGEN_PHAA_FUSED_RETURN_PACK requires XGEN_PHAA_DIRECT_RETURN=1"
        )
    if fp8_row_tiling and not async_exchange:
        raise ValueError("XGEN_PHAA_FP8_ROW_TILING requires XGEN_PHAA_ASYNC_EXCHANGE=1")
    if direct_gather and not async_exchange:
        raise ValueError("XGEN_PHAA_DIRECT_GATHER requires XGEN_PHAA_ASYNC_EXCHANGE=1")
    if async_exchange and not fp8_phaa:
        raise ValueError("XGEN_PHAA_ASYNC_EXCHANGE requires CUDA and XGEN_PHAA_FP8=1")
    if fp8_phaa:
        local_heads = (
            pipe.transformer.transformer_blocks[0].attn.heads
            if h3_parallel is not None
            else pipe.transformer.config.num_attention_heads
        ) // args.sp
        if args.sp < 2 or not 1 <= args.phaa_num <= local_heads:
            raise ValueError(
                "XGEN_PHAA_FP8=1 requires H3, SP > 1, and a valid --phaa_num"
            )
        chunk_order = os.environ.get("XGEN_PHAA_CHUNK_ORDER", "natural")
        if chunk_order not in ("natural", "tail-first"):
            raise ValueError("XGEN_PHAA_CHUNK_ORDER must be natural or tail-first")
        if chunk_order == "tail-first" and args.phaa_num != 4:
            raise ValueError("PHAA tail-first requires --phaa_num 4")
        gather_priority = os.environ.get("XGEN_PHAA_GATHER_PRIORITY") == "1"
        dual_comm = os.environ.get("XGEN_PHAA_DUAL_COMM") == "1"
        device_join = os.environ.get("XGEN_PHAA_DEVICE_JOIN") == "1"
        if async_exchange and not device_join:
            raise ValueError(
                "XGEN_PHAA_ASYNC_EXCHANGE requires XGEN_PHAA_DEVICE_JOIN=1"
            )
        if async_exchange and not callable(
            getattr(dist.Work, "block_current_stream", None)
        ):
            raise RuntimeError(
                "XGEN_PHAA_ASYNC_EXCHANGE requires distributed Work.block_current_stream"
            )
        if device_join and dual_comm:
            raise ValueError(
                "XGEN_PHAA_DEVICE_JOIN requires XGEN_PHAA_DUAL_COMM to be disabled"
            )
        scatter_group = scatter_stream = None
        if dual_comm:
            from x_base.phaa.utils import phaa_cuda_scatter_group

            sp_group = pipe.transformer.parallel_manager.sp_group
            if dist.get_process_group_ranks(sp_group) != list(
                range(dist.get_world_size())
            ):
                raise ValueError(
                    "PHAA dual communication requires a world-spanning SP group"
                )
            scatter_group = phaa_cuda_scatter_group(dist.group.WORLD)
            scatter_stream = torch.cuda.Stream()
        async_options = None
        if async_exchange:
            from x_base.phaa.fp8_async_transport import AsyncFP8Options

            async_options = AsyncFP8Options(
                direct_return=direct_return,
                row_tiling=fp8_row_tiling,
                fused_return_pack=fused_return_pack,
                direct_gather=direct_gather,
            )
        from x_diffusers.framework.transformer.minimax_h3 import configure_phaa_transport

        configure_phaa_transport(
            pipe.transformer,
            gather_priority=gather_priority,
            chunk_order=chunk_order,
            device_join=device_join,
            async_options=async_options,
            scatter_group=scatter_group,
            scatter_stream=scatter_stream,
        )
        logger.info(
            "Experimental H3 PHAA E4M3 transport enabled, chunks=%s, order=%s, gather_priority=%s, dual_comm=%s, device_join=%s, async_exchange=%s, NCCL_MAX_CTAS=%s",
            args.phaa_num,
            chunk_order,
            gather_priority,
            dual_comm,
            device_join,
            async_exchange,
            os.environ.get("NCCL_MAX_CTAS"),
        )
        if direct_return:
            logger.info("Experimental H3 PHAA direct return enabled")
        if fp8_row_tiling:
            logger.info("Experimental H3 PHAA FP8 row tiling enabled (4 rows/program)")
        if fused_return_pack:
            logger.info("Experimental H3 PHAA fused return packing enabled")
        if direct_gather:
            logger.info("Experimental H3 PHAA direct gather enabled")


def resolve_h3_execution_options(args, pipe=None):
    """Resolve once per build; reject a changed installation configuration."""
    from dataclasses import fields
    from x_diffusers.framework.h3_config import H3ExecutionOptions
    from x_diffusers.adaptor.validation import resolve_parallel_config
    from x_diffusers.adaptor.validation import h3_local_io_default as local_io_default
    from x_base.device import device_type

    values = {
        field.name: getattr(args, field.name, field.default)
        for field in fields(H3ExecutionOptions)
        if field.name not in ("parallel", "dit_fusions", "local_io_default")
    }
    parallel_fields = (
        "h3_dit_tp",
        "h3_tp_comm",
        "h3_cp_comm",
        "h3_tp_comm_chunk_elements",
        "h3_parallel_warmup_mib",
    )
    signature = (
        tuple(values.items()),
        tuple(getattr(args, name, None) for name in parallel_fields),
    )
    previous = (
        getattr(pipe, "_xgen_h3_execution_options", None) if pipe is not None else None
    )
    if isinstance(previous, H3ExecutionOptions):
        if signature != pipe._xgen_h3_execution_key:
            raise ValueError("Restart the pipeline to change H3 installation settings")
        return previous
    parallel = resolve_parallel_config(
        args, device_type(), dist.get_world_size() if dist.is_initialized() else 1
    )
    fusions = frozenset()
    if values["h3_cuda_precision"] in ("fp8-bf16", "fp8-sage"):
        fusions = frozenset(
            (
                "XGEN_H3_FUSED_INDEXED_AFFINE",
                "XGEN_H3_FUSED_PARTIAL_ROPE",
                "XGEN_H3_FUSED_QK_RMS",
            )
        )
        if os.environ.get("XGEN_H3_FUSED_OUTPUT_ADALN") == "1":
            fusions |= {"XGEN_H3_FUSED_OUTPUT_ADALN"}
    options = H3ExecutionOptions(
        **values,
        parallel=parallel,
        dit_fusions=fusions,
        local_io_default=local_io_default(parallel.mode),
    )
    if pipe is not None:
        pipe._xgen_h3_execution_options = options
        pipe._xgen_h3_execution_key = signature
    return options





def configure_h3_image_attention(args, backend):
    """Select Image attention through the shared operator manager."""
    from x_base import attention_manager

    if args.atten == "laser":
        if backend != "npu":
            raise ValueError("Laser attention requires an Ascend NPU")
        from x_base.operator.attention import LA_AVAILABLE
        if not LA_AVAILABLE:
            raise RuntimeError("Laser attention was requested but MindIE-SD is unavailable")
        attention_manager.enable_laser_attention()
    else:
        attention_manager.disable_laser_attention()


def _configure_h3_image_dit(pipe, args, *, root, device):
    from x_diffusers.framework.transformer.minimax_h3_image import fuse_image_qkv
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_adaln_table
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_dit_tp
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_ff_chunk
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_refiner_sp
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_swiglu
    from x_diffusers.framework.transformer.minimax_h3_image import load_image_adaln_table
    from x_diffusers.framework.transformer.minimax_h3_image import quantize_image_blocks
    from x_diffusers.framework.transformer.minimax_h3_image import save_image_adaln_table
    from x_diffusers.framework.transformer.minimax_h3_image import enable_h3_image_sp

    transformer = pipe.transformer
    adaln_table = getattr(args, "image_adaln_table_file", None)
    save_adaln_table = getattr(args, "image_adaln_table_save", None)
    if adaln_table:
        load_image_adaln_table(
            adaln_table,
            root,
            transformer,
            pipe.config.sample_sigmas,
            pipe.reference_noise_aug,
            device,
        )
    elif getattr(args, "image_adaln_table", False) or save_adaln_table:
        install_image_adaln_table(
            transformer, pipe.config.sample_sigmas, pipe.reference_noise_aug, device
        )
        if save_adaln_table and (not dist.is_initialized() or dist.get_rank() == 0):
            save_image_adaln_table(
                save_adaln_table,
                root,
                transformer,
                pipe.config.sample_sigmas,
                pipe.reference_noise_aug,
            )
    if getattr(args, "image_fused_qkv", False):
        fuse_image_qkv(transformer)
    if getattr(args, "image_int8", False):
        if device.type != "npu":
            raise ValueError("Image INT8 requires NPU")
        quantize_image_blocks(transformer)
    dit_tp = getattr(args, "image_dit_tp", 1)
    enable_h3_image_sp(transformer, args.sp, dit_tp)
    if dit_tp > 1:
        install_image_dit_tp(transformer, dist.group.WORLD)
    if getattr(args, "image_refiner_sp", False) and args.sp > 1:
        install_image_refiner_sp(transformer)
    from x_diffusers.framework.transformer.minimax_h3_image import configure_image_attention

    configure_image_attention(
        transformer,
        fused_rope=bool(getattr(args, "image_fused_rope", False)),
        dense_tail=getattr(args, "image_laser_dense_tail", 0),
    )
    if getattr(args, "image_ff_chunk", 0):
        install_image_ff_chunk(transformer, args.image_ff_chunk)
    if getattr(args, "image_swiglu", "none") != "none":
        install_image_swiglu(transformer, args.image_swiglu)


def _configure_h3_image_text_encoder(
    pipe, args, *, device, world, low_cpu_mem_usage
):
    from diffusers.hooks import apply_group_offloading

    if args.text_encoder_tp > 1:
        from x_diffusers.framework.h3_encoders import enable_text_tp

        enable_text_tp(pipe, args.text_encoder_tp, create_manager=True)
    if getattr(args, "image_text_int8", False):
        from x_diffusers.framework.transformer.minimax_h3_image import quantize_image_text_encoder

        quantize_image_text_encoder(pipe.text_encoder)
    if getattr(args, "image_text_resident", False) or args.text_encoder_tp > 1:
        pipe.text_encoder.to(device)
    else:
        apply_group_offloading(
            pipe.text_encoder,
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    if getattr(args, "image_vision_sp", False) and world > 1:
        from x_diffusers.framework.h3_encoders import install_vision_sp

        install_vision_sp(pipe.text_encoder.model.visual, lambda: dist.group.WORLD)


def _place_h3_image_dit(transformer, args, *, device, low_cpu_mem_usage):
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_condition_cache
    from x_diffusers.framework.transformer.minimax_h3_image import install_image_local_input

    # A BF16 DiT alone exceeds a 64 GiB 910B; stage one block at a time.
    if getattr(args, "image_resident", False):
        transformer.to(device)
    else:
        transformer.enable_group_offload(
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    if getattr(args, "image_condition_cache", False):
        install_image_condition_cache(transformer)
    if getattr(args, "image_local_input", False):
        install_image_local_input(transformer)


@torch.inference_mode(False)
@torch.no_grad()
def load_minimax_h3_image_pipe(args, *, device, world):
    """Load checkpoint components and install the selected xGen device policies."""
    try:
        from diffusers import MiniMaxH3ImageModularPipeline
    except ImportError as error:
        raise ImportError(
            "H3 Image needs the pinned optional Diffusers source in requirements-h3-image.txt"
        ) from error
    from x_diffusers.framework.h3_encoders import truncate_text_encoder

    model_root = args.pretrained_model_name_or_path
    validate_h3_image_checkpoint(model_root)
    root = str(Path(model_root).resolve())
    pipe = MiniMaxH3ImageModularPipeline.from_pretrained(root, local_files_only=True)
    pipe.load_components(
        dtype=torch.bfloat16, pretrained_model_name_or_path=root, local_files_only=True
    )
    truncate_text_encoder(pipe, num_keep=50)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    low_cpu_mem_usage = not getattr(args, "image_pinned_cpu_offload", False)
    _configure_h3_image_dit(pipe, args, root=root, device=device)
    _configure_h3_image_text_encoder(
        pipe, args, device=device, world=world, low_cpu_mem_usage=low_cpu_mem_usage
    )
    _place_h3_image_dit(
        pipe.transformer, args, device=device, low_cpu_mem_usage=low_cpu_mem_usage
    )

    # The upstream VAE keeps its required FP32 operations and latent normalization.
    pipe.vae.to(device=device)
    vae_parallel = getattr(args, "image_vae_parallel", False) and world > 1
    vae_decode_batch = getattr(args, "image_vae_decode_batch_size", 1)
    if vae_parallel or vae_decode_batch > 1:
        from x_diffusers.framework.vae.minimax_h3_image import (
            install_image_vae_parallel,
        )

        install_image_vae_parallel(
            pipe.vae,
            dist.group.WORLD if vae_parallel else None,
            gather_once=getattr(args, "image_vae_gather_once", False),
            decode_batch_size=vae_decode_batch,
        )
    if getattr(args, "image_rgb8_output", False):
        from ..framework.pipeline.minimax_h3_image import install_image_rgb8_output

        install_image_rgb8_output(pipe)
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
    # Load audio directly in FP32: upcasting rounded BF16/FP16 weights loses precision.
    component_dtypes = {"default": dtype, "audio_vae": torch.float32}
    task_type = getattr(args, "task_type", "t2va")
    packed_checkpoint = getattr(args, "h3_nvfp4_checkpoint", None)
    packed_resident = getattr(args, "h3_nvfp4_resident", False)
    packed_compile = getattr(args, "h3_nvfp4_compile", False)
    cuda_fusions = ()
    if device_type() == "cuda":
        cuda_fusions = get_h3_dit_fusions(packed_resident, packed_compile)
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

    from x_diffusers.adaptor.load_pipe import configure_h3_reference_image_size as configure_reference_image_size
    configure_reference_image_size(pipe, args)
    if task_type == "ref2va" and not getattr(args, "h3_refiner", False):
        from x_diffusers.framework.pipeline.minimax_h3 import install_reference_setup

        install_reference_setup(pipe)

    # Keep automatic T2VA/FL2VA routing, but do not load the unused Ref2VA DiT
    # from a combined repository into a resident CUDA/portable H3 pipeline.
    component_options = ({"workflow": "fl2va"} if task_type in ("t2va", "fl2va")
                         and not getattr(args, "h3_refiner", False)
                         and (getattr(args, "h3_cuda_precision", "none") != "none"
                              or getattr(args, "h3_features", None) == "full-parallel") else {})

    # Load transformer from a custom checkpoint directory (full or pre-stripped).
    # from_pretrained is effectively strict=False: missing keys (e.g. adaln in
    # pre-stripped checkpoints) only produce warnings, and with low_cpu_mem_usage
    # they stay on meta device (zero memory).
    transformer_path = getattr(args, 'transformer_path', None)
    if packed_checkpoint:
        from x_diffusers.framework.transformer.minimax_h3 import load_packed_h3

        config = MiniMaxH3Transformer3DModel.load_config(
            args.pretrained_model_name_or_path, subfolder="transformer"
        )
        logger.info("Loading experimental H3 FL2VA NVFP4 transformer")
        transformer = load_packed_h3(MiniMaxH3Transformer3DModel, config, packed_checkpoint)
        pipe.update_components(transformer=transformer)
        pipe.load_components(dtype=component_dtypes, pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                             **component_options)
    elif transformer_path:
        logger.info(f"Loading transformer from: {transformer_path}")
        transformer_key = "transformer_ref" if task_type == "ref2va" else "transformer"
        pipe.load_components(
            dtype=component_dtypes,
            **component_options,
            pretrained_model_name_or_path={
                transformer_key: transformer_path,
                "default": args.pretrained_model_name_or_path,
            },
            subfolder={
                transformer_key: "",  # transformer_path is the direct path, no subfolder
            },
        )
    else:
        pipe.load_components(dtype=component_dtypes, pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                             **component_options)

    from x_diffusers.framework.vae.minimax_h3 import prepare_audio_vae
    prepare_audio_vae(pipe.audio_vae)

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
    configure_h3_turbo_sla(pipe, args)
    if getattr(args, "h3_turbo8", False):
        configure_h3_turbo8(pipe)
    truncate_text_encoder(pipe, num_keep=50)

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
        from x_diffusers.framework.transformer.minimax_h3 import prepare_transformer

        prepare_transformer(pipe, resolve_h3_execution_options(args, pipe), onload_device)
    elif packed_resident:
        logger.info("Keeping packed H3 DiT resident on %s", onload_device)
        pipe.transformer.to(onload_device)
        if cuda_fusions:
            from x_diffusers.framework.transformer.minimax_h3 import install_dit_fusions

            install_dit_fusions(pipe.transformer, cuda_fusions)
        if packed_compile:
            from x_diffusers.framework.transformer.minimax_h3 import compile_resident_blocks
            compile_resident_blocks(pipe.transformer)
    elif args.matmul_quant == 'none' and not getattr(args, 'h3_refiner', False):
        pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    if args.text_encoder_tp <= 1:
        apply_group_offloading(pipe.text_encoder, offload_type="leaf_level", **offload)

    # VAEs are small enough to stay on device
    if getattr(args, "h3_cuda_precision", "none") != "none":
        from x_diffusers.framework.pipeline.minimax_h3 import prepare_vaes

        prepare_vaes(pipe, resolve_h3_execution_options(args, pipe), onload_device)
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


def to_dtype(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _load_pipe_components(args):
    backend = device_type()
    from x_diffusers.adaptor.validation import validate_pipeline_args
    validate_pipeline_args(args, backend)
    if args.atten == "vc":
        _apply_attention_backend(args)
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


@dataclass(frozen=True)
class PipelineBuildOptions:
    initial_lora_paths: object = None
    configure_scheduler: bool = False
    lora_after_setup: bool = False
    fsdp_for_lora: str | None = None
    fuse_lora: bool | None = None
    lora_transformer_list: object = None
    before_placement: Callable[[], None] | None = None


def _finish_pipeline(pipe, args, options=None):
    """Preserve entry-specific LoRA ordering while returning a fully built pipe."""
    options = options or PipelineBuildOptions()
    if options.configure_scheduler:
        update_scheduler(pipe, args.model, args.lora_path_list)

    def load_lora():
        kwargs = dict(
            fuse_lora=args.fuse_lora
            if options.fuse_lora is None
            else options.fuse_lora,
            weights_list=args.lora_scale_weight_list,
        )
        if options.lora_after_setup:
            kwargs["lora_transformer_list"] = options.lora_transformer_list
        update_lora(
            pipe, args.model, options.initial_lora_paths, args.lora_path_list, **kwargs
        )

    if options.lora_after_setup:
        if args.lora_path_list and options.fsdp_for_lora is not None:
            args.fsdp = options.fsdp_for_lora
        update_pipe(pipe, args)
        if args.lora_path_list:
            load_lora()
    else:
        load_lora()
        update_pipe(pipe, args)
    if options.before_placement is not None:
        options.before_placement()
    return pipe_to_device(pipe, args)


def load_pipe(args, *, build_options=None, model=None):
    """Build a ready-to-run pipeline through one application entry point."""
    initialize_framework()
    if (model or getattr(args, 'model', None)) == 'MiniMax-H3-Image':
        from x_diffusers.adaptor.validation import resolve_image_text_options
        from x_diffusers.adaptor.validation import validate_h3_image_options as _validate_request_options
        backend = device_type()
        world = dist.get_world_size() if dist.is_initialized() else 1
        resolve_image_text_options(args, backend)
        _validate_request_options(args, world)
        configure_h3_image_attention(args, backend)
        device = torch.device(backend, torch.get_device_module(backend).current_device())
        return load_minimax_h3_image_pipe(args, device=device, world=world)
    pipe = _load_pipe_components(args)
    return _finish_pipeline(pipe, args, build_options)


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
    else:
        attention_manager.configure_npu_attention(args.atten)


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
        from x_diffusers.adaptor.validation import h3_reference_conditioning as reference_conditioning

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
    if is_minimax_h3 and getattr(args, 'h3_parallel_mode', 'sp') in ('tp_cp', 'tp_cp_shared', 'tp_sp'):
        from x_diffusers.adaptor.validation import h3_features_enabled as enabled_features
        from x_diffusers.framework.transformer.minimax_h3 import attach_pipeline
        attach_pipeline(pipe, vae_parallel=bool(getattr(args, "h3_cuda_vae_sp", False) or enabled_features(args)))
        if args.sp > 1 and args.phaa_num > 0:
            phaa_on_pipe(pipe, phaa_split_num=args.phaa_num)
    elif args.sp > 1:
        pipe.enable_sp(args.sp, args.ulysses_degree, args.ring_degree,
                       enable_transformer_2=is_contain_transformer_2,
                       enable_transformer_3=args.joint)
        if args.phaa_num > 0:
            phaa_on_pipe(pipe, is_contain_transformer_2, transformer_3=args.joint, phaa_split_num=args.phaa_num)

    if is_minimax_h3 and device_type() == "cuda":
        configure_h3_parallelism(pipe, args)

    if is_minimax_h3 and getattr(args, 'h3_cuda_precision', 'none') == 'none':
        from x_diffusers.adaptor.validation import h3_features_enabled as enabled
        from x_diffusers.framework.pipeline.minimax_h3 import configure_components
        if enabled(args):
            configure_components(
                pipe, resolve_h3_execution_options(args, pipe),
                torch.device(device_type(), torch.get_device_module(device_type()).current_device()),
            )

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
            from x_diffusers.framework.h3_encoders import enable_text_tp
            enable_text_tp(pipe, args.text_encoder_tp)
            if getattr(args, "h3_cuda_precision", "none") != "none":
                from x_diffusers.framework.transformer.minimax_h3 import prepare_text_tp

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


def update_pipe(pipe, args):
    check_args_conflict(args)
    operator_ability(pipe, args)
    general_ability(pipe, args)
    if args.model == 'MiniMax-H3' and getattr(args, 'h3_cuda_precision', 'none') == 'none':
        from x_diffusers.adaptor.validation import h3_features_enabled as enabled
        from x_diffusers.framework.pipeline.minimax_h3 import configure_model
        if enabled(args):
            configure_model(
                pipe, resolve_h3_execution_options(args, pipe),
                torch.device(device_type(), torch.get_device_module(device_type()).current_device()),
            )
    parallelism_ability(pipe, args)
    if getattr(args, 'h3_vae_backend', 'legacy') != 'legacy':
        configure_h3_vae_backend(pipe, args, pipe._execution_device)
    from x_diffusers.adaptor.load_pipe import configure_h3_attention as configure_attention

    configure_attention(pipe, args)
    configure_h3_npu_fusions(pipe, args, device_type())
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
        from x_diffusers.framework.vae.minimax_h3 import offload_audio

        offload_audio(pipe.audio_vae)

    return pipe


def prepare_h3_refiner(manager, args):
    """Coordinate Refiner startup failures independently of device collectives."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() != 8:
        raise ValueError('Initialize the existing XGEN distributed environment with eight ranks first')
    guard = getattr(manager, '_h3_refiner_failure_guard', None)
    if guard is None:
        from x_base.utils.distributed import DistributedFailureGuard

        try:
            guard = DistributedFailureGuard()
        except BaseException:
            manager._h3_refiner_failed = True
            raise
        manager._h3_refiner_failure_guard = guard
    with guard:
        try:
            return _prepare_h3_refiner(manager, args)
        except BaseException as error:
            manager._h3_refiner_failed = True
            guard.fail(error)
            raise


def _prepare_h3_refiner(manager, args):
    """Load once through the original loader/ability functions; borrow manager.pipe."""
    from x_diffusers.adaptor.validation import h3_refiner_model_key
    from x_diffusers.adaptor.validation import resolve_h3_npu_features
    from x_diffusers.adaptor.validation import validate_h3_refiner_settings
    validate_h3_refiner_settings(args)
    resolve_h3_npu_features(args)
    if getattr(manager, '_h3_refiner_failed', False):
        raise RuntimeError('H3 Refiner initialization failed; restart the worker before loading again')
    if getattr(manager, '_h3_refiner_key', None) is not None:
        if manager._h3_refiner_key != h3_refiner_model_key(args):
            raise ValueError('H3 Refiner model/workflow/settings are fixed at startup; restart for changes')
        if manager.sr_pipe.failed:
            raise RuntimeError('H3 Refiner worker failed; restart before accepting more requests')
        return
    if manager.pipe is not None:
        raise ValueError('Enable H3 Refiner on a fresh manager; restart to change the backend')
    from x_diffusers.framework.pipeline.minimax_h3 import H3RefinerPipeline
    started = time.perf_counter()
    try:
        from x_base.vae_parallelism.save_av_stream import FFMPEG_PATH
        if which(FFMPEG_PATH) is None:
            raise RuntimeError('H3 Refiner requires FFmpeg on PATH or an executable FFMPEG_PATH')
        pipe = load_pipe(args)
        from x_diffusers.framework.h3_config import H3RefinerConfig
        config = H3RefinerConfig(
            weights=args.h3_refiner_weights, prepared=args.h3_refiner_prepared,
            backend=device_type(), loading=args.h3_refiner_loading,
            base_steps=args.num_inference_steps, turbo8=getattr(args, "h3_turbo8", False),
            base_cache_file=getattr(args, "adaln_cache_file", None),
        )
        refiner = H3RefinerPipeline(pipe, config)
    except Exception:
        manager._h3_refiner_failed = True
        logger.exception('H3_REFINER_INIT_FAILED retry=false restart_required=true')
        raise
    manager.pipe, manager.sr_pipe = pipe, refiner
    manager.init_args = deepcopy(args)
    manager.current_model = args.model
    manager._h3_refiner_key = h3_refiner_model_key(args)
    refiner.startup_seconds = time.perf_counter() - started
    logger.warning('H3_REFINER_READY experimental=true sp=8 init_seconds=%.3f backend=%s',
                   refiner.startup_seconds, args.h3_refiner_loading)


def configure_h3_attention(pipe, args):
    """Install a selected optional backend; leave the legacy pipeline untouched."""
    from x_diffusers.adaptor.validation import h3_attention_config
    from x_diffusers.adaptor.validation import validate_h3_attention
    if getattr(args, 'h3_attention', 'legacy') != 'legacy':
        from x_diffusers.framework.transformer.minimax_h3 import install_h3_attention as install

        validate_h3_attention(args)
        install(pipe.transformer, h3_attention_config(args))


def configure_h3_reference_image_size(pipe, args):
    """Apply the legacy fixed short-edge override."""
    import logging
    from x_diffusers.adaptor.validation import validate_h3_reference_image_size
    edge = validate_h3_reference_image_size(args)
    if edge is None:
        return
    if "reference_image_short_edge" not in pipe.config:
        raise ValueError("This Diffusers pipeline does not expose reference_image_short_edge; upgrade the H3 runtime")
    previous = pipe.config["reference_image_short_edge"]
    pipe.update_components(reference_image_short_edge=edge)
    logging.getLogger(__name__).info(
        "Experimental H3 reference image short edge: %s -> %d; shared VAE/Qwen input, output canvas unchanged",
        previous, edge,
    )
