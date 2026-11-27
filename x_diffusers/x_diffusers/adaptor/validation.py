"""Shared parameter defaults, compatibility checks and startup identities.

No model loading or accelerator imports execute when this module is imported.
Runtime facts are queried only by the checks that require them.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import fnmatch
import json
import subprocess
from x_diffusers.adaptor.config import (H3_REFERENCE_IMAGE_EXTS, H3_REFERENCE_VIDEO_EXTS, H3_REFERENCE_AUDIO_EXTS, H3_REFERENCE_LIMITS, H3_REFERENCE_MAX_FILES, H3_REFERENCE_MIN_SECONDS, H3_REFERENCE_MAX_SECONDS)
from x_diffusers.framework.h3_config import REFERENCE_RESIZE_MODES


import logging
import math
import os
from pathlib import Path
from x_diffusers.framework.h3_config import H3ParallelSettings
from x_diffusers.adaptor.config import (
    H3_CUDA_PRESETS,
    H3_CUDA_BOOL_OPTIONS,
    H3_CUDA_STARTUP_OPTIONS,
    H3_FEATURE_BOOL_FIELDS,
    H3_FEATURE_VALUE_FIELDS,
    H3_FEATURE_ALIASES,
    H3_NPU_FUSION_FEATURES,
)

logger = logging.getLogger("infer")


def h3_schedule_points(steps, cache_file=None):
    """Resolve H3 sigma-point count after checking the requested evaluations."""
    if type(steps) is not int or steps < 1:
        raise ValueError("H3 num_inference_steps must be a positive integer")
    if cache_file and steps != h3_adaln_cache_steps(cache_file):
        raise ValueError("H3 num_inference_steps must match the Base AdaLN cache evaluation count")
    # MiniMaxH3Scheduler expects the terminal sigma=0 in its point count.
    from x_diffusers.framework.h3_config import schedule_points as model_schedule_points

    return model_schedule_points(steps)


def h3_adaln_cache_steps(path):
    """Read the cache's evaluation count and check its sigma-point metadata."""
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(path, framework="pt") as cache:
            metadata = cache.metadata() or {}
        points = int(metadata["num_steps"])
        steps = int(metadata.get("num_eval_steps", points - 1))
    except (OSError, SafetensorError, KeyError, ValueError) as exc:
        raise ValueError("Invalid H3 AdaLN cache schedule metadata") from exc
    if steps < 1 or points != steps + 1:
        raise ValueError("H3 AdaLN cache must contain one more sigma point than evaluations")
    return steps


def _distributed():
    import torch.distributed as dist

    return dist


def validate_h3_attention(args):
    if getattr(args, "h3_attention", "legacy") == "legacy":
        return
    if args.h3_attention not in ("spark", "veda"):
        raise ValueError("Unknown H3 attention backend")
    if args.h3_attention == "spark" and os.environ.get("TASK_QUEUE_ENABLE") not in (
        "0",
        "1",
    ):
        raise ValueError(
            "Spark NPU graph capture requires TASK_QUEUE_ENABLE=1 (or 0) before process startup"
        )
    if any(
        (
            type(getattr(args, field)) is not int
            for field in (
                "sp",
                "num_inference_steps",
                "frames",
                "width",
                "height",
                "save_fps",
            )
        )
    ):
        raise ValueError(
            "Sparse H3 parallelism, schedule and video dimensions must be integers"
        )
    i2va = args.h3_attention == "veda" and args.task_type == "fl2va"
    if args.model != "MiniMax-H3" or (args.task_type != "t2va" and (not i2va)):
        raise ValueError(
            "Experimental H3 attention supports T2VA and the Veda I2VA profile"
        )
    if args.h3_attention == "veda" and getattr(args, "h3_veda_sparsity", 0.9) not in (
        0.7,
        0.9,
    ):
        raise ValueError("Veda sparsity must be 0.7 or 0.9")
    if args.dtype != "bf16" or args.atten != "laser" or args.sp != 8:
        raise ValueError(
            "Use BF16, SP8 and atten=laser (legacy components retain Laser)"
        )
    steps = 8 if getattr(args, "h3_turbo8", False) else 50
    if i2va:
        if (
            not getattr(args, "keyframe_image", None)
            or getattr(args, "h3_refiner", False)
            or getattr(args, "h3_turbo8", False)
        ):
            raise ValueError(
                "Veda I2VA requires one first frame and the original 768P Base model"
            )
        steps = 20
    frames = 243 if i2va else 124
    if args.num_inference_steps != steps or args.frames != frames:
        raise ValueError(
            "Initial H3 attention validation requires the configured schedule and native frame count"
        )
    expected = (2560, 1440) if getattr(args, "h3_refiner", False) else (1344, 768)
    if (args.width, args.height) != expected or args.save_fps != 24:
        raise ValueError(
            "Use 5s landscape 768P, or the existing 2K Refiner chain, at 24fps"
        )
    if getattr(args, "h3_refiner", False) and (
        type(args.h3_refiner_seconds) is not int or args.h3_refiner_seconds != 5
    ):
        raise ValueError("Initial H3 attention supports five seconds only")
    if (
        not i2va
        and getattr(args, "keyframe_image", None)
        or any(
            (
                getattr(args, h3_attention_key, None)
                for h3_attention_key in (
                    "keyframe_last_image",
                    "reference_paths",
                    "h3_turbo_sla_lora",
                    "lora_path_list",
                    "cache_dit",
                )
            )
        )
    ):
        raise ValueError(
            "Initial H3 attention does not accept conditioning media, LoRA or DiT cache"
        )
    if args.h3_attention == "veda" and (
        not Path(args.h3_veda_predictor or "").is_file()
    ):
        raise ValueError("Veda requires an existing h3_veda_predictor file")


def h3_attention_key(args):
    return (
        getattr(args, "h3_attention", "legacy"),
        getattr(args, "h3_veda_predictor", None),
        getattr(args, "h3_turbo8", False),
        getattr(args, "h3_veda_sparsity", 0.9),
    )


def h3_overlap_key(args):
    return (
        bool(getattr(args, "h3_condition_overlap", False)),
        float(getattr(args, "h3_condition_overlap_timeout", 600.0)),
    )


def validate_h3_overlap(args, backend):
    if not h3_overlap_key(args)[0]:
        return
    if backend not in ("cuda", "npu") or getattr(args, "model", None) != "MiniMax-H3":
        raise ValueError("h3_condition_overlap requires CUDA or NPU MiniMax-H3")
    if getattr(args, "task_type", None) != "ref2va" or getattr(
        args, "h3_refiner", False
    ):
        raise ValueError("h3_condition_overlap currently supports Base Ref2VA only")
    if getattr(args, "h3_parallel_mode", "sp") != "sp":
        raise ValueError("h3_condition_overlap requires the native SP layout")
    timeout = h3_overlap_key(args)[1]
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("h3_condition_overlap_timeout must be finite and positive")
    if os.environ.get("DEBUG_TIME"):
        raise ValueError(
            "Disable DEBUG_TIME barriers when using h3_condition_overlap; use its metrics instead"
        )


def validate_h3_overlap_manager(manager, args, backend):
    validate_h3_overlap(args, backend)
    key = h3_overlap_key(args)
    previous = getattr(
        manager,
        "_h3_condition_overlap_key",
        h3_overlap_key(getattr(manager, "init_args", None))
        if manager.pipe is not None
        else key,
    )
    if manager.pipe is not None and previous != key:
        raise ValueError(
            "Restart the manager/service to change H3 condition overlap options"
        )
    manager._h3_condition_overlap_key = key


def h3_local_io_default(mode):
    return mode in ("tp_cp", "tp_cp_shared")


def select_h3_default_recipe(args, backend):
    """Only unqualified CUDA H3 Base; explicit legacy/operator choices win."""
    if (
        backend != "cuda"
        or getattr(args, "model", None) != "MiniMax-H3"
        or getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va")
        or (getattr(args, "h3_cuda_precision", "auto") != "auto")
        or (getattr(args, "h3_cuda_preset", None) is not None)
        or (getattr(args, "h3_features", None) not in (None, "none"))
        or (getattr(args, "h3_parallel_mode", "sp") != "sp")
        or (getattr(args, "h3_cuda_linear_tp", 1) != 1)
        or (getattr(args, "sp", 1) != 8)
        or (getattr(args, "text_encoder_tp", 1) != 8)
        or (getattr(args, "dtype", "bf16") != "bf16")
        or (getattr(args, "atten", "none") != "none")
        or (getattr(args, "matmul_quant", "none") != "none")
        or (getattr(args, "h3_attention", "legacy") != "legacy")
        or (getattr(args, "phaa_num", 0) is None)
    ):
        return
    legacy = (
        "h3_nvfp4_checkpoint",
        "h3_nvfp4_resident",
        "h3_nvfp4_compile",
        "h3_turbo8",
        "h3_turbo_sla_lora",
        "h3_refiner",
        "lora_path_list",
        "fsdp",
        "inf_vram_blocks_num",
        "cache_dit",
    )
    if getattr(args, "h3_cuda_adaln_mode", None) != "table":
        legacy += ("transformer_path", "adaln_cache_file")
    if not any((getattr(args, name, None) for name in legacy)):
        args.h3_cuda_preset = "sage-fp8-full-parallel"


def resolve_h3_options(args, backend):
    """Select the CUDA default or explicit preset, preserving per-feature overrides."""
    if not hasattr(args, "_h3_residency_explicit"):
        args._h3_residency_explicit = getattr(
            args, "h3_cuda_dit_residency", None
        ) not in (None, "inherit")
    resolve_h3_feature_defaults(args)
    select_h3_default_recipe(args, backend)
    preset = getattr(args, "h3_cuda_preset", None) or "none"
    if preset not in H3_CUDA_PRESETS:
        raise ValueError("Unknown H3 CUDA preset")
    full = preset == "sage-fp8-full-parallel"
    environment = {}
    if full:
        if (
            backend != "cuda"
            or getattr(args, "model", None) != "MiniMax-H3"
            or getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va")
        ):
            raise ValueError(
                "sage-fp8-full-parallel requires CUDA H3 T2VA/FL2VA/Ref2VA"
            )
        if getattr(args, "h3_cuda_precision", "auto") == "auto":
            args.h3_cuda_precision = "fp8-sage"
        for key, value in (
            ("NCCL_MAX_CTAS", "2"),
            ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            ("XGEN_H3_FUSED_OUTPUT_ADALN", "1"),
            ("XGEN_PHAA_FP8_ALIGN16", "1"),
        ):
            environment[key] = os.environ.get(key, value)
    args.h3_cuda_preset = preset
    mode = getattr(args, "h3_parallel_mode", "sp")
    alias = getattr(args, "h3_cuda_linear_tp", 1)
    if type(alias) is not int or alias not in (1, 2):
        raise ValueError("h3_cuda_linear_tp must be 1 or 2")
    if alias == 2:
        if backend != "cuda" or getattr(args, "sp", 1) != 8:
            raise ValueError(
                "h3_cuda_linear_tp=2 requires CUDA H3 FP8 and original SP8 mode"
            )
        if mode not in ("sp", "tp_sp") or getattr(args, "h3_dit_tp", 1) not in (1, 2):
            raise ValueError(
                "h3_cuda_linear_tp=2 conflicts with the selected DiT parallel layout"
            )
        args.h3_parallel_mode, args.h3_dit_tp = mode = ("tp_sp", 2)
        args.h3_cuda_linear_tp = 1
    defaults = dict(
        h3_cuda_local_input=h3_local_io_default(mode),
        h3_cuda_local_output=h3_local_io_default(mode),
    )
    for key in H3_CUDA_BOOL_OPTIONS:
        if key in (
            "h3_cuda_phaa_fp8",
            "h3_cuda_phaa_direct_pack",
            "h3_cuda_input_refiner_sp",
        ):
            continue
        value = getattr(args, key, None)
        if value is None:
            value = defaults.get(key, False) or (
                full
                and key
                in (
                    "h3_cuda_local_input",
                    "h3_cuda_local_output",
                    "h3_cuda_precompute_conditioning",
                    "h3_cuda_vision_sp",
                    "h3_cuda_vae_encoder_parallel",
                    "h3_cuda_audio_encoder_parallel",
                )
            )
        if type(value) is not bool:
            raise ValueError(key + " must be boolean")
        setattr(args, key, value)
    old = getattr(args, "h3_cuda_input_refiner", None)
    legacy = {
        "none": (False, "legacy", "bf16"),
        "cudnn-sp": (True, "cudnn", "bf16"),
        "cudnn-sp-fp8": (True, "cudnn", "fp8"),
    }
    if old not in (None, "custom", *legacy):
        raise ValueError("Unknown h3_cuda_input_refiner profile")
    inherited = legacy.get(old, (True, "cudnn", "fp8") if full else legacy["none"])
    for key, default in zip(
        (
            "h3_cuda_input_refiner_sp",
            "h3_cuda_input_refiner_attention",
            "h3_cuda_input_refiner_ffn",
        ),
        inherited,
    ):
        value = getattr(args, key, None)
        setattr(args, key, default if value is None else value)
    if (
        type(args.h3_cuda_input_refiner_sp) is not bool
        or args.h3_cuda_input_refiner_attention not in ("legacy", "cudnn")
        or args.h3_cuda_input_refiner_ffn not in ("bf16", "fp8")
    ):
        raise ValueError("Invalid input Token Refiner SP/attention/FFN configuration")
    effective = (
        args.h3_cuda_input_refiner_sp,
        args.h3_cuda_input_refiner_attention,
        args.h3_cuda_input_refiner_ffn,
    )
    args.h3_cuda_input_refiner = next(
        (name for name, value in legacy.items() if value == effective), "custom"
    )
    if getattr(args, "h3_cuda_audio_decoder_parallel", None) is None:
        args.h3_cuda_audio_decoder_parallel = "long" if full else "none"
    residency = getattr(args, "h3_cuda_dit_residency", None) or "inherit"
    if residency not in ("inherit", "resident", "conditioning", "reference"):
        raise ValueError("Invalid H3 DiT residency policy")
    if residency != "inherit":
        expected = (residency != "resident", residency == "reference")
        actual = (args.h3_cuda_conditioning_offload, args.h3_cuda_reference_offload)
        if any(actual) and actual != expected:
            raise ValueError(
                "DiT residency conflicts with conditioning/reference offload flags"
            )
        args.h3_cuda_conditioning_offload, args.h3_cuda_reference_offload = expected
    args.h3_cuda_dit_residency = (
        "reference"
        if args.h3_cuda_reference_offload
        else "conditioning"
        if args.h3_cuda_conditioning_offload
        else "resident"
    )
    for key, env, default in (
        (
            "h3_cuda_phaa_fp8",
            "XGEN_PHAA_FP8",
            full and getattr(args, "h3_cuda_precision", "none") != "fp8-bf16",
        ),
        ("h3_cuda_phaa_direct_pack", "XGEN_PHAA_FP8_MIXED_STRIDES", full),
    ):
        value = getattr(args, key, None)
        explicit = value is not None
        if value is None:
            value = (
                default
                if full
                else backend == "cuda" and os.environ.get(env, "0") == "1"
            )
            if (
                key == "h3_cuda_phaa_fp8"
                and mode != "sp"
                and (getattr(args, "h3_cp_comm", "auto") != "auto")
            ):
                value = args.h3_cp_comm == "fp8"
            if (
                key == "h3_cuda_phaa_direct_pack"
                and full
                and (not args.h3_cuda_phaa_fp8)
            ):
                value = False
            if key == "h3_cuda_phaa_direct_pack" and mode == "tp_cp_shared":
                value = False
        if type(value) is not bool:
            raise ValueError(key + " must be boolean")
        setattr(args, key, value)
        if (explicit or full) and backend == "cuda":
            environment[env] = "1" if value else "0"
    cp_wire = getattr(args, "h3_cp_comm", "auto")
    if (
        mode != "sp"
        and cp_wire != "auto"
        and (args.h3_cuda_phaa_fp8 != (cp_wire == "fp8"))
    ):
        raise ValueError("Explicit PHAA FP8 setting conflicts with h3_cp_comm")
    if args.h3_cuda_phaa_direct_pack and (
        not args.h3_cuda_phaa_fp8 or mode == "tp_cp_shared"
    ):
        raise ValueError(
            "Direct PHAA packing requires FP8 PHAA; it is not applicable to shared TP8/CP8"
        )
    args._h3_cuda_env_updates = environment
    if backend != "cuda" and (args.h3_cuda_phaa_fp8 or args.h3_cuda_phaa_direct_pack):
        raise ValueError("H3 FP8 PHAA options require CUDA")
    return {name: getattr(args, name, None) for name in H3_CUDA_STARTUP_OPTIONS}


def bootstrap_h3_preset_environment(preset):
    """Fresh-process allocator/NCCL defaults; no request-side mutation."""
    if preset == "sage-fp8-full-parallel":
        for name, value in (
            ("NCCL_MAX_CTAS", "2"),
            ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ):
            os.environ.setdefault(name, value)


def h3_startup_key(args):
    """Only startup state; duration, prompt and reference media remain per request."""
    values = tuple(
        ((name, getattr(args, name, None)) for name in H3_CUDA_STARTUP_OPTIONS)
    )
    runtime = tuple(
        (
            (name, getattr(args, name, None))
            for name in (
                "sp",
                "text_encoder_tp",
                "dtype",
                "atten",
                "matmul_quant",
                "vae_matmul_quant",
                "phaa_num",
                "rope_fused",
                "transformer_path",
                "adaln_cache_file",
            )
            if name != "adaln_cache_file"
            or not getattr(args, "_h3_cuda_adaln_auto_paths", False)
        )
    )
    return (
        values
        + runtime
        + (
            (
                "_h3_residency_explicit",
                bool(getattr(args, "_h3_residency_explicit", False)),
            ),
            ("model", getattr(args, "model", None)),
            ("weights", getattr(args, "pretrained_model_name_or_path", None)),
            (
                "workflow",
                "ref2va" if getattr(args, "task_type", None) == "ref2va" else "fl2va",
            ),
        )
    )


def h3_needs_resolution(args):
    return (
        getattr(args, "h3_features", None) not in (None, "none")
        or getattr(args, "h3_cuda_adaln_mode", None) not in (None, "auto")
        or bool(getattr(args, "h3_cuda_adaln_dir", None))
        or (getattr(args, "h3_cuda_precision", "auto") != "none")
        or any((getattr(args, key, None) is not None for key in H3_CUDA_BOOL_OPTIONS))
        or (getattr(args, "h3_cuda_preset", None) not in (None, "none"))
        or (getattr(args, "h3_cuda_dit_residency", None) not in (None, "inherit"))
        or (getattr(args, "h3_cuda_input_refiner", None) not in (None, "none"))
        or (
            getattr(args, "h3_cuda_input_refiner_attention", None)
            not in (None, "legacy")
        )
        or (getattr(args, "h3_cuda_input_refiner_ffn", None) not in (None, "bf16"))
        or (getattr(args, "h3_cuda_audio_decoder_parallel", None) not in (None, "none"))
        or bool(getattr(args, "h3_cuda_refiner_ff_chunk_size", 0))
        or bool(getattr(args, "h3_vae_torch_compile", False))
    )


def commit_h3_environment(args):
    """Call only after configuration validation, or before a fresh CLI process initializes CUDA."""
    for name, value in getattr(args, "_h3_cuda_env_updates", {}).items():
        os.environ[name] = value


H3_CONDITION_PARALLEL = (
    "h3_cuda_vision_sp",
    "h3_cuda_vae_encoder_parallel",
    "h3_cuda_audio_encoder_parallel",
    "h3_cuda_audio_decoder_parallel",
)


def condition_parallel_key(args):
    return tuple(
        (
            getattr(args, name, "none" if name.endswith("decoder_parallel") else False)
            for name in H3_CONDITION_PARALLEL
        )
    )


def resolve_h3_precision(args, backend):
    """Resolve the hardware-specific default without loading CUDA-only operators."""
    resolve_h3_options(args, backend)
    requested = getattr(args, "h3_cuda_precision", "auto")
    if requested not in ("auto", "none", "fp8-bf16", "fp8-sage"):
        raise ValueError("Unknown H3 CUDA precision profile")
    if requested == "auto":
        legacy = (
            any(
                (
                    getattr(args, name, None)
                    for name in (
                        "h3_nvfp4_checkpoint",
                        "h3_nvfp4_resident",
                        "h3_nvfp4_compile",
                        "transformer_path",
                        "adaln_cache_file",
                        "h3_turbo8",
                        "h3_turbo_sla_lora",
                        "h3_refiner",
                        "lora_path_list",
                        "fsdp",
                        "inf_vram_blocks_num",
                        "cache_dit",
                    )
                )
            )
            or getattr(args, "h3_attention", "legacy") != "legacy"
        )
        legacy = (
            legacy
            or getattr(args, "atten", "none") != "none"
            or getattr(args, "matmul_quant", "none") != "none"
            or (getattr(args, "dtype", "bf16") != "bf16")
            or (getattr(args, "phaa_num", 0) is None)
        )
        requested = (
            "fp8-bf16"
            if backend == "cuda"
            and getattr(args, "model", None) == "MiniMax-H3"
            and (getattr(args, "task_type", None) in ("t2va", "fl2va"))
            and (not legacy)
            else "none"
        )
    if requested != "none" and (
        backend != "cuda" or getattr(args, "model", None) != "MiniMax-H3"
    ):
        raise ValueError(
            f"{requested} requires CUDA MiniMax-H3; use auto or none on NPU and other models"
        )
    portable = requested == "none"
    if portable:
        validate_h3_features(args, backend)
    linear_tp = getattr(args, "h3_cuda_linear_tp", 1)
    if type(linear_tp) is not int or linear_tp not in (1, 2):
        raise ValueError("h3_cuda_linear_tp must be 1 or 2")
    if linear_tp == 2 and (
        backend != "cuda"
        or requested not in ("fp8-bf16", "fp8-sage")
        or getattr(args, "sp", 1) != 8
        or (getattr(args, "h3_parallel_mode", "sp") != "sp")
    ):
        raise ValueError(
            "h3_cuda_linear_tp=2 requires CUDA H3 FP8 and original SP8 mode"
        )
    if (
        getattr(args, "h3_cuda_local_output", False)
        and (not portable)
        and (backend != "cuda")
    ):
        raise ValueError(
            "h3_cuda_local_output requires a CUDA H3 FP8 precision profile"
        )
    if (
        getattr(args, "h3_cuda_local_input", False)
        and (not portable)
        and (backend != "cuda")
    ):
        raise ValueError("h3_cuda_local_input requires a CUDA H3 FP8 precision profile")
    if getattr(args, "h3_cuda_conditioning_offload", False) and (
        not portable
        and backend != "cuda"
        or getattr(args, "task_type", None) != "ref2va"
    ):
        raise ValueError(
            "h3_cuda_conditioning_offload requires an explicit CUDA FP8 Ref2VA profile"
        )
    chunk = getattr(args, "h3_cuda_refiner_ff_chunk_size", 0)
    if (
        type(chunk) is not int
        or chunk < 0
        or (chunk and (chunk < 128 or chunk > 32768 or chunk % 128))
    ):
        raise ValueError(
            "h3_cuda_refiner_ff_chunk_size must be zero or 128..32768 in multiples of 128"
        )
    if chunk and (
        not portable
        and backend != "cuda"
        or getattr(args, "task_type", None) != "ref2va"
    ):
        raise ValueError(
            "h3_cuda_refiner_ff_chunk_size requires an explicit CUDA FP8 Ref2VA profile"
        )
    input_refiner = getattr(args, "h3_cuda_input_refiner", "none")
    if input_refiner not in ("none", "cudnn-sp", "cudnn-sp-fp8", "custom"):
        raise ValueError("Unknown h3_cuda_input_refiner profile")
    if (
        input_refiner != "none"
        and (not portable)
        and (
            backend != "cuda"
            or getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va")
            or (
                getattr(args, "h3_parallel_mode", "sp") == "sp"
                and getattr(args, "sp", 1) != 8
            )
        )
    ):
        raise ValueError("h3_cuda_input_refiner requires a CUDA FP8 H3 SP8 profile")
    if (
        getattr(args, "h3_cuda_precompute_conditioning", False)
        and (not portable)
        and (
            backend != "cuda"
            or getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va")
            or (
                getattr(args, "h3_parallel_mode", "sp") == "sp"
                and getattr(args, "sp", 1) != 8
            )
        )
    ):
        raise ValueError(
            "h3_cuda_precompute_conditioning requires a CUDA FP8 H3 SP8 profile"
        )
    parallel_key = condition_parallel_key(args)
    if parallel_key[-1] not in ("none", "long"):
        raise ValueError("Unknown h3_cuda_audio_decoder_parallel profile")
    if (any(parallel_key[:3]) or parallel_key[-1] != "none") and (
        not portable
        and (
            backend != "cuda"
            or getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va")
            or (
                getattr(args, "h3_parallel_mode", "sp") == "sp"
                and getattr(args, "sp", 1) != 8
            )
        )
    ):
        raise ValueError("H3 condition parallelism requires a CUDA FP8 H3 SP8 profile")
    if getattr(args, "h3_cuda_reference_offload", False) and (
        not portable
        and backend != "cuda"
        or getattr(args, "task_type", None) != "ref2va"
        or (not getattr(args, "h3_cuda_conditioning_offload", False))
    ):
        raise ValueError(
            "h3_cuda_reference_offload requires CUDA FP8 Ref2VA with h3_cuda_conditioning_offload"
        )
    from x_diffusers.adaptor.validation import resolve_h3_cuda_adaln_assets as resolve_assets

    resolve_assets(args, backend, requested)
    args.h3_cuda_precision = requested
    resolve_vae_backend(args)
    if requested in ("fp8-bf16", "fp8-sage"):
        if getattr(args, "phaa_num", 0) == 0:
            args.phaa_num = 2
        args.h3_cuda_vae_sp = True
    return requested


def validate_h3_precision(args, backend):
    profile = getattr(args, "h3_cuda_precision", "none")
    if profile == "none":
        return
    new_parallel = getattr(args, "h3_parallel_mode", "sp") in (
        "tp_cp",
        "tp_cp_shared",
        "tp_sp",
    )
    if new_parallel:
        resolve_parallel_config(
            args,
            backend,
            _distributed().get_world_size() if _distributed().is_initialized() else 1,
        )
    tasks = ("t2va", "fl2va", "ref2va")
    if (
        backend != "cuda"
        or args.model != "MiniMax-H3"
        or args.dtype != "bf16"
        or (args.task_type not in tasks)
        or (not new_parallel and args.sp != 8)
        or (not _distributed().is_initialized())
        or (_distributed().get_world_size() != 8)
        or (args.text_encoder_tp != 8)
    ):
        layout = (
            "world=text TP=8 and a valid DiT TP/CP layout"
            if new_parallel
            else "world=SP=text TP=8"
        )
        raise ValueError(f"{profile} requires CUDA BF16 {'/'.join(tasks)} and {layout}")
    conflicts = (
        "h3_nvfp4_checkpoint",
        "h3_nvfp4_resident",
        "h3_nvfp4_compile",
        "h3_turbo8",
        "h3_turbo_sla_lora",
        "h3_refiner",
        "lora_path_list",
        "fsdp",
        "inf_vram_blocks_num",
        "cache_dit",
    )
    if (
        any((getattr(args, name, None) for name in conflicts))
        or args.atten != "none"
        or args.matmul_quant != "none"
    ):
        raise ValueError(
            f"{profile} cannot combine with packed weights, LoRA, caches or other attention/quantization recipes; use --h3_cuda_precision none"
        )
    from x_diffusers.adaptor.validation import validate_h3_cuda_adaln as validate_options

    validate_options(args)
    if (
        not new_parallel
        and args.h3_cuda_precision == "fp8-bf16"
        and getattr(args, "h3_cuda_phaa_fp8", os.environ.get("XGEN_PHAA_FP8") == "1")
    ):
        raise ValueError("fp8-bf16 requires BF16 communication; unset XGEN_PHAA_FP8")
    if type(args.phaa_num) is not int or args.phaa_num < 1:
        raise ValueError(f"{profile} requires PHAA; the recommended value is 2")
    if not getattr(args, "h3_cuda_vae_sp", False):
        raise ValueError(f"{profile} requires --h3_cuda_vae_sp")


def validate_h3_manager(manager, args, backend):
    """A resident HTTP worker cannot silently change its precision recipe."""
    profile = resolve_h3_precision(args, backend)
    validate_h3_vae(args, backend)
    if manager.pipe is not None:
        installed = getattr(manager.pipe, "_h3_vae_decode_backend", None)
        selected = getattr(args, "h3_vae_backend", "legacy")
        if selected != getattr(manager.pipe, "_h3_vae_backend_name", "legacy"):
            raise ValueError("Restart the service to change H3 VAE decode backend")
        if (installed is not None) != (selected == "native-stream") or (
            installed is not None and str(installed.path) != str(args.h3_vae_model_path)
        ):
            raise ValueError(
                "Restart the service to change H3 VAE decode backend or checkpoint"
            )
        if installed is not None and bool(
            getattr(installed, "torch_compile_enabled", False)
        ) != bool(getattr(args, "h3_vae_torch_compile", False)):
            raise ValueError("Restart the service to change H3 VAE Torch Compiler")
    pair_tp = getattr(args, "h3_cuda_linear_tp", 1)
    if manager.pipe is not None and pair_tp != getattr(
        manager, "_h3_cuda_linear_tp_key", 1
    ):
        raise ValueError("Restart the service to change H3 linear TP")
    condition_key = condition_parallel_key(args)
    if manager.pipe is not None and condition_key != getattr(
        manager, "_h3_condition_parallel_key", (False, False, False, "none")
    ):
        raise ValueError("Restart the service to change H3 condition parallelism")
    input_refiner = getattr(args, "h3_cuda_input_refiner", "none")
    precompute = bool(getattr(args, "h3_cuda_precompute_conditioning", False))
    if manager.pipe is not None and precompute != getattr(
        manager, "_h3_precompute_conditioning_key", False
    ):
        raise ValueError(
            "Restart the service to change h3_cuda_precompute_conditioning"
        )
    if manager.pipe is not None and input_refiner != getattr(
        manager, "_h3_cuda_input_refiner_key", "none"
    ):
        raise ValueError("Restart the service to change h3_cuda_input_refiner")
    refiner_chunk = getattr(args, "h3_cuda_refiner_ff_chunk_size", 0)
    reference_offload = bool(getattr(args, "h3_cuda_reference_offload", False))
    if manager.pipe is not None and reference_offload != getattr(
        manager, "_h3_cuda_reference_offload_key", False
    ):
        raise ValueError("Restart the service to change h3_cuda_reference_offload")
    if manager.pipe is not None and refiner_chunk != getattr(
        manager, "_h3_cuda_refiner_ff_chunk_key", 0
    ):
        raise ValueError("Restart the service to change h3_cuda_refiner_ff_chunk_size")
    local_output = bool(getattr(args, "h3_cuda_local_output", False))
    if manager.pipe is not None and local_output != getattr(
        manager, "_h3_cuda_local_output_key", False
    ):
        raise ValueError("Restart the service to change h3_cuda_local_output")
    conditioning_offload = bool(getattr(args, "h3_cuda_conditioning_offload", False))
    if manager.pipe is not None and conditioning_offload != getattr(
        manager, "_h3_cuda_conditioning_offload_key", False
    ):
        raise ValueError("Restart the service to change h3_cuda_conditioning_offload")
    parallel_key = resolve_parallel_config(
        args,
        backend,
        _distributed().get_world_size() if _distributed().is_initialized() else 1,
    )
    previous_parallel = getattr(manager, "_h3_parallel_key", H3ParallelConfig())
    if manager.pipe is not None and parallel_key != previous_parallel:
        raise ValueError(
            "Restart the service to change H3 TP/CP layout or transport dtype"
        )
    previous = getattr(manager, "_h3_cuda_precision_key", None)
    if (
        manager.pipe is not None
        and previous != profile
        and (
            previous in ("fp8-bf16", "fp8-sage") or profile in ("fp8-bf16", "fp8-sage")
        )
    ):
        raise ValueError("Restart the service to change h3_cuda_precision")
    if profile != "none":
        validate_h3_precision(args, backend)
        if manager.pipe is not None and bool(
            getattr(args, "adaln_cache_file", None)
        ) != bool(
            getattr(
                getattr(manager.pipe, "transformer", None),
                "_h3_cuda_adaln_table",
                False,
            )
        ):
            raise ValueError("Restart the manager to change CUDA AdaLN table mode")
    key = h3_startup_key(args)
    portable_now = h3_features_enabled(args)
    previous_portable_key = getattr(manager, "_h3_portable_startup_key", None)
    if (
        manager.pipe is not None
        and portable_now
        and (previous not in ("fp8-bf16", "fp8-sage"))
    ):
        if previous_portable_key is None or previous_portable_key != key:
            raise ValueError(
                "Restart the service to enable or change portable H3 startup options"
            )
    cuda_resident = (
        profile in ("fp8-bf16", "fp8-sage")
        or previous in ("fp8-bf16", "fp8-sage")
        or portable_now
        or getattr(manager, "_h3_portable_enabled", False)
    )
    manager._h3_portable_enabled = portable_now
    if (
        cuda_resident
        and manager.pipe is not None
        and (getattr(manager, "_h3_cuda_options_key", key) != key)
    ):
        raise ValueError(
            "Restart the manager/service to change H3 CUDA startup options"
        )
    commit_h3_environment(args)
    if cuda_resident:
        manager._h3_cuda_options_key = key
    if getattr(args, "model", None) == "MiniMax-H3":
        manager._h3_portable_startup_key = key
    manager._h3_cuda_linear_tp_key = pair_tp
    manager._h3_cuda_precision_key = profile
    manager._h3_condition_parallel_key = condition_key
    manager._h3_parallel_key = parallel_key
    manager._h3_cuda_local_output_key = local_output
    manager._h3_cuda_conditioning_offload_key = conditioning_offload
    manager._h3_cuda_refiner_ff_chunk_key = refiner_chunk
    manager._h3_cuda_reference_offload_key = reference_offload
    manager._h3_cuda_input_refiner_key = input_refiner
    manager._h3_precompute_conditioning_key = precompute
    logger.info("H3 CUDA effective startup options=%s", dict(key))


def normalize_h3_mapping(values):
    result = dict(values)
    for public, legacy in H3_FEATURE_ALIASES.items():
        if public in result:
            if legacy in result and result[legacy] != result[public]:
                raise ValueError(
                    "Conflicting H3 option aliases: " + public + " / " + legacy
                )
            result[legacy] = result.pop(public)
    return result


def resolve_h3_feature_defaults(args):
    if getattr(args, "h3_features", None) not in (None, "none", "full-parallel"):
        raise ValueError("Unknown portable H3 feature preset")
    if getattr(args, "h3_features", None) != "full-parallel":
        return
    for name in H3_FEATURE_BOOL_FIELDS:
        key = "h3_cuda_" + name
        if (
            name not in ("conditioning_offload", "reference_offload")
            and getattr(args, key, None) is None
        ):
            setattr(args, key, True)
    for name, value in (
        ("input_refiner_attention", "legacy"),
        ("input_refiner_ffn", "bf16"),
        ("audio_decoder_parallel", "long"),
        ("dit_residency", "resident"),
    ):
        key = "h3_cuda_" + name
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def h3_features_enabled(args):
    return (
        any(
            (getattr(args, "h3_cuda_" + name, False) for name in H3_FEATURE_BOOL_FIELDS)
        )
        or getattr(args, "h3_cuda_audio_decoder_parallel", None) == "long"
        or bool(getattr(args, "h3_cuda_refiner_ff_chunk_size", 0))
        or (getattr(args, "h3_cuda_input_refiner", None) not in (None, "none"))
        or bool(getattr(args, "_h3_residency_explicit", False))
        or (getattr(args, "h3_parallel_mode", "sp") != "sp")
        or (getattr(args, "h3_features", None) == "full-parallel")
    )


def validate_h3_features(args, backend):
    if not h3_features_enabled(args):
        return
    if backend not in ("cuda", "npu") or getattr(args, "model", None) != "MiniMax-H3":
        raise ValueError("Portable H3 scheduling requires CUDA or NPU MiniMax-H3")
    if getattr(args, "dtype", "bf16") != "bf16":
        raise ValueError(
            "Portable H3 scheduling currently requires BF16 module interfaces"
        )
    if getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va"):
        raise ValueError("Portable H3 scheduling requires an H3 workflow")
    if getattr(args, "h3_parallel_mode", "sp") == "sp" and getattr(
        args, "sp", 1
    ) not in (2, 4, 8):
        raise ValueError("Portable H3 component parallelism requires SP2/4/8")
    if getattr(args, "fsdp", None) or int(getattr(args, "inf_vram_blocks_num", 0) or 0):
        raise ValueError(
            "Portable scheduling owns residency; disable FSDP/infinite-vram hooks"
        )
    if getattr(args, "h3_refiner", False) and any(
        (
            getattr(args, "h3_cuda_" + name, False)
            for name in ("conditioning_offload", "reference_offload")
        )
    ):
        raise ValueError(
            "The 2K Refiner coordinator owns residency; disable standalone conditioning/reference offload"
        )
    if backend == "npu":
        if getattr(args, "h3_cuda_input_refiner_attention", "legacy") != "legacy":
            raise ValueError(
                "cuDNN input refiner attention is CUDA-only; use legacy on NPU"
            )
        if getattr(args, "h3_cuda_input_refiner_ffn", "bf16") != "bf16":
            raise ValueError("FP8 input-refiner GEMM is CUDA-only; use bf16 on NPU")


def resolve_image_text_options(args, backend):
    """Default Image decoder W8A8 on NPU, preserving explicit precision choices."""
    if getattr(args, "image_text_int8", None) is None:
        args.image_text_int8 = backend == "npu"
    if args.image_text_int8 and backend != "npu":
        raise ValueError("Image text INT8 requires NPU")


def validate_h3_image_options(args, world_size):
    dit_tp = getattr(args, "image_dit_tp", 1)
    if (
        world_size not in (1, 2, 4)
        or args.sp * dit_tp != world_size
        or (args.sp > 1 and dit_tp > 1)
    ):
        raise ValueError("Use pure SP1/SP2/SP4 or pure DiT TP2/TP4")
    if args.text_encoder_tp not in (1, world_size):
        raise ValueError("text_encoder_tp must be 1 or the distributed world size")
    text_int8 = getattr(args, "image_text_int8", False)
    text_resident = getattr(args, "image_text_resident", False)
    if (text_int8 or text_resident) and args.text_encoder_tp != world_size:
        raise ValueError(
            "Image text quantization/residency requires text_encoder_tp equal to world size"
        )
    if args.num_inference_steps != 6:
        raise ValueError("MiniMax-H3-Image has exactly six PDD sampling steps")
    if getattr(args, "image_vae_decode_batch_size", 1) < 1:
        raise ValueError("VAE decode batch size must be at least one")
    if getattr(args, "image_vae_gather_once", False) and (
        world_size == 1 or not getattr(args, "image_vae_parallel", False)
    ):
        raise ValueError(
            "--image_vae_gather_once requires --image_vae_parallel on multiple devices"
        )
    if getattr(args, "image_adaln_table_file", None) and getattr(
        args, "image_adaln_table_save", None
    ):
        raise ValueError("Choose either reading or saving an AdaLN table")
    if args.width <= 0 or args.height <= 0 or args.width * args.height > 20 * 1024**2:
        raise ValueError("Image dimensions must be positive and at most 20 megapixels")
    if len(args.image_path_list or []) > 16:
        raise ValueError(
            "MiniMax-H3-Image supports at most 16 ordered reference images"
        )
    if not 1024 <= args.min_generation_area <= 20 * 1024**2:
        raise ValueError(
            "min_generation_area must be between 1024 pixels and 20 megapixels"
        )
    for path in args.image_path_list or []:
        if not Path(path).is_file():
            raise ValueError(f"Reference image does not exist: {path}")


def resolve_h3_npu_features(args):
    """CLI list overrides environment; individual flags override that list."""
    value = getattr(args, "h3_opts", None)
    if value is None:
        value = os.environ.get("H3_OPT_FEATURES", "")
        if (
            "H3_OPT_FEATURES" not in os.environ
            and os.environ.get("H3_OPT_X1_X5") == "1"
        ):
            value = "shared_qkv,merge_quant"
    selected = set()
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            selected.update(H3_NPU_FUSION_FEATURES)
        elif token in ("none", "off"):
            selected.clear()
        elif (
            token.startswith("-")
            and token[1:].replace("-", "_") in H3_NPU_FUSION_FEATURES
        ):
            selected.discard(token[1:].replace("-", "_"))
        elif token.replace("-", "_") in H3_NPU_FUSION_FEATURES:
            selected.add(token.replace("-", "_"))
        else:
            raise ValueError(f"Unknown H3 fusion feature: {token!r}")
    for name in H3_NPU_FUSION_FEATURES:
        override = getattr(args, "h3_opt_" + name, None)
        if override is True:
            selected.add(name)
        elif override is False:
            selected.discard(name)
    args.h3_fused_ops = tuple(
        (name for name in H3_NPU_FUSION_FEATURES if name in selected)
    )
    return args.h3_fused_ops


def validate_h3_npu_fusions(args, backend):
    features = resolve_h3_npu_features(args)
    if features and (backend != "npu" or args.model != "MiniMax-H3"):
        raise ValueError("H3 fusions selected by --h3-opts require NPU MiniMax-H3")
    if getattr(args, "atten", "none") == "vc" and backend != "npu":
        raise ValueError("VC-Attention requires Ascend NPU")
    return features


def validate_h3_resident_args(initial, requested):
    """A resident DiT cannot silently change its groups, weights or wire dtype."""
    if initial is None:
        return
    defaults = dict(
        h3_parallel_mode="sp",
        h3_dit_tp=1,
        sp=1,
        h3_tp_comm="bf16",
        h3_cp_comm="auto",
        h3_cuda_precision="auto",
        model=None,
        pretrained_model_name_or_path=None,
        h3_tp_comm_chunk_elements=262144,
        h3_parallel_warmup_mib=0,
    )
    changed = any(
        (
            getattr(initial, key, default) != getattr(requested, key, default)
            for key, default in defaults.items()
        )
    )
    changed |= (getattr(initial, "task_type", None) == "ref2va") != (
        getattr(requested, "task_type", None) == "ref2va"
    )
    if changed:
        raise ValueError(
            "Restart the manager to change H3 TP/CP layout, transport, precision or workflow"
        )


H3ParallelConfig = H3ParallelSettings


def resolve_parallel_config(args, backend, world_size):
    mode = getattr(args, "h3_parallel_mode", "sp")
    tp = getattr(args, "h3_dit_tp", 1)
    cp = getattr(args, "sp", 1)
    tp_comm = getattr(args, "h3_tp_comm", "bf16")
    cp_comm = getattr(args, "h3_cp_comm", "auto")
    chunk_elements = getattr(args, "h3_tp_comm_chunk_elements", 262144)
    warmup_mib = getattr(args, "h3_parallel_warmup_mib", 0)
    if type(warmup_mib) is not int or not 0 <= warmup_mib <= 1024:
        raise ValueError("h3_parallel_warmup_mib must be an integer in [0, 1024]")
    if (
        type(chunk_elements) is not int
        or not 128 <= chunk_elements <= 4194304
        or chunk_elements % 128
    ):
        raise ValueError(
            "h3_tp_comm_chunk_elements must be a multiple of 128 in [128, 4194304]"
        )
    if (
        mode not in ("sp", "tp_cp", "tp_cp_shared", "tp_sp")
        or tp_comm not in ("bf16", "fp8")
        or cp_comm not in ("auto", "bf16", "fp8")
    ):
        raise ValueError("Invalid H3 parallel mode or communication dtype")
    if mode == "sp":
        if (tp, tp_comm, cp_comm, chunk_elements, warmup_mib) != (
            1,
            "bf16",
            "auto",
            262144,
            0,
        ):
            raise ValueError(
                "H3 DiT TP/communication options require --h3_parallel_mode tp_cp"
            )
        return H3ParallelSettings()
    if backend not in ("cuda", "npu") or getattr(args, "model", None) != "MiniMax-H3":
        raise ValueError("tp_cp requires CUDA/NPU MiniMax-H3")
    if mode == "tp_cp_shared":
        if (
            type(tp) is not int
            or type(cp) is not int
            or (tp, cp, world_size) != (8, 8, 8)
        ):
            raise ValueError(
                "tp_cp_shared requires TP8 and CP8 on the same eight ranks"
            )
    elif mode == "tp_sp":
        if (
            type(tp) is not int
            or type(cp) is not int
            or (tp, cp, world_size) != (2, 8, 8)
        ):
            raise ValueError(
                "tp_sp requires TP2 projections and SP8 attention on eight ranks"
            )
    elif (
        type(tp) is not int
        or type(cp) is not int
        or tp not in (1, 2, 4)
        or (cp < 1)
        or (tp * cp != world_size)
    ):
        raise ValueError(
            "tp_cp requires DiT TP=1, 2 or 4 and TP * --sp = world size (DP=CFG=1)"
        )
    if getattr(args, "ring_degree", None) not in (None, 1) or getattr(
        args, "ulysses_degree", None
    ) not in (None, cp):
        raise ValueError("tp_cp supports pure Ulysses only; Ring must be 1")
    if getattr(args, "task_type", None) not in ("t2va", "fl2va", "ref2va"):
        raise ValueError("tp_cp supports T2VA, FL2VA and Ref2VA")
    conflicts = (
        "h3_refiner",
        "joint",
        "fsdp",
        "inf_vram_blocks_num",
        "h3_nvfp4_checkpoint",
        "h3_nvfp4_resident",
        "h3_nvfp4_compile",
        "h3_turbo8",
        "h3_turbo_sla_lora",
        "adaln_cache_file",
        "transformer_path",
        "lora_path_list",
        "cache_dit",
    )
    if backend == "npu":
        conflicts = tuple(
            (k for k in conflicts if k not in ("adaln_cache_file", "transformer_path"))
        )
        if getattr(args, "matmul_quant", "none") != "none":
            raise ValueError(
                "NPU DiT TP currently uses native BF16 Linear; INT8 TP needs a separate operator adapter"
            )
        if tp_comm != "bf16" or cp_comm not in ("auto", "bf16"):
            raise ValueError(
                "NPU TP/CP currently uses BF16 communication; FP8 packet kernels are CUDA-only"
            )
    if any((getattr(args, key, None) for key in conflicts)):
        raise ValueError(
            "tp_cp does not support Refiner, packed weights, LoRA or external offload/cache recipes"
        )
    if backend == "cuda" and getattr(args, "h3_cuda_precision", "none") not in (
        "fp8-bf16",
        "fp8-sage",
        "none",
    ):
        raise ValueError(
            "tp_cp requires an explicit none, fp8-bf16 or fp8-sage precision profile"
        )
    if cp_comm == "auto":
        phaa = getattr(args, "h3_cuda_phaa_fp8", None)
        phaa = os.environ.get("XGEN_PHAA_FP8") == "1" if phaa is None else phaa
        cp_comm = (
            tp_comm
            if mode == "tp_cp_shared"
            else "fp8"
            if backend == "cuda" and phaa
            else "bf16"
        )
    if mode == "tp_cp_shared" and cp_comm != tp_comm:
        raise ValueError(
            "Shared TP/CP fuses layout boundaries and requires matching TP/CP wire dtypes"
        )
    shared_ignored = (
        "XGEN_PHAA_GATHER_PRIORITY",
        "XGEN_PHAA_DUAL_COMM",
        "XGEN_PHAA_DEVICE_JOIN",
        "XGEN_PHAA_ASYNC_EXCHANGE",
        "XGEN_PHAA_DIRECT_RETURN",
        "XGEN_PHAA_FP8_ROW_TILING",
        "XGEN_PHAA_FUSED_RETURN_PACK",
        "XGEN_PHAA_DIRECT_GATHER",
        "XGEN_PHAA_FP8_MIXED_STRIDES",
    )
    if mode == "tp_cp_shared" and (
        any(
            (
                getattr(args, "h3_cuda_phaa_direct_pack", None)
                if key == "XGEN_PHAA_FP8_MIXED_STRIDES"
                and getattr(args, "h3_cuda_phaa_direct_pack", None) is not None
                else os.environ.get(key) == "1"
                for key in shared_ignored
            )
        )
        or os.environ.get("XGEN_PHAA_CHUNK_ORDER", "natural") != "natural"
    ):
        raise ValueError(
            "Shared TP/CP fuses QKV redistribution; standalone PHAA transport options are unsupported"
        )
    if cp == 1 and cp_comm == "fp8":
        raise ValueError("FP8 CP transport requires --sp > 1")
    if tp == 1 and tp_comm == "fp8":
        raise ValueError("FP8 TP transport requires --h3_dit_tp 2 or 4")
    if chunk_elements != 262144 and (tp == 1 or tp_comm != "fp8"):
        raise ValueError(
            "Non-default TP packet size requires multi-rank FP8 TP transport"
        )
    if any(
        (
            os.environ.get(key) == "1"
            for key in (
                "XGEN_PHAA_DUAL_COMM",
                "XGEN_PHAA_ASYNC_EXCHANGE",
                "XGEN_PHAA_DIRECT_RETURN",
                "XGEN_PHAA_DIRECT_GATHER",
            )
        )
    ):
        raise ValueError("tp_cp currently supports the ordinary PHAA transport only")
    return H3ParallelSettings(
        mode, tp, cp, tp_comm, cp_comm, chunk_elements, warmup_mib
    )


H3_BASE_MAX_DURATION_S = 15


H3_MAX_PROMPT_LENGTH = 7000


def h3_prompt_length(text):
    """Match JavaScript string.length, including surrogate pairs and lone surrogates."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def h3_refiner_profile(seconds):
    if type(seconds) is not int or seconds not in (5, 10, 15):
        raise ValueError("H3 Refiner duration must be an integer: 5, 10 or 15 seconds")
    from x_diffusers.framework.h3_config import refiner_shape

    return refiner_shape(seconds)


def h3_refiner_model_key(args):
    """Request shapes are not weight-affecting; changing workflow/assets is."""
    fields = (
        "model",
        "task_type",
        "pretrained_model_name_or_path",
        "transformer_path",
        "adaln_cache_file",
        "num_inference_steps",
        "dtype",
        "sp",
        "text_encoder_tp",
        "matmul_quant",
        "vae_matmul_quant",
        "atten",
        "phaa_num",
        "rope_fused",
        "h3_refiner_weights",
        "h3_refiner_prepared",
        "h3_refiner_loading",
        "h3_attention",
        "h3_veda_predictor",
        "h3_turbo8",
        "h3_nvfp4_checkpoint",
        "h3_fused_ops",
    )
    return tuple((getattr(args, name, None) for name in fields))


def validate_h3_refiner_settings(args):
    if not getattr(args, "h3_refiner", False):
        return None
    if args.model != "MiniMax-H3" or args.task_type not in ("t2va", "fl2va", "ref2va"):
        raise ValueError("H3 Refiner requires MiniMax-H3 t2va/fl2va/ref2va")
    cuda = os.environ.get("XGEN_DEVICE", "npu") == "cuda"
    if args.sp != 8 or args.dtype != "bf16":
        raise ValueError("Experimental H3 Refiner requires SP8 and BF16")
    if cuda and (
        args.task_type != "fl2va"
        or not getattr(args, "h3_nvfp4_checkpoint", None)
        or getattr(args, "h3_nvfp4_resident", False)
        or getattr(args, "h3_nvfp4_compile", False)
    ):
        raise ValueError(
            "CUDA H3 Refiner requires a packed FL2VA Base checkpoint with staged residency"
        )
    if cuda and args.atten not in ("none", "sage", "sparge"):
        raise ValueError(
            "Experimental CUDA H3 Refiner supports --atten none, sage or sparge"
        )
    if not cuda and args.atten != "laser":
        raise ValueError("Experimental NPU H3 Refiner requires Laser attention")
    steps = (8,) if getattr(args, "h3_turbo8", False) else (20, 50)
    if (
        type(args.num_inference_steps) is not int
        or args.num_inference_steps not in steps
    ):
        raise ValueError("Base schedule does not match the selected inference profile")
    if (args.width, args.height) != (2560, 1440) or args.save_fps != 24:
        raise ValueError(
            "Experimental H3 Refiner supports landscape 2560x1440 at 24fps only"
        )
    if not 1 <= args.phaa_num <= 7:
        raise ValueError("H3 Refiner requires phaa_num in [1,7]")
    if cuda and args.matmul_quant != "none":
        raise ValueError("Experimental CUDA H3 Refiner requires --matmul_quant none")
    if args.matmul_quant not in ("none", "int8") or args.vae_matmul_quant != "none":
        raise ValueError(
            "Base supports BF16/INT8; Refiner and VAE require BF16 (vae_matmul_quant=none)"
        )
    if args.vae_lightning != "none" or int(args.inf_vram_blocks_num) != 0:
        raise ValueError(
            "H3 Refiner owns staged residency: use vae_lightning=none and inf_vram_blocks_num=0"
        )
    if any(
        (
            getattr(args, "adopt_sr", False),
            getattr(args, "cache_dit", False),
            getattr(args, "h3_turbo_sla_lora", None),
            getattr(args, "lora_path_list", None),
            getattr(args, "ten_second", False),
            getattr(args, "frame_interpolation", False),
            getattr(args, "frame_interpolation_sr", False),
            getattr(args, "fsdp", None),
            getattr(args, "save_memory", False),
            getattr(args, "turbo_mode", "default") != "default",
        )
    ):
        raise ValueError(
            "H3 Refiner cannot be combined with SeedVR2, cache, Turbo/SLA, extra LoRA or interpolation"
        )
    for name in ("h3_refiner_weights", "h3_refiner_prepared"):
        value = getattr(args, name, None)
        if not value or not Path(value).is_dir():
            raise ValueError(f"{name} must name an existing local directory")
    return h3_refiner_profile(args.h3_refiner_seconds)


def validate_h3_refiner_inputs(args):
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not isinstance(args.prompt, str) or not 0 < len(args.prompt.strip()) <= 32768:
        raise ValueError("prompt must contain 1..32768 characters")
    refs = getattr(args, "reference_paths", None) or []
    first, last = (
        getattr(args, "keyframe_image", ""),
        getattr(args, "keyframe_last_image", ""),
    )
    if not isinstance(refs, (list, tuple)) or any(
        (not isinstance(p, str) for p in refs)
    ):
        raise ValueError("reference_paths must be a list of media paths")
    if not isinstance(first, str) or not isinstance(last, str):
        raise ValueError("Keyframes must be image paths")
    if args.task_type == "t2va" and (refs or first or last):
        raise ValueError("t2va does not accept images")
    if args.task_type == "fl2va" and (not first or last or refs):
        raise ValueError("Initial fl2va support requires one first-frame image only")
    if args.task_type == "ref2va":
        if first or last:
            raise ValueError("Ref2VA does not accept first or last keyframes")
        from x_diffusers.adaptor.validation import h3_reference_conditioning as reference_conditioning

        reference_conditioning(args)
        return
    paths = [first] if first else []
    if len(set(paths)) != len(paths):
        raise ValueError("Reference image paths must be distinct")
    from PIL import Image

    for path in paths:
        with Image.open(path) as image:
            if min(image.size) < 1 or image.width * image.height > 4096 * 4096:
                raise ValueError("Reference images must contain 1..4096*4096 pixels")
            image.verify()


def validate_h3_turbo8(args):
    if (args.model, args.task_type, args.dtype, args.sp, args.atten) != (
        "MiniMax-H3",
        "t2va",
        "bf16",
        8,
        "laser",
    ):
        raise ValueError("Turbo8 requires MiniMax-H3 T2VA, BF16, SP8 and atten=laser")
    expected = (2560, 1440) if getattr(args, "h3_refiner", False) else (1344, 768)
    if (
        (args.width, args.height) != expected
        or args.frames != 124
        or args.save_fps != 24
        or (args.num_inference_steps != 8)
        or (getattr(args, "h3_refiner_seconds", 5) != 5)
    ):
        raise ValueError(
            "Turbo8 requires 8 denoising steps and the 5s 768P or 2K profile"
        )
    if getattr(args, "h3_attention", "legacy") not in ("legacy", "veda"):
        raise ValueError("Turbo8 supports Laser and experimental Veda")
    if any(
        (
            getattr(args, name, None)
            for name in (
                "h3_turbo_sla_lora",
                "lora_path_list",
                "fuse_lora",
                "cache_dit",
                "keyframe_image",
                "keyframe_last_image",
                "reference_paths",
                "adopt_sr",
            )
        )
    ):
        raise ValueError(
            "Turbo8 requires prepared assets without additional adapters, caches or media"
        )


def validate_h3_turbo8_assets(args):
    """Require matching prepared transformer and AdaLN assets for Turbo8."""
    import json
    from safetensors import safe_open
    from x_diffusers.framework.h3_config import H3_TURBO8_LORA_SHA256

    manifest = Path(args.transformer_path or "") / "turbo8.json"
    if not args.transformer_path or not manifest.is_file() or not args.adaln_cache_file:
        raise ValueError("Turbo8 requires a prepared transformer and matching AdaLN cache")
    model = json.loads(manifest.read_text(encoding="utf-8"))
    with safe_open(args.adaln_cache_file, framework="pt") as cache:
        metadata = cache.metadata() or {}
    expected = {"turbo8_lora_sha256": H3_TURBO8_LORA_SHA256, "num_eval_steps": "8",
                "video_shift": "6.0", "audio_shift": "3.0", "model_variant": "T2VA"}
    if model.get("turbo8_lora_sha256") != H3_TURBO8_LORA_SHA256 or any(
            metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Turbo8 transformer or AdaLN metadata does not match the 768P preset")


def validate_h3_turbo_sla(args):
    path = getattr(args, "h3_turbo_sla_lora", None)
    if not path:
        return
    if args.model != "MiniMax-H3" or args.task_type not in {"t2va", "fl2va"}:
        raise ValueError("Turbo-SLA requires MiniMax-H3 with task_type t2va or fl2va.")
    if getattr(args, "reference_paths", None):
        raise ValueError(
            "The FL2VA Turbo-SLA adapter does not support Ref2VA reference media."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Turbo-SLA LoRA not found: {path}")
    if args.num_inference_steps != 4:
        raise ValueError(
            "Turbo-SLA requires --num_inference_steps 4 (four transformer forwards)."
        )
    if args.atten != "none":
        raise ValueError("Turbo-SLA selects its own attention; use --atten none.")
    if args.dtype != "bf16" or args.matmul_quant != "none":
        raise ValueError(
            "The Turbo-SLA reference preset requires BF16 and --matmul_quant none."
        )
    if args.lora_path_list or args.fuse_lora:
        raise ValueError(
            "Use --h3_turbo_sla_lora alone; ordinary LoRA overlays/fusion are not qualified."
        )
    if args.phaa_num > 0 or (args.ring_degree or 1) != 1:
        raise ValueError(
            "Turbo-SLA requires full KV: PHAA and ring parallelism are unsupported; use Ulysses."
        )
    if args.sp not in (1, 2, 4, 8):
        raise ValueError(
            "Turbo-SLA Ulysses SP must be 1, 2, 4 or 8 (must divide H3's 56 heads)."
        )
    if (
        args.fsdp
        or args.inf_vram_blocks_num
        or args.cache_dit
        or (args.turbo_mode != "default")
    ):
        raise ValueError(
            "Turbo-SLA reference supports group offload/Ulysses; disable FSDP, custom VRAM offload and caches."
        )
    if min(args.height, args.width) > 768 or args.height % 32 or args.width % 32:
        raise ValueError(
            "Turbo-SLA expects a short edge <= 768 and dimensions divisible by 32."
        )
    if not 124 <= args.frames <= 362 or (args.frames - 5) % 17:
        raise ValueError("Turbo-SLA frame count must be 17*n+5 in [124, 362].")


def validate_h3_vae(args, backend):
    selected = resolve_vae_backend(args)
    validate_h3_vae_compile(args, backend)
    if selected != "native-stream":
        if getattr(args, "h3_vae_model_path", None) or getattr(
            args, "h3_vae_trust_remote_code", False
        ):
            raise ValueError(
                "Native H3 VAE options require --h3_vae_backend native-stream"
            )
        return
    if (
        selected != "native-stream"
        or backend not in ("cuda", "npu")
        or args.model != "MiniMax-H3"
    ):
        raise ValueError("Native-stream VAE decode requires CUDA or Ascend MiniMax-H3")
    if not getattr(args, "h3_vae_model_path", None) or not getattr(
        args, "h3_vae_trust_remote_code", False
    ):
        raise ValueError(
            "Native-stream VAE requires --h3_vae_model_path and --h3_vae_trust_remote_code"
        )
    if getattr(args, "vae_matmul_quant", "none") != "none":
        raise ValueError(
            "Native-stream uses the native VAE weights; set --vae_matmul_quant none"
        )
    if getattr(args, "h3_refiner", False):
        raise ValueError("H3 Refiner is not yet supported by native-stream decode")
    if _distributed().is_initialized() and _distributed().get_world_size() != getattr(
        args, "sp", 1
    ):
        raise ValueError(
            "Native-stream H3 decode currently requires one request across the complete SP world; DP groups are not supported"
        )


def validate_h3_vae_compile(args, backend: str) -> None:
    """Validate the opt-in compiler before any model/device work starts."""
    if not bool(getattr(args, "h3_vae_torch_compile", False)):
        return
    if backend != "cuda":
        raise ValueError("--h3_vae_torch_compile is CUDA-only; keep it disabled on NPU")
    if getattr(args, "model", None) != "MiniMax-H3":
        raise ValueError("--h3_vae_torch_compile requires MiniMax-H3")
    if getattr(args, "h3_vae_backend", None) != "native-stream":
        raise ValueError(
            "--h3_vae_torch_compile requires --h3_vae_backend native-stream"
        )
    import torch

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "Torch Compiler requires a PyTorch build exposing torch.compile"
        )


def resolve_vae_backend(args):
    selected = getattr(args, "h3_vae_backend", None)
    if selected is None:
        selected = (
            "native-stream"
            if getattr(args, "h3_cuda_preset", None) == "sage-fp8-full-parallel"
            or getattr(args, "h3_cuda_precision", None) in ("fp8-bf16", "fp8-sage")
            else "legacy"
        )
    if getattr(args, "model", None) != "MiniMax-H3":
        if selected != "legacy":
            raise ValueError("--h3_vae_backend requires MiniMax-H3")
        args.h3_vae_backend = selected
        return selected
    if selected not in ("legacy", "lightning", "native-stream"):
        raise ValueError("H3 VAE backend must be legacy, lightning or native-stream")
    previous = getattr(args, "_h3_vae_selection", None)
    supplied = getattr(args, "vae_lightning", "none")
    expected = previous[1] if previous is not None else "none"
    if supplied != expected:
        raise ValueError(
            "For MiniMax-H3 use --h3_vae_backend lightning instead of --vae_lightning"
        )
    if selected == "lightning" and getattr(args, "sp", 1) <= 1:
        raise ValueError("H3 Lightning requires --sp greater than 1")
    args.vae_lightning = "decoder" if selected == "lightning" else "none"
    args.h3_vae_backend = selected
    args._h3_vae_selection = (selected, args.vae_lightning)
    return selected


_H3_DIT_FUSION_FLAGS = (
    "XGEN_H3_FUSED_INDEXED_AFFINE",
    "XGEN_H3_FUSED_PARTIAL_ROPE",
    "XGEN_H3_NVFP4_FUSED_AMAX",
    "XGEN_H3_FUSED_SWIGLU",
    "XGEN_H3_FUSED_QK_RMS",
    "XGEN_H3_FUSED_FF_AMAX",
)


def get_h3_dit_fusions(packed_resident, packed_compile):
    enabled = frozenset(
        (flag for flag in _H3_DIT_FUSION_FLAGS if os.environ.get(flag) == "1")
    )
    if enabled and (not packed_resident or packed_compile):
        raise ValueError(
            f"{', '.join(sorted(enabled))} requires resident CUDA NVFP4 without block compile"
        )
    return enabled


def _check_feature_conflicts(args):
    if args.fuse_lora and args.matmul_quant != "none":
        args.fuse_lora = False
        logger.warning(
            "Conflict between fuse lora and matmul quant! Disable fuse_lora!"
        )
    if args.joint:
        if not (
            args.x
            and "Wan2.2" in args.model
            and ("I2V" in args.model or "i2v" == args.task_type)
        ):
            args.joint = False
            logger.warning(
                "Jointinfer only support Wan2.2_distill_i2v! Disable jointinfer!"
            )
    if args.ada_brighten:
        if not ("I2V" in args.model or "i2v" == args.task_type):
            args.ada_brighten = False
            logger.warning("Ada_brighten only support i2v! Disable ada_brighten!")


def _check_parallelism_conflicts(args):
    if args.text_encoder_tp > 1 and args.fsdp in ("text_encoder", "all"):
        logger.warning(
            "text_encoder_tp conflicts with --fsdp text_encoder/all (both target text_encoder)! Disabling FSDP for text_encoder."
        )
        if args.fsdp == "all":
            args.fsdp = "transformer"
        else:
            args.fsdp = None
    if getattr(args, "transformer_path", None) and "MiniMax-H3" not in args.model:
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


def validate_pipeline_args(args, backend):
    if getattr(args, "h3_condition_overlap", False):
        validate_h3_overlap(args, backend)
    validate_h3_npu_fusions(args, backend)
    resolve_h3_precision(args, backend)
    validate_h3_vae(args, backend)
    resolve_parallel_config(
        args,
        backend,
        _distributed().get_world_size() if _distributed().is_initialized() else 1,
    )
    precision = getattr(args, "h3_cuda_precision", "auto")
    if precision != "none":
        precision = resolve_h3_precision(args, backend)
        if precision != "none":
            validate_h3_precision(args, backend)
    commit_h3_environment(args)
    if (backend != "cuda" or args.model != "MiniMax-H3") and any(
        (
            getattr(args, "h3_nvfp4_checkpoint", None),
            getattr(args, "h3_nvfp4_resident", False),
            getattr(args, "h3_nvfp4_compile", False),
        )
    ):
        raise ValueError("H3 NVFP4 options require CUDA MiniMax-H3")
    if getattr(args, "h3_cuda_vae_sp", False) and (
        backend != "cuda" or args.model != "MiniMax-H3" or args.sp < 2
    ):
        raise ValueError("--h3_cuda_vae_sp requires CUDA MiniMax-H3 and SP > 1")
    if backend == "cuda":
        if args.text_encoder_tp == 8 and (
            not _distributed().is_initialized() or _distributed().get_world_size() != 8
        ):
            raise ValueError(
                "CUDA H3 text encoder TP8 requires eight distributed ranks"
            )
        unsupported = (
            args.model != "MiniMax-H3",
            args.atten not in ("none", "sage", "sparge"),
            args.matmul_quant != "none",
            args.vae_matmul_quant != "none",
            args.vae_lightning != "none",
            args.rope_fused,
            args.text_encoder_tp not in (1, 8),
            args.inf_vram_blocks_num,
            args.fsdp,
            getattr(args, "h3_attention", "legacy") != "legacy",
        )
        if any(unsupported):
            raise ValueError(
                "CUDA currently supports MiniMax-H3 with SDPA, SageAttention or SpargeAttention, group offload, and without NPU-specific optimizations"
            )
    if getattr(args, "h3_turbo8", False):
        validate_h3_turbo8(args)
        validate_h3_turbo8_assets(args)
    if getattr(args, "h3_attention", "legacy") != "legacy":
        validate_h3_attention(args)
    validate_h3_turbo_sla(args)


def bootstrap_cli_options(args):
    if getattr(args, "h3_cuda_preset", None) not in (None, "none") or (
        args.model == "MiniMax-H3" and args.h3_cuda_precision == "auto"
    ):
        from x_base.device import device_type

        backend = device_type()
        select_h3_default_recipe(args, backend)
        if getattr(args, "h3_cuda_preset", None) not in (None, "none"):
            resolve_h3_options(args, backend)
            commit_h3_environment(args)


def apply_h3_attention_startup_settings(config, args, argv):
    for name in ('h3_attention','h3_veda_predictor'):
        if any(value.split('=',1)[0]=='--'+name for value in argv):
            config[name]=getattr(args,name)


def validate_h3_attention_warmup(config, enabled):
    if enabled and config.get('h3_attention','legacy')!='legacy':
        raise ValueError('Sparse H3 attention requires a full 50-step first request, not reduced-step warmup')


def validate_h3_attention_request(config, data, errors):
    for name in ('h3_attention','h3_veda_predictor'):
        if name in data:
            errors.append(name+' is a startup setting; restart to change it')
    if type(data.get('duration',5)) is not int or data.get('duration',5)!=5:
        errors.append('Initial sparse H3 attention requires duration=5')
    if data.get('task_type',config.get('task_type','t2va'))!='t2va':
        errors.append('Initial sparse H3 attention requires t2va')
    steps=data.get('num_inference_steps',config.get('num_inference_steps',50))
    if type(steps) is not int or steps!=50:
        errors.append('Initial sparse H3 attention requires 50 Base denoising steps')
    resolution='2k' if config.get('h3_refiner',False) else '768p'
    if data.get('resolution',resolution)!=resolution or data.get('aspect_ratio','16:9')!='16:9':
        errors.append('Sparse H3 resolution and landscape aspect ratio are fixed at startup')
    if any(data.get(name) for name in ('first_frame','last_frame','ref_path','adopt_sr',
                                        'frame_interpolation','attention_sparse')):
        errors.append('Sparse H3 does not accept additional media, attention modes or super-resolution plugins')


def configure_h3_attention_args(config, data, args):
    errors=[];validate_h3_attention_request(config,data,errors)
    if errors:
        raise ValueError('; '.join(errors))
    args.h3_attention=config['h3_attention']
    args.h3_veda_predictor=config.get('h3_veda_predictor')
    args.frames=124
    args.num_inference_steps=data.get('num_inference_steps',config.get('num_inference_steps',50))
    validate_h3_attention(args)


def validate_h3_attention_manager(manager, args):
    validate_h3_attention(args)
    if manager.init_args is not None and h3_attention_key(manager.init_args)!=h3_attention_key(args):
        raise ValueError('Restart the service to change H3 attention backend or predictor')


def h3_attention_parameter_spec(config):
    return dict(experimental=True,backend=config['h3_attention'],scope='Base DiT only',parameters={
        'prompt':dict(type='string',required=True,max_length=H3_MAX_PROMPT_LENGTH,length_unit='utf16_code_units'),
        'task_type':dict(type='string',allowed_values=['t2va']),
        'duration':dict(type='integer',allowed_values=[5],default=5),
        'resolution':dict(type='string',allowed_values=['2k' if config.get('h3_refiner',False) else '768p']),
        'aspect_ratio':dict(type='string',allowed_values=['16:9']),
        'num_inference_steps':dict(type='integer',allowed_values=[50]),
        'seed':dict(type='integer',min=0,max=2**31-1),
        'enable_audio':dict(type='boolean',default=True)})


def h3_attention_config(args):
    from x_diffusers.framework.h3_config import H3AttentionConfig

    return H3AttentionConfig(
        backend=getattr(args, 'h3_attention', 'legacy'),
        predictor=getattr(args, 'h3_veda_predictor', None),
        turbo8=getattr(args, 'h3_turbo8', False),
        sparsity=getattr(args, 'h3_veda_sparsity', 0.9),
    )


def apply_h3_refiner_startup_settings(config, args, argv):
    """CLI overrides only explicitly supplied new fields; never erase JSON asset paths."""
    config['h3_refiner'] = True
    for name in ('h3_refiner_weights','h3_refiner_prepared','h3_refiner_loading','h3_refiner_seconds'):
        explicit = any(value.split('=',1)[0] == '--'+name for value in argv)
        if explicit or name not in config:
            config[name] = getattr(args,name)


def h3_refiner_parameter_spec(config):
    return dict(experimental=True,backend='h3_regenerate',parameters={
        'prompt':dict(type='string',required=True,max_length=H3_MAX_PROMPT_LENGTH,length_unit='utf16_code_units'),
        'task_type':dict(type='string',allowed_values=[config.get('task_type','t2va')]),
        'duration':dict(type='integer',allowed_values=[5,10,15],default=config.get('h3_refiner_seconds',5)),
        'resolution':dict(type='string',allowed_values=['2k'],default='2k'),
        'aspect_ratio':dict(type='string',allowed_values=['16:9'],default='16:9'),
        'num_inference_steps':dict(type='integer',allowed_values=[config.get('num_inference_steps',50)]),
        'first_frame':dict(type='string',required=config.get('task_type')=='fl2va'),
        'ref_path':dict(type='array',max_items=H3_REFERENCE_MAX_FILES,media_limits=H3_REFERENCE_LIMITS,
                        clip_seconds=[H3_REFERENCE_MIN_SECONDS,H3_REFERENCE_MAX_SECONDS],max_seconds_per_media_type=H3_REFERENCE_MAX_SECONDS,
                        required=config.get('task_type')=='ref2va'),
        'enable_audio':dict(type='boolean',default=True),
        'seed':dict(type='integer',min=0,max=2**63-1)},
        example=dict(prompt='A small robot walks through a sunlit garden.',duration=5,resolution='2k',seed=42))


def configure_h3_refiner_args(config, data, args):
    # Startup settings cannot be supplied by a remote request.
    for key in ('h3_refiner', 'h3_refiner_weights', 'h3_refiner_prepared', 'h3_refiner_loading'):
        setattr(args, key, config.get(key, getattr(args, key, None)))
    args.h3_refiner_seconds = data.get('duration', config.get('h3_refiner_seconds', 5))
    spec = h3_refiner_profile(args.h3_refiner_seconds)
    args.frames = spec['generation_frames']
    args.width, args.height, args.save_fps = 2560, 1440, 24
    args.num_inference_steps = data.get('num_inference_steps', config.get('num_inference_steps', 50))
    args.adopt_sr = False


def validate_h3_refiner_request(service, data, errors, warnings):
    """Reuse existing prompt/media validators; add only the enabled feature's bounds."""
    prompt = service._validate_prompt(data, errors)
    task = data.get('task_type', service.default_config.get('task_type', 't2va'))
    if task != service.default_config.get('task_type', 't2va'):
        errors.append('Experimental H3 Refiner fixes task_type at startup; use a separate worker for another workflow')
    if task not in ('t2va', 'fl2va', 'ref2va'):
        errors.append('task_type must be t2va/fl2va/ref2va')
    first, last = service._validate_mmh3_frame_fields(data, task, errors)
    refs = service._validate_mmh3_ref_path(data, task, errors)
    if task == 'fl2va' and (not first or last):
        errors.append('Experimental fl2va accepts one first-frame image only')
    duration = data.get('duration', service.default_config.get('h3_refiner_seconds', 5))
    try:
        spec = h3_refiner_profile(duration)
    except ValueError as exc:
        errors.append(str(exc)); spec = h3_refiner_profile(5)
    if data.get('resolution', '2k') != '2k' or data.get('aspect_ratio', '16:9') != '16:9':
        errors.append('Experimental H3 Refiner requires resolution=2k and aspect_ratio=16:9')
    steps = data.get('num_inference_steps', service.default_config.get('num_inference_steps', 50))
    if type(steps) is not int or steps != service.default_config.get('num_inference_steps', 50):
        errors.append('num_inference_steps is fixed at startup and must match Base AdaLN assets')
    seed = data.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        errors.append('seed must be an integer in [0, 2**63)')
    if data.get('adopt_sr', False) is not False:
        errors.append('SeedVR2 adopt_sr cannot be combined with H3 Refiner')
    audio = data.get('enable_audio', True)
    if type(audio) is not bool:
        errors.append('enable_audio must be boolean')
    if any(key.startswith('h3_refiner') for key in data):
        errors.append('h3_refiner settings are startup-only; restart to change or disable the feature')
    if errors:
        return False, dict(errors=errors, warnings=warnings)
    validated = dict(model='MiniMax-H3', task_type=task, prompt=prompt, resolution='2k',aspect_ratio='16:9',
                     duration=duration,frames=spec['generation_frames'],num_inference_steps=steps,
                     seed=seed,save_fps=24,first_frame=first,last_frame=last,ref_path=refs,
                     enable_audio=audio,adopt_sr=False,attention_sparse=False)
    if data.get('save_path'):
        validated['save_path'] = data['save_path']
    return True, dict(data=validated,warnings=warnings+['experimental H3 Refiner; no automatic retry'])


def resolve_h3_cuda_adaln_assets(args, backend, profile):
    """Resolve workflow-scoped paths without I/O or creating model assets."""
    mode = getattr(args, 'h3_cuda_adaln_mode', None) or 'auto'
    if mode not in ('auto', 'table', 'owner'):
        raise ValueError('Unknown CUDA AdaLN mode')
    root = getattr(args, 'h3_cuda_adaln_dir', None)
    if backend != 'cuda' or profile == 'none':
        if mode != 'auto' or root:
            raise ValueError('CUDA AdaLN modes require a CUDA H3 FP8 profile')
        return
    owned = getattr(args, '_h3_cuda_adaln_auto_paths', False)
    checkpoint = getattr(args, 'transformer_path', None)
    table = getattr(args, 'adaln_cache_file', None)
    if mode == 'owner':
        if owned:
            args.transformer_path = args.adaln_cache_file = None
        elif checkpoint or table or root:
            raise ValueError('CUDA AdaLN owner mode cannot combine with table assets')
        return
    active = mode == 'table' or bool(checkpoint or table or root) or (
        getattr(args, 'h3_cuda_preset', None) == 'sage-fp8-full-parallel' and profile == 'fp8-sage')
    if not active:
        return
    if getattr(args, 'h3_parallel_mode', 'sp') != 'sp':
        raise ValueError('CUDA AdaLN tables require native SP; use --h3_cuda_adaln_mode owner for TP/CP')
    args.h3_cuda_adaln_mode = 'table'
    if (checkpoint or table) and not owned:
        if root:
            raise ValueError('Choose h3_cuda_adaln_dir or explicit transformer_path/adaln_cache_file, not both')
        return
    variant = 'ref2va' if getattr(args, 'task_type', None) == 'ref2va' else 'fl2va'
    directory = Path(root) if root else Path(getattr(args, 'pretrained_model_name_or_path', 'weights/h3'))/'xgen-adaln'/variant
    steps = getattr(args, 'num_inference_steps', 50)
    if type(steps) is not int or steps < 1:
        raise ValueError('CUDA AdaLN table selection requires positive actual DiT steps')
    args.h3_cuda_adaln_dir = str(directory)
    args.transformer_path = str(directory/'transformer')
    args.adaln_cache_file = str(directory/'tables'/f'adaln_{variant}_bf16_{steps}step.safetensors')
    args._h3_cuda_adaln_auto_paths = True


def validate_h3_cuda_adaln(args):
    checkpoint = getattr(args, 'transformer_path', None)
    table = getattr(args, 'adaln_cache_file', None)
    if not checkpoint and not table:
        return False
    if not checkpoint or not table:
        raise ValueError('CUDA FP8 AdaLN tables require both transformer_path and adaln_cache_file')
    if not Path(table).is_file() or not (Path(checkpoint)/'.xgen_adaln_source.json').is_file():
        raise ValueError('Missing prepared CUDA AdaLN assets: generate matched tables and a stripped checkpoint '
                         'with tools/adaln_cache/prepare_cuda_adaln.py (see docs/H3-MODELS.md#adaln-cuda), '
                         'or explicitly select --h3_cuda_adaln_mode owner')
    task = getattr(args, 'task_type', None)
    variant = {'t2va': 'FL2VA', 'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}.get(task)
    if variant is None or getattr(args, 'h3_parallel_mode', 'sp') != 'sp':
        raise ValueError('CUDA FP8 AdaLN tables require H3 T2VA/FL2VA/Ref2VA native SP')
    from safetensors import safe_open

    h3_schedule_points(args.num_inference_steps, table)
    with safe_open(table, framework='pt') as cache:
        metadata = cache.metadata() or {}
    manifest = json.loads((Path(checkpoint) / '.xgen_adaln_source.json').read_text())
    from x_base.adaln_cache import ADALN_DELETE_PATTERNS
    index = json.loads((Path(checkpoint) / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    if (len(manifest.get('removed_keys', [])) != 106
            or any(any(fnmatch.fnmatch(key, pattern) for pattern in ADALN_DELETE_PATTERNS) for key in index['weight_map'])):
        raise ValueError('CUDA AdaLN table mode requires a checkpoint with AdaLN weights physically removed')
    if (metadata.get('model_variant') != variant or manifest.get('model_variant') != variant
            or metadata.get('quantization', 'none') != 'none'
            or int(metadata.get('num_blocks', 0)) != 50
            or not manifest.get('adaln_weight_sha256')
            or metadata.get('adaln_weight_sha256') != manifest['adaln_weight_sha256']
            or any(float(metadata[name]) != float(manifest[name]) for name in ('video_shift', 'audio_shift'))):
        raise ValueError('AdaLN table must match the workflow, stripped checkpoint and scheduler')
    return True


def h3_reference_image_resize_mode(args):
    mode = getattr(args, "h3_reference_image_resize_mode", None)
    if mode is None:
        return None
    if mode not in REFERENCE_RESIZE_MODES:
        choices = ", ".join(REFERENCE_RESIZE_MODES)
        raise ValueError(f"Invalid h3_reference_image_resize_mode {mode!r}. Choose one of {choices}.")
    if getattr(args, "task_type", None) != "ref2va" or getattr(args, "h3_refiner", False):
        raise ValueError("h3_reference_image_resize_mode requires Ref2VA Base")
    if getattr(args, "h3_reference_image_short_edge", None) is not None:
        raise ValueError("Set either h3_reference_image_resize_mode or h3_reference_image_short_edge")
    return mode


def h3_reference_media_kind(path):
    suffix = Path(path).suffix.lower()
    for kind, extensions in (("image", H3_REFERENCE_IMAGE_EXTS), ("video", H3_REFERENCE_VIDEO_EXTS), ("audio", H3_REFERENCE_AUDIO_EXTS)):
        if suffix in extensions:
            return kind
    raise ValueError(f"Unsupported Ref2VA reference type: {suffix or '(no extension)'}")


def validate_h3_reference_image_size(args):
    """Resolve the optional reference short edge before pipeline assembly."""
    h3_reference_image_resize_mode(args)
    edge = getattr(args, "h3_reference_image_short_edge", None)
    if edge is None:
        return None
    if getattr(args, "task_type", None) != "ref2va" or getattr(args, "h3_refiner", False):
        raise ValueError("h3_reference_image_short_edge currently supports Ref2VA Base only")
    if type(edge) is not int or not 32 <= edge <= 2048 or edge % 32:
        raise ValueError("h3_reference_image_short_edge must be an integer multiple of 32 in [32, 2048]")
    return edge


def probe_h3_reference_clip(path, kind):
    try:
        result = subprocess.run(
            [os.environ.get("FFPROBE_PATH", "ffprobe"), "-v", "error", "-show_streams",
             "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        streams = [stream for stream in info.get("streams", ())
                   if not stream.get("disposition", {}).get("attached_pic")]
        stream = next(stream for stream in streams if stream.get("codec_type") == kind)
        duration = stream.get("duration")
        if duration in (None, "N/A"):
            duration = info.get("format", {}).get("duration")
        seconds = Decimal(str(duration))
        if not seconds.is_finite():
            raise ValueError("Nonfinite stream duration")
    except (OSError, subprocess.SubprocessError, ValueError, InvalidOperation, StopIteration) as exc:
        raise ValueError(f"Could not probe {kind} reference {Path(path).name}: {exc}") from exc
    return seconds, any(stream.get("codec_type") == "audio" for stream in streams)


def validate_h3_references(paths):
    """Check local media before model loading. A video soundtrack is part of that file."""
    if not isinstance(paths, (list, tuple)) or not paths or any(not isinstance(p, str) or not p for p in paths):
        raise ValueError("Ref2VA requires a nonempty list of reference file paths")
    kinds = [h3_reference_media_kind(path) for path in paths]
    counts = Counter(kinds)
    if len(paths) > H3_REFERENCE_MAX_FILES:
        raise ValueError(f"Ref2VA accepts at most {H3_REFERENCE_MAX_FILES} reference files total")
    for kind, maximum in H3_REFERENCE_LIMITS.items():
        if counts[kind] > maximum:
            raise ValueError(f"Ref2VA accepts at most {maximum} {kind} references")
    if not (counts["image"] or counts["video"]):
        raise ValueError("Audio references require at least one image or video reference")
    files = [Path(path).resolve() for path in paths]
    if len(set(files)) != len(files):
        raise ValueError("Ref2VA reference file paths must be distinct")

    has_audio = False
    totals = Counter()
    for path, kind in zip(files, kinds):
        if not path.is_file():
            raise ValueError(f"Ref2VA reference file does not exist: {path.name}")
        if kind == "image":
            from PIL import Image

            with Image.open(path) as image:
                if min(image.size) < 1 or image.width * image.height > 4096 * 4096:
                    raise ValueError("Reference images must contain 1..4096*4096 pixels")
                image.verify()
            continue
        seconds, audio = probe_h3_reference_clip(path, kind)
        if not H3_REFERENCE_MIN_SECONDS <= seconds <= H3_REFERENCE_MAX_SECONDS:
            raise ValueError(f"Each {kind} reference must be {H3_REFERENCE_MIN_SECONDS}–{H3_REFERENCE_MAX_SECONDS} seconds: {path.name}")
        totals[kind] += seconds
        if totals[kind] > H3_REFERENCE_MAX_SECONDS:
            raise ValueError(f"Combined {kind} reference duration must be at most {H3_REFERENCE_MAX_SECONDS} seconds")
        has_audio |= audio
    return "both" if has_audio else "video"


def h3_reference_conditioning(args):
    paths = getattr(args, "reference_paths", None)
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Ref2VA requires a nonempty list of reference file paths")
    paths = tuple(paths)
    cached = getattr(args, "_h3_reference_conditioning", None)
    if cached is None or cached[0] != paths:
        cached = (paths, validate_h3_references(paths))
        args._h3_reference_conditioning = cached
    return cached[1]


def validate_h3_image_request(args, world_size):
    validate_h3_image_options(args, world_size)
    pass

    validate_h3_image_checkpoint(args.pretrained_model_name_or_path)


def validate_h3_image_checkpoint(model_root):
    from ..framework.pipeline.minimax_h3_image import validate_pdd_config

    root = Path(model_root)
    for name in (
        "modular_model_index.json",
        "transformer_diffusers/config.json",
        "vae_diffusers/config.json",
        "text_encoder/config.json",
        "pdd_config.json",
    ):
        if not (root / name).is_file():
            raise ValueError(f"Missing local H3 Image component: {root / name}")
    pdd = json.loads((root / "pdd_config.json").read_text())
    validate_pdd_config(pdd)
