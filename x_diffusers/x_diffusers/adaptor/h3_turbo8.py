"""Experimental offline LightX2V 768P profile, shared by Laser and Veda."""
import json
import logging
from pathlib import Path
from types import MethodType

LORA_SHA256 = "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4"
SCHEDULE_POINTS = 9
SHIFTS = {"scheduler": 6.0, "audio_scheduler": 3.0}
logger = logging.getLogger("infer")


def validate(args):
    if (args.model, args.task_type, args.dtype, args.sp, args.atten) != (
            "MiniMax-H3", "t2va", "bf16", 8, "laser"):
        raise ValueError("Turbo8 requires MiniMax-H3 T2VA, BF16, SP8 and atten=laser")
    expected = (2560, 1440) if getattr(args, "h3_refiner", False) else (1344, 768)
    if ((args.width, args.height) != expected or args.frames != 124 or args.save_fps != 24
            or args.num_inference_steps != SCHEDULE_POINTS
            or getattr(args, "h3_refiner_seconds", 5) != 5):
        raise ValueError("Turbo8 requires 9 schedule points and the 5s 768P or 2K profile")
    if getattr(args, "h3_attention", "legacy") not in ("legacy", "veda"):
        raise ValueError("Turbo8 supports Laser and experimental Veda")
    if any(getattr(args, name, None) for name in (
            "h3_turbo_sla_lora", "lora_path_list", "fuse_lora", "cache_dit",
            "keyframe_image", "keyframe_last_image", "reference_paths", "adopt_sr")):
        raise ValueError("Turbo8 requires prepared assets without additional adapters, caches or media")


def validate_assets(args):
    from safetensors import safe_open

    manifest = Path(args.transformer_path or "") / "turbo8.json"
    if not args.transformer_path or not manifest.is_file() or not args.adaln_cache_file:
        raise ValueError("Turbo8 requires a prepared transformer and matching AdaLN cache")
    model = json.loads(manifest.read_text(encoding="utf-8"))
    with safe_open(args.adaln_cache_file, framework="pt") as cache:
        metadata = cache.metadata() or {}
    expected = {"turbo8_lora_sha256": LORA_SHA256, "num_eval_steps": "8",
                "video_shift": "6.0", "audio_shift": "3.0", "model_variant": "T2VA"}
    if model.get("turbo8_lora_sha256") != LORA_SHA256 or any(
            metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Turbo8 transformer or AdaLN metadata does not match the 768P preset")


def configure(pipe):
    from .h3_turbo_sla import training_euler_step

    for name, shift in SHIFTS.items():
        scheduler = getattr(pipe, name)
        scheduler.set_shift(shift)
        scheduler.set_timesteps(SCHEDULE_POINTS, device="cpu")
        if len(scheduler.timesteps) != 8 or len(scheduler.sigmas) != SCHEDULE_POINTS:
            raise RuntimeError("Turbo8 scheduler must produce eight DiT evaluations")
        scheduler.step = MethodType(training_euler_step, scheduler)
    logger.warning("H3_TURBO8 experimental=true nfe=8 video_shift=6 audio_shift=3")
