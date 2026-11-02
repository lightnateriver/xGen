"""Explicit MiniMax-H3 FL2VA Turbo-SLA preset; no changes to other LoRAs."""

import logging
import math
import os
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("infer")


def validate_args(args):
    path = getattr(args, "h3_turbo_sla_lora", None)
    if not path:
        return
    if args.model != "MiniMax-H3" or args.task_type not in {"t2va", "fl2va"}:
        raise ValueError("Turbo-SLA requires MiniMax-H3 with task_type t2va or fl2va.")
    if getattr(args, "reference_paths", None):
        raise ValueError("The FL2VA Turbo-SLA adapter does not support Ref2VA reference media.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Turbo-SLA LoRA not found: {path}")
    if args.num_inference_steps != 4:
        raise ValueError("Turbo-SLA requires --num_inference_steps 4 (four transformer forwards).")
    if args.atten != "none":
        raise ValueError("Turbo-SLA selects its own attention; use --atten none.")
    if args.dtype != "bf16" or args.matmul_quant != "none":
        raise ValueError("The Turbo-SLA reference preset requires BF16 and --matmul_quant none.")
    if args.lora_path_list or args.fuse_lora:
        raise ValueError("Use --h3_turbo_sla_lora alone; ordinary LoRA overlays/fusion are not qualified.")
    if args.phaa_num > 0 or (args.ring_degree or 1) != 1:
        raise ValueError("Turbo-SLA requires full KV: PHAA and ring parallelism are unsupported; use Ulysses.")
    if args.sp not in (1, 2, 4, 8):
        raise ValueError("Turbo-SLA Ulysses SP must be 1, 2, 4 or 8 (must divide H3's 56 heads).")
    if args.fsdp or args.inf_vram_blocks_num or args.cache_dit or args.turbo_mode != "default":
        raise ValueError("Turbo-SLA reference supports group offload/Ulysses; disable FSDP, custom VRAM offload and caches.")
    if min(args.height, args.width) > 768 or args.height % 32 or args.width % 32:
        raise ValueError("Turbo-SLA expects a short edge <= 768 and dimensions divisible by 32.")
    if not 124 <= args.frames <= 362 or (args.frames - 5) % 17:
        raise ValueError("Turbo-SLA frame count must be 17*n+5 in [124, 362].")


class TurboSLALinear(nn.Module):
    """Dynamic BF16 LoRA branch, installed before group-offload discovery."""

    def __init__(self, base, down, up, alpha):
        super().__init__()
        self.base_layer = base
        self.lora_down = nn.Parameter(down.to(base.weight), requires_grad=False)
        self.lora_up = nn.Parameter(up.to(base.weight), requires_grad=False)
        self.scale = alpha / down.shape[0]

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def forward(self, x):
        result = self.base_layer(x)
        delta = F.linear(F.linear(x.to(self.lora_down.dtype), self.lora_down), self.lora_up)
        return result + (delta * self.scale).to(result.dtype)


def _normalize_key(key):
    for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    for suffix, part in ((".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
                         (".lora_A.weight", "down"), (".lora_B.weight", "up"),
                         (".lora.down.weight", "down"), (".lora.up.weight", "up"),
                         (".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
        if key.endswith(suffix):
            return key[:-len(suffix)], part
    raise ValueError(f"Unsupported Turbo-SLA tensor: {key}")


def install_lora(transformer, state_dict, *, default_alpha=128.0, has_adaln_cache=False):
    """Validate every pair/shape before mutation; never silently skip a tensor.

    Native LightX2V keys already use the Diffusers layout. ComfyUI blocks/qkv
    and SwiGLU layouts deliberately fail target lookup rather than guessing.
    """
    pairs = {}
    for key, tensor in state_dict.items():
        target, part = _normalize_key(key)
        pair = pairs.setdefault(target, {})
        if part in pair:
            raise ValueError(f"Duplicate Turbo-SLA tensor after normalization: {target}.{part}")
        pair[part] = tensor
    if not pairs:
        raise ValueError("Empty Turbo-SLA adapter.")
    replacements = []
    for name, pair in pairs.items():
        if "down" not in pair or "up" not in pair:
            raise ValueError(f"Incomplete LoRA pair: {name}")
        if has_adaln_cache and any(token in name for token in ("adaln", "time_embedder", "norm_out.linear")):
            raise ValueError(f"LoRA changes cached AdaLN projections: {name}; disable the cache.")
        base = transformer.get_submodule(name)
        down, up = pair["down"], pair["up"]
        if not isinstance(base, nn.Linear) or base.weight.is_meta:
            raise ValueError(f"Turbo-SLA needs a materialized Linear target: {name}")
        if (down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or
                down.shape[0] != up.shape[1] or (up.shape[0], down.shape[1]) != tuple(base.weight.shape)):
            raise ValueError(f"LoRA shape mismatch for {name}: down={tuple(down.shape)}, up={tuple(up.shape)}")
        alpha = float(pair.get("alpha", default_alpha))
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid LoRA alpha for {name}: {alpha}")
        if not down.is_floating_point() or not up.is_floating_point():
            raise ValueError(f"LoRA matrices must be floating-point: {name}")
        replacements.append((name, base, down, up, alpha))
    # Avoid fused QKV paths reading the original projections' .weight directly.
    for module in transformer.modules():
        if getattr(module, "fused_projections", False):
            raise ValueError("Unfuse QKV projections before installing Turbo-SLA.")
    for name, base, down, up, alpha in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, child_name, TurboSLALinear(base, down, up, alpha))
    logger.info("Turbo-SLA loaded %d Linear pairs; default alpha=%g, strength=1", len(replacements), default_alpha)
    return len(replacements)


def training_euler_step(self, model_output, timestep, sample, return_dict=True):
    """Match LightX2V training_euler: x += (sigma - sigma_next) * velocity."""
    if self._step_index is None:
        self._step_index = self.index_for_timestep(timestep) if self.begin_index is None else self.begin_index
    sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=torch.float32)
    next_sigma = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=torch.float32)
    prev_sample = sample.float() + (sigma - next_sigma) * model_output.float()
    self._step_index += 1
    if not return_dict:
        return (prev_sample,)
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput
    return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)


def configure_pipe(pipe, args):
    """Called after components load, before hooks/offloading/parallel wrappers."""
    if not getattr(args, "h3_turbo_sla_lora", None):
        return
    from safetensors.torch import load_file

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


def inference_params(pipe, args):
    """Diffusers counts sigma points; LightX2V counts transformer evaluations."""
    validate_args(args)
    if not getattr(args, "h3_turbo_sla_lora", None):
        return {}
    if getattr(pipe, "_h3_turbo_sla_path", None) != os.path.abspath(args.h3_turbo_sla_lora):
        raise RuntimeError("Turbo-SLA adapter changed; rebuild the pipeline before inference.")
    # Five sigma points -> four forwards. Verify upstream semantics on every request.
    for name in ("scheduler", "audio_scheduler"):
        scheduler = getattr(pipe, name)
        scheduler.set_timesteps(5, device="cpu")
        if len(scheduler.timesteps) != 4 or len(scheduler.sigmas) != 5:
            raise RuntimeError("Diffusers H3 scheduler step-count contract changed; expected 5 sigmas / 4 forwards.")
    return {"num_inference_steps": 5}
