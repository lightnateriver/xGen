"""H3 LoRA weight adaptation and training-Euler scheduler implementation."""

import logging
import math
import os
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("infer")


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
        delta = F.linear(
            F.linear(x.to(self.lora_down.dtype), self.lora_down), self.lora_up
        )
        return result + (delta * self.scale).to(result.dtype)


def _normalize_key(key):
    for prefix in (
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    for suffix, part in (
        (".lora_A.default.weight", "down"),
        (".lora_B.default.weight", "up"),
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
        (".lora.down.weight", "down"),
        (".lora.up.weight", "up"),
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".alpha", "alpha"),
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    raise ValueError(f"Unsupported Turbo-SLA tensor: {key}")


def install_lora(
    transformer, state_dict, *, default_alpha=128.0, has_adaln_cache=False, fuse=False
):
    """Validate every pair/shape before mutation; never silently skip a tensor.

    Native LightX2V keys already use the Diffusers layout. ComfyUI blocks/qkv
    and SwiGLU layouts deliberately fail target lookup rather than guessing.
    """
    pairs = {}
    for key, tensor in state_dict.items():
        target, part = _normalize_key(key)
        pair = pairs.setdefault(target, {})
        if part in pair:
            raise ValueError(
                f"Duplicate Turbo-SLA tensor after normalization: {target}.{part}"
            )
        pair[part] = tensor
    if not pairs:
        raise ValueError("Empty Turbo-SLA adapter.")
    replacements = []
    for name, pair in pairs.items():
        if "down" not in pair or "up" not in pair:
            raise ValueError(f"Incomplete LoRA pair: {name}")
        if has_adaln_cache and any(
            token in name for token in ("adaln", "time_embedder", "norm_out.linear")
        ):
            raise ValueError(
                f"LoRA changes cached AdaLN projections: {name}; disable the cache."
            )
        base = transformer.get_submodule(name)
        down, up = pair["down"], pair["up"]
        if not isinstance(base, nn.Linear) or base.weight.is_meta:
            raise ValueError(f"Turbo-SLA needs a materialized Linear target: {name}")
        if (
            down.ndim != 2
            or up.ndim != 2
            or down.shape[0] < 1
            or down.shape[0] != up.shape[1]
            or (up.shape[0], down.shape[1]) != tuple(base.weight.shape)
        ):
            raise ValueError(
                f"LoRA shape mismatch for {name}: down={tuple(down.shape)}, up={tuple(up.shape)}"
            )
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
        if fuse:
            with torch.no_grad():
                merged = torch.addmm(
                    base.weight.float(),
                    up.float(),
                    down.float(),
                    alpha=alpha / down.shape[0],
                )
                base.weight.copy_(merged)
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, child_name, TurboSLALinear(base, down, up, alpha))
    logger.info(
        "Turbo-SLA loaded %d Linear pairs; default alpha=%g, strength=1",
        len(replacements),
        default_alpha,
    )
    return len(replacements)


def training_euler_step(self, model_output, timestep, sample, return_dict=True):
    """Match LightX2V training_euler: x += (sigma - sigma_next) * velocity."""
    if self._step_index is None:
        self._step_index = (
            self.index_for_timestep(timestep)
            if self.begin_index is None
            else self.begin_index
        )
    sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=torch.float32)
    next_sigma = self.sigmas[self._step_index + 1].to(
        device=sample.device, dtype=torch.float32
    )
    prev_sample = sample.float() + (sigma - next_sigma) * model_output.float()
    self._step_index += 1
    if not return_dict:
        return (prev_sample,)
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput

    return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)
