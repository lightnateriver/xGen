"""Kernel capability checks, lazy loading and bounded diagnostic logging."""
from collections import Counter
import importlib
import json
import logging
import os

import torch


LOG = logging.getLogger("xgen.optim")


def configure_logging():
    """Compatibility helper: handlers and levels belong to the application."""


class Diagnostics:
    def __init__(self, prefix="OP_OPT"):
        self.prefix = prefix
        self.seen = set()
        self.persistent = Counter()
        self.in_request = False
        self.reset()

    def reset(self):
        self.hits = Counter()
        self.fallbacks = self.persistent.copy()

    def hit(self, feature, detail=""):
        self.hits[feature] += 1
        key = (feature, "enabled")
        if key not in self.seen:
            self.seen.add(key)
            LOG.info("[%s] rank=%s enabled=%s %s", self.prefix, os.environ.get("RANK", "0"), feature, detail)

    def fallback(self, feature, reason):
        self.fallbacks[(feature, reason)] += 1
        if not self.in_request:
            self.persistent[(feature, reason)] += 1
        key = (feature, reason)
        if key not in self.seen:
            self.seen.add(key)
            LOG.warning("[%s] rank=%s fallback=%s reason=%s", self.prefix, os.environ.get("RANK", "0"), feature, reason)
        return False

    def summary(self, phase="request_end"):
        data = {"phase": phase, "rank": os.environ.get("RANK", "0"), "hits": dict(self.hits),
                "fallbacks": [{"feature": feature, "reason": reason, "count": count}
                              for (feature, reason), count in sorted(self.fallbacks.items())]}
        LOG.info("[%s] summary=%s", self.prefix, json.dumps(data, ensure_ascii=False, sort_keys=True))
        return data


def tensors_ok(*values, dtype=torch.bfloat16):
    return (not torch.is_grad_enabled() and bool(values)
            and all(isinstance(x, torch.Tensor) and x.dtype == dtype and not x.requires_grad
                    and x.device.type in ("npu", "privateuseone") for x in values)
            and all(x.device == values[0].device for x in values))


def no_hooks(module):
    """Do not bypass offload, adapter or user hooks on a skipped module."""
    global_hooks = torch.nn.modules.module
    return (not any(getattr(global_hooks, name, None) for name in
                    ("_global_forward_hooks", "_global_forward_pre_hooks"))
            and not any(getattr(module, name, None) for name in
                    ("_hf_hook", "_diffusers_hook", "_forward_hooks", "_forward_pre_hooks",
                     "peft_config", "lora_A", "lora_B"))
            and "forward" not in vars(module))


def quant_linear(module, width=None, device=None):
    from x_base.operator.matmul import WeightQuantLinearModule
    weight = getattr(module, "weight", None)
    scale = getattr(module, "antiquant_scale", None)
    bias = getattr(module, "bias", None)
    return (type(module) is WeightQuantLinearModule and no_hooks(module)
            and isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.dtype == torch.int8
            and isinstance(scale, torch.Tensor) and scale.dtype == torch.float32
            and scale.numel() == weight.shape[0]
            and (width is None or weight.shape[1] == width)
            and (device is None or weight.device == scale.device == device)
            and (bias is None or device is None or bias.device == device))


def quant_matmul(q, scale, linear, shape):
    import torch_npu
    return torch_npu.npu_quant_matmul(q, linear.weight.t(), linear.antiquant_scale,
                                    bias=linear.bias, pertoken_scale=scale,
                                    output_dtype=torch.bfloat16).reshape(*shape[:-1], linear.weight.shape[0])


class Kernels:
    SPECS = {
        "swiglu": ("swiglu_quant", "reuse"),
        "gate_residual": ("gate_residual", None),
        "norm_quant": ("nq2", True),
        "rope": ("partial_rope", None),
        "phaa_layout": ("phaa_layout", None),
        "laser_preprocess": ("phaa_preprocess", None),
        "merge_quant": ("merge_quant", None),
    }

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        self.loaded = {}
        self.arch = {}

    def get(self, feature, value):
        """Only loading failures fall back; errors after kernel dispatch propagate."""
        device = str(value.device)
        if device not in self.arch:
            try:
                name = torch.npu.get_device_name(value.device)
                self.arch[device] = str(name).startswith("Ascend910B")
            except (AttributeError, RuntimeError):
                self.arch[device] = False
        if not self.arch[device]:
            self.diagnostics.fallback(feature, "compiled kernel requires dav-2201 / Ascend910B")
            return None
        if feature not in self.loaded:
            module_name, argument = self.SPECS[feature]
            module = importlib.import_module("x_base.operator.npu_fusion." + module_name)
            path = module.library_path() if argument is None else module.library_path(argument)
            reason = None
            if not path.is_file():
                reason = f"library missing: {path.name}"
            else:
                try:
                    from x_base.operator.npu_fusion.loader import load_library
                    load_library(path)
                except (OSError, RuntimeError) as error:
                    reason = f"library load failed: {type(error).__name__}: {error}"
            self.loaded[feature] = (module, reason)
        module, reason = self.loaded[feature]
        if reason:
            self.diagnostics.fallback(feature, reason)
            return None
        return module
