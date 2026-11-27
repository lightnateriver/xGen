"""Diffusers hook-aware residency mechanism, independent of model selection."""

import torch


def retain_module(module, device):
    """Exclude complete modules from the existing Diffusers offloader."""
    for child in list(module.modules()):
        registry = getattr(child, "_diffusers_hook", None)
        if registry is None:
            continue
        for name in (
            "group_offloading",
            "lazy_prefetch_group_offloading",
            "layer_execution_tracker",
        ):
            hook = registry.get_hook(name)
            if hook is not None:
                group = getattr(hook, "group", None)
                if group is not None:
                    group.offload_()
                registry.remove_hook(name, recurse=False)
    module.to(device)
