from typing import Any, Optional

import peft
import torch
from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.aqlm import dispatch_aqlm
from peft.tuners.lora.awq import dispatch_awq
from peft.tuners.lora.eetq import dispatch_eetq
from peft.tuners.lora.gptq import dispatch_gptq
from peft.tuners.lora.hqq import dispatch_hqq
from peft.tuners.lora.layer import Conv2d, LoraLayer, dispatch_default
from peft.tuners.lora.torchao import dispatch_torchao
from peft.tuners.lora.tp_layer import dispatch_megatron
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

import abc
from x_base import WeightQuantLinearModule

# ---- PEFT version compatibility ----
# PEFT v0.19.0 changed the LoRA dispatch/update_layer API:
#   - dispatch functions: lora_config= → config=
#   - update_layer: individual params → config: LoraConfig object
_PEFT_VERSION = tuple(int(x) for x in peft.__version__.split(".")[:2])
_IS_NEW_PEFT_API = _PEFT_VERSION >= (0, 19)


class _CombinedMeta(abc.ABCMeta, type):
    """
    兼容元类，同时满足 abc.ABCMeta 和 type 的要求。
    用于解决 LoraWeightQuantLinearModule 同时继承 torch.nn.Module 和 LoraLayer 的元类冲突。
    """
    pass


class LoraWeightQuantLinearModule(torch.nn.Module, LoraLayer, metaclass=_CombinedMeta):
    def __init__(
            self,
            base_layer,
            adapter_name,
            config: Any = None,
            r: int = 0,
            lora_alpha: int = 1,
            **kwargs,
    ):
        if config is not None and config.use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer)

        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            config=config,
        )

    def forward(self, x: torch.Tensor):
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result = result + output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_wql(
        target: torch.nn.Module,
        adapter_name: str,
        config: Any = None,
        **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, WeightQuantLinearModule):
        # Raise the error only at the dispatch level
        new_module = LoraWeightQuantLinearModule(target, adapter_name, config=config, **kwargs)

    return new_module


@staticmethod
def _create_new_module_ascend(lora_config, adapter_name, target, **kwargs):
    # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
    # because the first match is always used. Therefore, the default layers should be checked last.
    dispatchers = []

    if lora_config._custom_modules:
        # Experimental custom LoRA module support. Allows users to pass a custom mapping for unsupported layer
        # types by impelementing their own LoRA layers.
        def dynamic_dispatch_func(target, adapter_name, config, **kwargs):
            new_module = None

            if isinstance(target, BaseTunerLayer):
                target_base_layer = target.get_base_layer()
            else:
                target_base_layer = target

            for key, custom_cls in config._custom_modules.items():
                if isinstance(target_base_layer, key):
                    new_module = custom_cls(target, adapter_name, config=config, **kwargs)
                    break

            return new_module

        dispatchers.append(dynamic_dispatch_func)

    dispatchers.extend(
        [
            dispatch_wql,
            dispatch_eetq,
            dispatch_aqlm,
            dispatch_awq,
            dispatch_gptq,
            dispatch_hqq,
            dispatch_torchao,
            dispatch_megatron,
            dispatch_default,
        ]
    )

    new_module = None
    for dispatcher in dispatchers:
        new_module = dispatcher(target, adapter_name, config=lora_config, **kwargs)
        if new_module is not None:  # first match wins
            break

    if new_module is None:
        # no module could be matched
        raise ValueError(
            f"Target module {target} is not supported. Currently, only the following modules are supported: "
            "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
            "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
        )

    return new_module


LoraModel._create_new_module = _create_new_module_ascend
_SET_ADAPTER_SCALE_FN_MAPPING["AscendWanTransformer3DModel"] = lambda model_cls, weights: weights
_SET_ADAPTER_SCALE_FN_MAPPING["MiniMaxH3Transformer3DModel"] = lambda model_cls, weights: weights
