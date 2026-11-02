import torch
from torch import nn
import torch_npu
import logging

logging.basicConfig(level=logging.INFO)

DYNAMIC_QUANT_SKIPPED_CASE = ["time_embedding", "time_projection", "text_embedding", "head"]


class WeightQuantLinearModule(nn.Module):
    def __init__(self, weight, bias, name=None):
        super(WeightQuantLinearModule, self).__init__()
        quant_weight, pertoken_weight_scale = torch_npu.npu_dynamic_quant(weight.npu())
        del weight
        self.register_parameter('weight', torch.nn.Parameter(quant_weight.cpu(), requires_grad=False))
        self.register_parameter('antiquant_scale', torch.nn.Parameter(pertoken_weight_scale.cpu(), requires_grad=False))
        self.register_parameter('bias', torch.nn.Parameter(bias.cpu(), requires_grad=False))
        self.out_features, self.in_features = quant_weight.shape

    def forward(self, x):
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])
        x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        out = torch_npu.npu_quant_matmul(x_int8, self.weight.t(), self.antiquant_scale, bias=self.bias,
                                         pertoken_scale=pertoken_scale, output_dtype=ori_dtype)
        out = out.reshape(*ori_shape[:-1], -1)
        return out


def enable_weight_quant_linear(model):
    for name, layer in model.named_modules():
        if (isinstance(layer, nn.Linear) or isinstance(layer, WeightQuantLinearModule)) and not any(skip_case in name for skip_case in DYNAMIC_QUANT_SKIPPED_CASE):
            quant_module = WeightQuantLinearModule(layer.weight, layer.bias, name)
            submodules, layer_name = name.split('.')[:-1], name.split('.')[-1]
            setattr(model.get_submodule('.'.join(submodules)), layer_name, quant_module)
            logging.info(f"quant module:{submodules}, {layer_name}")
            del layer.weight
            del layer.bias
            del layer
    return model