# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-tensor W8A8 projections using PyTorch's CUDA scaled GEMM."""

import torch
from torch import nn
import torch.nn.functional as F


def quantize_per_tensor(value):
    maximum = torch.finfo(torch.float8_e4m3fn).max
    scale = (value.float().abs().amax() / maximum).clamp_min(torch.finfo(torch.float32).tiny)
    return (value.float() / scale).clamp(-maximum, maximum).to(torch.float8_e4m3fn), scale


class H3FP8Linear(nn.Module):
    """Keep FP8 weights; round only CUDA GEMM activations to E4M3."""

    def __init__(self, weight, bias=None):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 16:
            raise ValueError("H3 FP8 requires projection dimensions divisible by 16")
        self.out_features, self.in_features = weight.shape
        packed, scale = quantize_per_tensor(weight.detach())
        self.register_buffer('weight', packed.contiguous())
        self.register_buffer('weight_scale', scale)
        self.register_buffer('bias', bias.detach().clone() if bias is not None else None)
        self.calls = 0

    @classmethod
    def from_packed(cls, weight, scale, bias=None):
        """Reuse already quantized bytes and their original scale without requantization."""
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or any(n % 16 for n in weight.shape):
            raise ValueError('Packed H3 FP8 projection requires an aligned E4M3 matrix')
        if scale.numel() != 1 or scale.dtype != torch.float32:
            raise ValueError('Packed H3 FP8 projection requires one FP32 weight scale')
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.out_features, result.in_features = weight.shape
        result.register_buffer('weight', weight.contiguous())
        result.register_buffer('weight_scale', scale.detach().clone())
        result.register_buffer('bias', bias.detach().clone() if bias is not None else None)
        result.calls = 0
        return result

    def forward(self, x):
        return self._local_forward(x, self.bias)

    def _local_forward(self, x, bias):
        if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("H3 FP8 Linear requires CUDA BF16/FP16 input")
        if x.shape[-1] != self.in_features:
            raise ValueError("H3 FP8 Linear input width does not match its weight")
        matrix = x.reshape(-1, self.in_features)
        if not matrix.shape[0]:
            return x.new_empty((*x.shape[:-1], self.out_features))
        from .h3_fp8_quant import quantize_activation

        padding=(-matrix.shape[0]) % 16
        padded = F.pad(matrix, (0, 0, 0, padding)) if padding else matrix.contiguous()
        activation_group = getattr(self, 'activation_group', None)
        packed, scale = (quantize_activation(padded) if activation_group is None
                         else quantize_activation(padded, group=activation_group))
        output = torch._scaled_mm(
            packed.contiguous(), self.weight.t(), scale_a=scale, scale_b=self.weight_scale,
            bias=bias, out_dtype=x.dtype, use_fast_accum=False,
        )
        self.calls += 1
        return output[:matrix.shape[0]].reshape(*x.shape[:-1], self.out_features)


class H3FP8TPLinear(H3FP8Linear):
    """Retain xGen's existing column/row TP communication after local FP8 GEMM."""

    def __init__(self, original):
        from x_base.tensor_parallelism.layers import ColumnParallelLinear

        super().__init__(original.weight, original.bias)
        self.column_parallel = isinstance(original, ColumnParallelLinear)
        self.gather_output = getattr(original, 'gather_output', False)
        self.input_is_parallel = getattr(original, 'input_is_parallel', True)
        self.tp_group = getattr(original, 'tp_group', None)
        self.reduce_fn = getattr(original, 'reduce_fn', None)

    def forward(self, x):
        import torch.distributed as dist
        from x_base.tensor_parallelism.state import get_tp_group, get_tp_rank, get_tp_world_size

        group = self.tp_group
        world = dist.get_world_size(group) if self.tp_group is not None else get_tp_world_size()
        if world > 1 and group is None:
            group = get_tp_group()
        if not self.column_parallel and not self.input_is_parallel and world > 1:
            rank = dist.get_rank(group) if self.tp_group is not None else get_tp_rank()
            x = x.chunk(world, dim=-1)[rank]
        # Row bias belongs after the reduction, as in RowParallelLinear.
        bias = self.bias
        output = self._local_forward(x, bias if self.column_parallel else None)
        if world > 1:
            if self.column_parallel and self.gather_output:
                values = [torch.empty_like(output) for _ in range(world)]
                dist.all_gather(values, output, group=group)
                output = torch.cat(values, dim=-1)
            elif not self.column_parallel:
                if self.reduce_fn is None:
                    dist.all_reduce(output, group=group)
                else:
                    output = self.reduce_fn(output, group)
        return output + bias if not self.column_parallel and bias is not None else output
