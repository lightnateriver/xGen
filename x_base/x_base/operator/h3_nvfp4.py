# SPDX-License-Identifier: Apache-2.0
"""Prepacked H3 linear using the Comfy-Kitchen NVFP4 operator contract."""

import torch
from torch import nn


class H3NVFP4Linear(nn.Module):
    """Keep the checkpoint in packed E2M1 form across offload and forward."""

    def __init__(self, weight, block_scale, global_scale, bias=None):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.uint8:
            raise ValueError("NVFP4 weights must be a matrix of packed uint8 values")
        self.out_features, packed_in = weight.shape
        self.in_features = packed_in * 2
        rows = (self.out_features + 127) // 128 * 128
        cols = (self.in_features + 63) // 64 * 4
        if block_scale.shape != (rows, cols) or block_scale.dtype != torch.float8_e4m3fn:
            raise ValueError("NVFP4 block scales must use the padded E4M3 layout")
        if global_scale.shape != () or global_scale.dtype != torch.float32:
            raise ValueError("NVFP4 global scale must be a scalar float32 tensor")
        if bias is not None and bias.shape != (self.out_features,):
            raise ValueError("NVFP4 bias must match the output dimension")
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("block_scale", block_scale.contiguous())
        self.register_buffer("global_scale", global_scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("NVFP4 activations require CUDA FP16 or BF16")
        if x.shape[-1] != self.in_features:
            raise ValueError("NVFP4 input dimension does not match the packed weight")
        # Same scaling and packed GEMM contract as the FastVideo H3 recipe.
        import comfy_kitchen

        matrix = x.reshape(-1, self.in_features).contiguous()
        scale = (matrix.detach().abs().amax().float() / (448.0 * 6.0)).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        packed, block_scale = comfy_kitchen.quantize_nvfp4(matrix, scale, pad_16x=True)
        output = comfy_kitchen.scaled_mm_nvfp4(
            packed, self.weight,
            tensor_scale_a=scale, tensor_scale_b=self.global_scale,
            block_scale_a=block_scale, block_scale_b=self.block_scale,
            bias=self.bias, out_dtype=x.dtype,
        )
        return output[:matrix.shape[0], :self.out_features].reshape(
            *x.shape[:-1], self.out_features
        )
