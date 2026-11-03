# SPDX-License-Identifier: Apache-2.0
"""Prepacked H3 linear using the Comfy-Kitchen NVFP4 operator contract."""

import json

import torch
from torch import nn


def interpolate_h3_adaln(timestep, table):
    """Evaluate the normalized FastVideo H3 timestep curve in FP32."""
    position = timestep.float().clamp(0, 1) * (table.shape[0] - 1)
    lower = position.floor().long().clamp_max(table.shape[0] - 2)
    fraction = (position - lower).unsqueeze(-1)
    return torch.lerp(table[lower], table[lower + 1], fraction)


class H3GateFirstSwiGLU(nn.Module):
    """Apply the checkpoint's gate-first layout without repacking FP4 rows."""

    def __init__(self, proj):
        super().__init__()
        self.proj = proj
        self.fused_swiglu = False

    def forward(self, hidden_states):
        projected = self.proj(hidden_states)
        if (
            self.fused_swiglu and not torch.is_grad_enabled() and projected.is_cuda
            and projected.dtype == torch.bfloat16 and projected.is_contiguous()
            and projected.numel() > 0 and projected.shape[-1] % 2 == 0
        ):
            from .h3_swiglu import gate_first_swiglu

            return gate_first_swiglu(projected)
        gate, value = projected.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * value


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
        self.fused_amax = False

    def forward(self, x, *, activation_amax=None):
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("NVFP4 activations require CUDA FP16 or BF16")
        if x.shape[-1] != self.in_features:
            raise ValueError("NVFP4 input dimension does not match the packed weight")
        # Same scaling and packed GEMM contract as the FastVideo H3 recipe.
        import comfy_kitchen

        matrix = x.reshape(-1, self.in_features).contiguous()
        values = matrix.detach()
        if activation_amax is not None:
            amax = activation_amax
        elif self.fused_amax and x.dtype == torch.bfloat16 and values.numel() > 0:
            from .h3_abs_peak import activation_abs_peak

            amax = activation_abs_peak(values)
        else:
            amax = values.abs().amax()
        scale = (amax.float() / (448.0 * 6.0)).clamp_min(
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


def fused_h3_ff_forward(self, hidden_states):
    if not self._h3_fused_ff_amax or self.training or torch.is_grad_enabled():
        return self._h3_eager_ff_forward(hidden_states)
    from .h3_swiglu import gate_first_swiglu

    activation, dropout, down_proj = self.net
    projected = activation.proj(hidden_states)
    if (
        not projected.is_cuda or projected.dtype != torch.bfloat16
        or not projected.is_contiguous() or projected.numel() == 0 or projected.shape[-1] % 2
    ):
        gate, value = projected.chunk(2, dim=-1)
        return down_proj(dropout(torch.nn.functional.silu(gate) * value))
    activated, amax = gate_first_swiglu(projected, with_amax=True)
    return down_proj(dropout(activated), activation_amax=amax)

# Main-block aliases from the FastVideo pruned H3 checkpoint to Diffusers.
PACKED_PROJECTIONS = {
    "attn.qkv_proj": "attn.to_qkv",
    "attn.out_proj": "attn.to_out.0",
    "mlp.fc1": "ff.net.0.proj",
    "mlp.fc2": "ff.net.2",
}


def read_packed_h3_projections(checkpoint, num_blocks=50):
    """Read the packed linear submodules without materializing dense weights."""
    from safetensors import safe_open

    projections = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        expected = {
            f"blocks.{layer}.{name}.weight"
            for layer in range(num_blocks)
            for name in PACKED_PROJECTIONS
        }
        tagged = {k.removesuffix(".comfy_quant") + ".weight" for k in keys if k.endswith(".comfy_quant")}
        if expected != tagged:
            raise ValueError(
                f"Unexpected packed H3 projection set: missing={sorted(expected-tagged)[:4]}, "
                f"extra={sorted(tagged-expected)[:4]}"
            )
        for layer in range(num_blocks):
            for original, target in PACKED_PROJECTIONS.items():
                prefix = f"blocks.{layer}.{original}"
                tag = json.loads(bytes(source.get_tensor(prefix + ".comfy_quant").tolist()))
                if tag.get("format") != "nvfp4":
                    raise ValueError(f"{prefix} is not tagged as NVFP4")
                bias = source.get_tensor(prefix + ".bias") if prefix + ".bias" in keys else None
                projections[f"transformer_blocks.{layer}.{target}"] = H3NVFP4Linear(
                    source.get_tensor(prefix + ".weight"),
                    source.get_tensor(prefix + ".weight_scale"),
                    source.get_tensor(prefix + ".weight_scale_2"),
                    bias,
                )
    return projections
