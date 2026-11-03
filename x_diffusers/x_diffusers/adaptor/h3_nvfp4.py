# SPDX-License-Identifier: Apache-2.0
"""Load the FastVideo pruned NVFP4 H3 checkpoint into the XGEN DiT."""

import torch
from torch import nn
from safetensors import safe_open

from x_base.operator.h3_nvfp4 import H3GateFirstSwiGLU, read_packed_h3_projections


_ALIASES = {
    "video_patch_proj": "proj_in",
    "audio_patch_proj": "audio_proj_in",
    "condition_proj": "context_embedder",
    "blocks": "transformer_blocks",
    "token_refiner.blocks": "token_refiner.refiner_blocks",
    "final_layer.norm": "norm_out.norm",
    "final_layer.adaln_proj.linear": "norm_out.linear",
    "final_layer.video_out": "proj_out",
    "final_layer.audio_out": "audio_proj_out",
}
_BLOCK_ALIASES = {
    ".attn.qkv_proj.": ".attn.to_qkv.",
    ".attn.out_proj.": ".attn.to_out.0.",
    ".attn.q_norm.": ".attn.norm_q.",
    ".attn.k_norm.": ".attn.norm_k.",
    ".mlp.fc1.": ".ff.net.0.proj.",
    ".mlp.fc2.": ".ff.net.2.",
}


@torch.no_grad()
def load_packed_h3(model_cls, config, checkpoint):
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        table = source.get_tensor("adaln_t_table")
        if table.shape != (1025, 8) or table.dtype != torch.float32:
            raise ValueError("Expected a 1025 by 8 FP32 H3 AdaLN curve")

        with torch.device("meta"):
            model = model_cls.from_config(config, time_embed_dim=table.shape[1])
            del model.time_embedder
            for block in [*model.transformer_blocks, *model.token_refiner.refiner_blocks]:
                block.attn.fuse_projections()
                del block.attn.to_q, block.attn.to_k, block.attn.to_v
                block.ff.net[0] = H3GateFirstSwiGLU(block.ff.net[0].proj)

        packed = read_packed_h3_projections(checkpoint, len(model.transformer_blocks))
        packed_buffers = set()
        for name, module in packed.items():
            original = model.get_submodule(name)
            if (original.in_features, original.out_features) != (module.in_features, module.out_features):
                raise ValueError(f"Packed H3 projection shape mismatch for {name}")
            model.set_submodule(name, module)
            packed_buffers.update(f"{name}.{key}" for key in module.state_dict())

        for block in model.transformer_blocks:
            block.adaln_proj._h3_packed_adaln = True
        model.norm_out._h3_packed_adaln = True
        model.register_buffer("_h3_packed_adaln_table", table, persistent=False)
        model.rope.inv_freq = source.get_tensor("rope.inv_freq")

        expected = model.state_dict()
        dense = {}
        for key in source.keys():
            if key in {"adaln_t_table", "rope.inv_freq"}:
                continue
            name = key
            for original, target in _ALIASES.items():
                if name.startswith(original + "."):
                    name = target + name[len(original):]
                    break
            for original, target in _BLOCK_ALIASES.items():
                if original in name:
                    name = name.replace(original, target, 1)
                    break
            module_name, _, suffix = name.rpartition(".")
            if module_name in packed and suffix in {"weight", "bias", "weight_scale", "weight_scale_2", "comfy_quant"}:
                continue
            if name not in expected or name in dense:
                raise ValueError(f"Unexpected H3 checkpoint tensor {key} mapped to {name}")
            value = source.get_tensor(key)
            dtype = torch.float32 if ".adaln_proj.linear." in name or name.startswith("norm_out.linear.") else torch.bfloat16
            dense[name] = value.to(dtype)
        missing = expected.keys() - dense.keys() - packed_buffers
        if missing:
            raise ValueError(f"Missing H3 checkpoint tensors: {sorted(missing)[:8]}")
        model.load_state_dict(dense, strict=False, assign=True)
    model.requires_grad_(False)
    return model.eval()
