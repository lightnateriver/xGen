"""Opt-in H3 Image DiT optimizations."""
import functools
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn


class ImageAdaLNTable(nn.Module):
    def __init__(self, edit, text, state):
        super().__init__()
        self.register_buffer("edit", edit)
        self.register_buffer("text", text)
        self.state = state

    def forward(self, temb):
        table = self.edit if temb.shape[0] == 2 else self.text
        return table[self.state["step"]].unbind(0)


def attach_image_adaln_step_hook(transformer, state):
    def select_step(module, args, kwargs):
        state["step"] = kwargs["pdd_step_index"]

    transformer.register_forward_pre_hook(select_step, with_kwargs=True)
    transformer._xgen_image_adaln_state = state


@torch.inference_mode()
def install_image_adaln_table(transformer, sample_sigmas, reference_noise_aug, device):
    """Evaluate each exact one/two-row timestep layout before releasing 26 GB of weights."""
    state = {"step": 0}
    timesteps = torch.tensor([1.0 - s for s in sample_sigmas[:-1]], dtype=torch.float32)
    tembs = {"edit": [], "text": []}
    transformer.time_embedder.to(device)
    for t in timesteps:
        for mode in tembs:
            values = torch.tensor([float(t), reference_noise_aug] if mode == "edit" else [float(t)], dtype=torch.float32, device=device)
            values = values.sort().values
            tembs[mode].append(transformer.time_embedder(transformer.time_proj(values)))
    for block in transformer.transformer_blocks:
        original = block.adaln_proj.to(device)
        tables = {mode: torch.stack([torch.stack(original(t)) for t in embeds]).cpu()
                  for mode, embeds in tembs.items()}
        block.adaln_proj = ImageAdaLNTable(tables["edit"], tables["text"], state)
        del original
        torch.get_device_module(device.type).empty_cache()

    attach_image_adaln_step_hook(transformer, state)


def _image_adaln_metadata(model_root, transformer, sigmas, reference_noise_aug):
    root = Path(model_root)
    dit = root / "transformer_diffusers"
    shards = sorted(dit.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No local DiT Safetensors weights found in {dit}")
    snapshot = []
    for path in (root / "pdd_config.json", dit / "config.json", *shards):
        stat = path.stat()
        snapshot.append([str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns])
    schedule = [list(map(float, sigmas)), float(reference_noise_aug), len(transformer.transformer_blocks)]
    return {
        "schema": "xgen-h3-image-adaln-v1",
        "snapshot": json.dumps(snapshot),
        "schedule": json.dumps(schedule),
    }


def save_image_adaln_table(path, model_root, transformer, sigmas, reference_noise_aug):
    from x_base.adaln_cache import save_adaln_tables

    tensors = {
        f"{index}.{mode}": getattr(block.adaln_proj, mode).contiguous()
        for index, block in enumerate(transformer.transformer_blocks)
        for mode in ("edit", "text")
    }
    metadata = _image_adaln_metadata(model_root, transformer, sigmas, reference_noise_aug)
    save_adaln_tables(path, tensors, metadata=metadata)


def load_image_adaln_table(path, model_root, transformer, sigmas, reference_noise_aug, device):
    from x_base.adaln_cache import open_adaln_tables

    blocks = transformer.transformer_blocks
    metadata = _image_adaln_metadata(model_root, transformer, sigmas, reference_noise_aug)
    state = {"step": 0}
    replacements = []
    expected_keys = {f"{index}.{mode}" for index in range(len(blocks)) for mode in ("edit", "text")}
    with open_adaln_tables(path, expected_metadata=metadata, expected_keys=expected_keys) as cached:
        for index, block in enumerate(blocks):
            projection = block.adaln_proj
            hidden = projection.hidden_size
            rows = projection.linear.out_features // (6 * hidden)
            tables = {}
            for mode, timestep_rows in (("edit", 2), ("text", 1)):
                table = cached.get_tensor(f"{index}.{mode}")
                expected = (len(sigmas) - 1, 6, rows * timestep_rows, hidden)
                if table.shape != expected or table.dtype != projection.linear.weight.dtype:
                    raise ValueError(f"AdaLN table {index}.{mode} has incompatible shape or dtype")
                tables[mode] = table
            replacements.append(ImageAdaLNTable(tables["edit"], tables["text"], state))
    transformer.time_embedder.to(device)
    for block, replacement in zip(blocks, replacements):
        block.adaln_proj = replacement
    attach_image_adaln_step_hook(transformer, state)


def install_image_condition_cache(transformer):
    """Cache only within one request: projected/refined condition and positional embedding."""
    state = {}
    for name in ("context_embedder", "token_refiner", "rope"):
        module = getattr(transformer, name)
        original = module.forward
        @functools.wraps(original)
        def forward(*args, _name=name, _original=original, **kwargs):
            if _name not in state:
                state[_name] = _original(*args, **kwargs)
            return state[_name]
        module.forward = forward
    transformer._xgen_image_condition_cache = state


def clear_image_condition_cache(transformer):
    getattr(transformer, "_xgen_image_condition_cache", {}).clear()
    getattr(transformer, "_xgen_image_io_cache", {}).clear()


@torch.no_grad()
def quantize_image_blocks(transformer):
    from x_base.config import QuantLayerConfig
    from x_base.operator.matmul import enable_weight_quant_linear
    config = QuantLayerConfig(include_patterns=["transformer_blocks.*.attn.*", "transformer_blocks.*.ff.*"])
    return enable_weight_quant_linear(transformer, config, dtype="w8a8")


@torch.no_grad()
def fuse_image_qkv(transformer):
    for block in transformer.transformer_blocks:
        attn = block.attn
        layers = (attn.to_q, attn.to_k, attn.to_v)
        fused = nn.Linear(layers[0].in_features, sum(x.out_features for x in layers), bias=False,
                          device=layers[0].weight.device, dtype=layers[0].weight.dtype)
        fused.weight.copy_(torch.cat([x.weight for x in layers]))
        attn.to_qkv = fused.requires_grad_(False)
        attn.fused_projections = True
        del attn.to_q, attn.to_k, attn.to_v


def install_image_refiner_sp(transformer):
    from x_base import set_pad, split_sequence, gather_sequence
    from .minimax_h3_image import H3ImageAttnProcessor
    manager = transformer._xgen_image_parallel_manager
    state = {}
    key = "h3_image_refiner"
    for block in transformer.token_refiner.refiner_blocks:
        block.attn.set_processor(H3ImageAttnProcessor(manager, pad_key=key))

    def split(module, args):
        x = args[0]
        state["pad"] = set_pad(key, x.shape[1], manager.sp_group)
        return (split_sequence(x, manager.sp_group, dim=1, pad=state["pad"]),)

    def gather(module, args, output):
        return gather_sequence(output, manager.sp_group, dim=1, pad=state["pad"])
    transformer.token_refiner.register_forward_pre_hook(split)
    transformer.token_refiner.register_forward_hook(gather)


class ImageRowParallel(nn.Module):
    def __init__(self, linear, group):
        super().__init__()
        self.linear, self.group = linear, group

    def forward(self, x):
        import torch.distributed as dist
        output = self.linear(x)
        dist.all_reduce(output, group=self.group)
        return output


def install_image_dit_tp(transformer, group):
    """Shard attention heads and packed SwiGLU channels; AdaLN and the six heads stay replicated."""
    import torch.distributed as dist
    rank, world = dist.get_rank(group), dist.get_world_size(group)

    def col(linear, parts=1):
        rows = linear.weight.shape[0]
        if rows % (parts * world):
            raise ValueError("Packed projection rows must divide evenly across TP ranks")
        rows_per_part = rows // parts
        rows_per_rank = rows_per_part // world
        rank_rows = torch.arange(rank * rows_per_rank, (rank + 1) * rows_per_rank)
        indices = (torch.arange(parts)[:, None] * rows_per_part + rank_rows).flatten()
        linear.weight = nn.Parameter(linear.weight.index_select(0, indices).contiguous(), requires_grad=False)
        if hasattr(linear, "antiquant_scale"):
            linear.antiquant_scale = nn.Parameter(linear.antiquant_scale.index_select(0, indices).contiguous(), requires_grad=False)
        linear.out_features = linear.weight.shape[0]

    def row(linear):
        part = linear.weight.shape[1] // world
        linear.weight = nn.Parameter(linear.weight[:,rank*part:(rank+1)*part].contiguous(), requires_grad=False)
        linear.in_features = part
        return ImageRowParallel(linear, group)

    for block in transformer.transformer_blocks:
        attn = block.attn
        if attn.fused_projections:
            col(attn.to_qkv, parts=3)
        else:
            for linear in (attn.to_q, attn.to_k, attn.to_v):
                col(linear)
        attn.to_out[0] = row(attn.to_out[0])
        col(block.ff.net[0].proj, parts=2)
        block.ff.net[2] = row(block.ff.net[2])


def install_image_ff_chunk(transformer, chunk_size=2048):
    """Bound FFN temporaries; attention still sees the entire document."""
    for block in list(transformer.transformer_blocks) + list(transformer.token_refiner.refiner_blocks):
        original = block.ff.forward
        @functools.wraps(original)
        def forward(x, *args, _original=original, **kwargs):
            if x.shape[1] <= chunk_size:
                return _original(x, *args, **kwargs)
            result = torch.empty_like(x)
            for start in range(0, x.shape[1], chunk_size):
                end = min(start + chunk_size, x.shape[1])
                result[:, start:end] = _original(x[:, start:end], *args, **kwargs)
            return result
        block.ff.forward = forward


def install_image_swiglu(transformer, backend):
    if backend not in ("native", "triton"):
        raise ValueError("Unknown Image SwiGLU backend")
    for block in transformer.transformer_blocks:
        module = block.ff.net[0]
        gate_first = backend == "native"
        if getattr(module, "_xgen_gate_first", False) != gate_first:
            linear = module.proj
            for name in ("weight", "antiquant_scale"):
                if hasattr(linear, name):
                    tensor = getattr(linear, name)
                    left, right = tensor.chunk(2, dim=0)
                    setattr(linear, name, nn.Parameter(torch.cat((right, left), dim=0), requires_grad=False))
            module._xgen_gate_first = gate_first
        if backend == "native":
            def forward(self, x):
                return torch.ops.npu.npu_swiglu(self.proj(x), dim=-1)
        else:
            from .h3_image_triton import image_swiglu
            def forward(self, x):
                return image_swiglu(self.proj(x))
        module.forward = MethodType(forward, module)


def install_image_local_input(transformer):
    """Project only image rows owned by this SP rank; never materialize the full-width packed buffer."""
    import torch.distributed as dist
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    manager = transformer._xgen_image_parallel_manager
    if manager is None:
        raise ValueError("Image local input requires sequence parallelism")
    rank, world = dist.get_rank(manager.sp_group), dist.get_world_size(manager.sp_group)
    cache = transformer._xgen_image_io_cache = {}
    transformer._xgen_image_local_input = True

    def forward(self, hidden_states, encoder_hidden_states, timestep, timestep_indices, token_tags,
                position_ids, image_indices, text_indices, pdd_step_index, return_dict=True):
        length = position_ids.shape[0]
        local_length = (length + world - 1) // world
        start, end = rank * local_length, min((rank + 1) * local_length, length)
        local_images = torch.nonzero((image_indices >= start) & (image_indices < end)).flatten()
        local_text = torch.nonzero((text_indices >= start) & (text_indices < end)).flatten()
        use_cache = hasattr(self, "_xgen_image_condition_cache")
        if use_cache and "text" in cache:
            text_embeds = cache["text"]
        else:
            text = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
            text = self.token_refiner(text)
            text_embeds = text.index_select(1, local_text)
            if use_cache:
                cache["text"] = text_embeds
                # The local refined rows replace the larger intermediate conditioning caches.
                for name in ("context_embedder", "token_refiner"):
                    self._xgen_image_condition_cache.pop(name, None)
            del text
        packed = text_embeds.new_zeros((hidden_states.shape[0], local_length, text_embeds.shape[-1]))
        if local_text.numel():
            packed.index_copy_(1, text_indices.index_select(0, local_text) - start, text_embeds)
        if local_images.numel():
            image_embeds = self.proj_in(hidden_states.index_select(1, local_images).to(get_parameter_dtype(self.proj_in)))
            packed.index_copy_(1, image_indices.index_select(0, local_images) - start, image_embeds.to(packed.dtype))
            del image_embeds
        rotary_emb = self.rope(position_ids)
        temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * 3 + token_tags
        for block in self.transformer_blocks:
            packed = block(packed, temb, adaln_indices, rotary_emb)
        head = self.pdd_proj_out[pdd_step_index]
        output = head(self.norm_out(packed, temb, timestep_indices).to(get_parameter_dtype(head)))
        output = output.index_select(1, image_indices)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)
    transformer.forward = MethodType(forward, transformer)
