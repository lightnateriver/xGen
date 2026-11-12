"""Opt-in Image optimizations; preserve the released six-step numeric path."""
import functools
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

    def select_step(module, args, kwargs):
        state["step"] = kwargs["pdd_step_index"]
    transformer.register_forward_pre_hook(select_step, with_kwargs=True)
    transformer._xgen_image_adaln_state = state


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
def quantize_image_text_encoder(text_encoder):
    """W8A8 retained Qwen decoder projections after optional TP sharding, before offload."""
    from x_base.config import QuantLayerConfig
    from x_base.operator.matmul import WeightQuantLinearModule, enable_weight_quant_linear
    from x_base.tensor_parallelism.layers import ColumnParallelLinear, RowParallelLinear
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    names = [f"model.language_model.layers.{i}.{projection}"
             for i in range(len(text_encoder.model.language_model.layers)) for projection in projections]
    types = (nn.Linear, ColumnParallelLinear, RowParallelLinear)
    if not names or any(not isinstance(text_encoder.get_submodule(name), types) for name in names):
        raise ValueError("Image text quantization requires Qwen decoder Linear or TP projections")
    before = sum(p.numel() * p.element_size() for p in text_encoder.parameters())
    plain = [name for name in names if isinstance(text_encoder.get_submodule(name), nn.Linear)]
    if plain:
        enable_weight_quant_linear(text_encoder, QuantLayerConfig(include_patterns=plain), dtype="w8a8")
    tp_count = 0
    for name in names:
        layer = text_encoder.get_submodule(name)
        if not isinstance(layer, (ColumnParallelLinear, RowParallelLinear)):
            continue
        # Quantize the already-owned weight shard. Keep the TP module, group and bias.
        quant = WeightQuantLinearModule(layer.weight.detach().cpu(), None)
        layer.weight = quant.weight
        layer.register_parameter("antiquant_scale", quant.antiquant_scale)
        layer.forward = MethodType(_image_text_column_quant_forward if isinstance(layer, ColumnParallelLinear)
                                   else _image_text_row_quant_forward, layer)
        layer._xgen_image_text_quantized_tp = True
        tp_count += 1
        del quant
        torch.npu.empty_cache()
    if any(text_encoder.get_submodule(name).weight.dtype != torch.int8 for name in names):
        raise RuntimeError("Some Image text decoder projections were not quantized")
    text_encoder._xgen_image_text_quantization = dict(
        dtype="dynamic-w8a8", projection_count=len(names), tp_projection_count=tp_count, parameter_bytes_before=before,
        parameter_bytes_after=sum(p.numel() * p.element_size() for p in text_encoder.parameters()),
        scope="Retained decoder attention/MLP linears only; embedding, norm, vision and lm_head unchanged")


def _image_text_tp_quant_linear(layer, x, include_bias):
    from types import SimpleNamespace
    from x_base.operator.matmul import WeightQuantLinearModule
    for name in ("weight", "antiquant_scale", "bias"):
        value = getattr(layer, name)
        if value is not None and value.device != x.device:
            value.data = value.data.to(x.device)
    proxy = SimpleNamespace(weight=layer.weight, antiquant_scale=layer.antiquant_scale,
                            bias=layer.bias if include_bias else None)
    return WeightQuantLinearModule.forward(proxy, x)


def _image_text_column_quant_forward(self, x):
    import torch.distributed as dist
    output = _image_text_tp_quant_linear(self, x, True)
    if self.gather_output and self._tp_world_size() > 1:
        outputs = [torch.empty_like(output) for _ in range(self._tp_world_size())]
        dist.all_gather(outputs, output, group=self._tp_group())
        output = torch.cat(outputs, dim=-1)
    return output


def _image_text_row_quant_forward(self, x):
    import torch.distributed as dist
    if not self.input_is_parallel and self._tp_world_size() > 1:
        chunk = x.shape[-1] // self._tp_world_size()
        x = x[..., self._tp_rank() * chunk:(self._tp_rank() + 1) * chunk]
    # Row-parallel bias is added once, after the reduction, exactly as in BF16 TP.
    output = _image_text_tp_quant_linear(self, x, False)
    if self._tp_world_size() > 1:
        if self.reduce_fn is None:
            dist.all_reduce(output, group=self._tp_group())
        else:
            output = self.reduce_fn(output, self._tp_group())
    if self.bias is not None:
        output = output + self.bias
    return output


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


def install_image_vae_parallel(vae, group):
    """Distribute original spatial tiles, retaining the exact upstream blend order and dtype."""
    import torch.distributed as dist
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    originals = {"encode": vae._encode, "decode": vae._decode}

    def tiled(self, x, mode):
        if not self.use_tiling or not getattr(self, "_xgen_tile_parallel_enabled", True):
            return originals[mode](x)
        ratio = self.spatial_compression_ratio
        height, width = x.shape[-2:]
        if mode == "decode":
            height, width = height * ratio, width * ratio
        ys, yl, yo = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        xs, xl, xo = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        if len(ys) * len(xs) < world:
            return originals[mode](x)
        positions = [(y, h, xx, w) for y, h in zip(ys, yl) for xx, w in zip(xs, xl)]
        # All tiles in a given image have equal shape, including images below the tile size on one axis.
        output = []
        for start in range(0, len(positions), world):
            index = min(start + rank, len(positions) - 1)
            y, h, xx, w = positions[index]
            if mode == "encode":
                tile = x[..., y:y+h, xx:xx+w]
                local = self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1].contiguous()
            else:
                tile = x[..., y//ratio:(y+h)//ratio, xx//ratio:(xx+w)//ratio]
                local = self.decoder(self.post_quant_conv(tile)).contiguous()
            gathered = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(gathered, local, group=group)
            output.extend(gathered[:min(world, len(positions)-start)])
        rows = [output[i:i+len(xs)] for i in range(0, len(output), len(xs))]
        if mode == "encode":
            yo, xo = [n//ratio for n in yo], [n//ratio for n in xo]
        return self._stitch_tiles(rows, yo, xo)

    vae._encode = MethodType(lambda self, x: tiled(self, x, "encode"), vae)
    vae._decode = MethodType(lambda self, x: tiled(self, x, "decode"), vae)
    vae._xgen_tile_parallel_enabled = True


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

    def col(linear, packed=False):
        rows = linear.weight.shape[0]
        if packed:
            half = rows // 2; part = half // world
            indices = torch.cat([torch.arange(rank*part,(rank+1)*part), torch.arange(half+rank*part,half+(rank+1)*part)])
        else:
            indices = torch.arange(rank*(rows//world),(rank+1)*(rows//world))
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
            raise ValueError("TP and fused QKV are separate experiments")
        for linear in (attn.to_q, attn.to_k, attn.to_v):col(linear)
        attn.to_out[0] = row(attn.to_out[0])
        col(block.ff.net[0].proj, packed=True)
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


def install_image_rgb8_output(pipe):
    """Transport the released quantized pixels as RGB8, avoiding a float32 D2H round trip."""
    from PIL import Image
    processor = pipe.image_processor
    original = processor.postprocess
    @functools.wraps(original)
    def postprocess(image, output_type="pil", do_denormalize=None):
        if image.device.type != "npu" or output_type != "pil" or processor.config.do_normalize or (do_denormalize is not None and any(do_denormalize)):
            return original(image, output_type=output_type, do_denormalize=do_denormalize)
        # DecodeStep already applies the checkpoint's truncating quantizer before this call.
        pixels = (image * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [Image.fromarray(item) for item in pixels]
    processor.postprocess = postprocess


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
