"""Instance-local MiniMax-H3-Image attention and Ulysses sequence parallelism.

Keep the upstream packed layout, mixed precision and six PDD heads intact.
Only shard the block stack; gather before the upstream image-row selection.
"""
import torch.distributed as dist

from x_base import (
    ParallelManager, all_to_all_after_attn, all_to_all_before_attn,
    attention_manager, gather_sequence, set_pad, split_sequence,
)

PAD_KEY = "h3_image_seq"


class H3ImageAttnProcessor:
    def __init__(self, manager=None):
        self.manager = manager

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3_image import _apply_rotary_emb

        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = (layer(hidden_states) for layer in (attn.to_q, attn.to_k, attn.to_v))
        q, k, v = (x.unflatten(-1, (attn.heads, attn.head_dim)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = (_apply_rotary_emb(x, *rotary_emb) for x in (q, k))
        if self.manager is not None:
            if attention_mask is not None:
                raise ValueError("H3 Image SP expects the upstream single packed document without a mask")
            q, k, v = (all_to_all_before_attn(x, self.manager.sp_group, pad_key=PAD_KEY)
                       for x in (q, k, v))
        out = attention_manager.attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2)
        if self.manager is not None:
            out = all_to_all_after_attn(out, self.manager.sp_group, pad_key=PAD_KEY)
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))


def enable_h3_image_sp(transformer, sp_size=1):
    """Attach xGen attention to one model, with optional SP over all ranks."""
    if getattr(transformer, "_xgen_image_sp", None) is not None:
        raise ValueError("H3 Image parallelism is already configured on this transformer")
    world = dist.get_world_size() if dist.is_initialized() else 1
    if sp_size != world or world not in (1, 4):
        raise ValueError("H3 Image currently supports one rank or four ranks with SP4")
    if sp_size > 1 and transformer.config.num_attention_heads % sp_size:
        raise ValueError("H3 Image attention heads must be divisible by SP size")
    manager = (ParallelManager(1, 1, sp_size, register_tp_global=False)
               if sp_size > 1 else None)
    transformer._xgen_image_sp = sp_size
    transformer._xgen_image_parallel_manager = manager
    # The text token refiner still sees the whole text sequence, so has no SP manager.
    for block in transformer.token_refiner.refiner_blocks:
        block.attn.set_processor(H3ImageAttnProcessor())
    for block in transformer.transformer_blocks:
        block.attn.set_processor(H3ImageAttnProcessor(manager))
    if manager is None:
        return transformer

    state = {}
    handles = []

    def block_hook(first):
        def hook(module, args, kwargs):
            # Upstream calls the blocks positionally; accept keyword calls as well.
            names = ("hidden_states", "temb", "adaln_indices", "rotary_emb", "attention_mask")
            values = dict(zip(names, args))
            values.update(kwargs)
            indices = values["adaln_indices"]
            if first:
                state["pad"] = set_pad(PAD_KEY, indices.shape[0], manager.sp_group)
                state["sequence_length"] = indices.shape[0]
                values["hidden_states"] = split_sequence(
                    values["hidden_states"], manager.sp_group, dim=1, pad=state["pad"])
            values["adaln_indices"] = split_sequence(indices, manager.sp_group, dim=0, pad=state["pad"])
            values["rotary_emb"] = tuple(split_sequence(x, manager.sp_group, dim=0, pad=state["pad"])
                                         for x in values["rotary_emb"])
            return (), values
        return hook

    for i, block in enumerate(transformer.transformer_blocks):
        handles.append(block.register_forward_pre_hook(block_hook(i == 0), with_kwargs=True))

    def norm_hook(module, args, kwargs):
        values = dict(zip(("hidden_states", "temb", "timestep_indices"), args))
        values.update(kwargs)
        values["timestep_indices"] = split_sequence(
            values["timestep_indices"], manager.sp_group, dim=0, pad=state["pad"])
        return (), values

    def output_hook(module, args, output):
        return gather_sequence(output, manager.sp_group, dim=1, pad=state["pad"])

    handles.append(transformer.norm_out.register_forward_pre_hook(norm_hook, with_kwargs=True))
    for head in transformer.pdd_proj_out:
        handles.append(head.register_forward_hook(output_hook))
    transformer._xgen_image_sp_hooks = handles
    transformer._xgen_image_sp_state = state
    return transformer
