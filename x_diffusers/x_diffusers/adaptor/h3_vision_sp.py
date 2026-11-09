"""Opt-in Qwen3-VL visual block SP using the native xGen collectives."""
from contextlib import contextmanager
from functools import wraps
from types import MethodType

import torch
import torch.distributed as dist

PAD_KEY = 'h3_reference_vision'


@contextmanager
def full_bf16_accumulation():
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


class VisionSP:
    def __init__(self, group_getter):
        self.group_getter = group_getter
        self.group = None
        self.metrics = dict(calls=0, completed=0, blocks=0, tokens=0, sp=8, communication='bf16')

    def split(self, tensor, pad):
        from x_base import split_sequence
        return split_sequence(tensor.unsqueeze(0), self.group, dim=1, pad=pad).squeeze(0)

    def gather(self, tensor, pad):
        from x_base import gather_sequence
        return gather_sequence(tensor.unsqueeze(0), self.group, dim=1, pad=pad).squeeze(0)

    def attention(self, attn, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as m
        from x_base import all_to_all_before_attn, all_to_all_after_attn

        n = hidden_states.shape[0]
        query, key, value = attn.qkv(hidden_states).reshape(n, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        query, key = m.apply_rotary_pos_emb_vision(query, key, *position_embeddings)
        query, key, value = [all_to_all_before_attn(t.unsqueeze(0), self.group, pad_key=PAD_KEY)
                             .transpose(1, 2).contiguous() for t in (query, key, value)]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        interface = m.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, m.eager_attention_forward)
        chunks = [torch.split(t, lengths, dim=2) for t in (query, key, value)]
        output = torch.cat([interface(attn, q, k, v, attention_mask=None, scaling=attn.scaling,
                                     dropout=0., is_causal=False)[0] for q, k, v in zip(*chunks)], dim=1)
        output = all_to_all_after_attn(output.contiguous(), self.group, pad_key=PAD_KEY)
        return attn.proj(output.reshape(n, attn.dim))

    def forward(self, model, pixels, grid):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as m
        from x_base import set_pad

        if torch.is_grad_enabled() or not pixels.is_cuda:
            raise ValueError('H3 vision SP is CUDA inference-only')
        self.group = self.group_getter()
        if dist.get_world_size(self.group) != 8 or model.config.num_heads % 8:
            raise ValueError('H3 vision SP requires eight ranks and divisible heads')
        self.metrics['calls'] += 1
        with full_bf16_accumulation():
            indices, weights = m.get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=model.num_grid_per_side, mode=model.interpolation_mode,
                align_corners=model.interpolation_align_corners, spatial_merge_size=model.config.spatial_merge_size,
                kwargs={})
            positions = m.get_vision_position_ids(grid, model.spatial_merge_size, kwargs={})
            cu, maxlen = m.get_vision_attention_seqlens(grid, model.config, kwargs={})
            hidden = model.patch_embed(pixels)
            hidden = hidden + (model.pos_embed(indices) * weights[:, :, None]).sum(1).to(hidden.dtype)
            rotary = model.rotary_pos_emb(positions).reshape(hidden.shape[0], -1)
            embedding = torch.cat((rotary, rotary), dim=-1)
            self.metrics['tokens'] += hidden.shape[0]
            pad = set_pad(PAD_KEY, hidden.shape[0], self.group)
            hidden = self.split(hidden, pad)
            position = (self.split(embedding.cos(), pad), self.split(embedding.sin(), pad))
            deep = []
            for i, block in enumerate(model.blocks):
                hidden = block(hidden, cu_seqlens=cu, max_seqlen=maxlen, position_embeddings=position)
                self.metrics['blocks'] += 1
                if i in model.deepstack_visual_indexes:
                    deep.append(model.deepstack_merger_list[model.deepstack_visual_indexes.index(i)](
                        self.gather(hidden, pad)))
            full = self.gather(hidden, pad)
            output = m.BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=full, pooler_output=model.merger(full), deepstack_features=deep)
        self.metrics['completed'] += 1
        return output


def install_vision_sp(visual, group_getter):
    adapter = VisionSP(group_getter)
    for block in visual.blocks:
        def attention(attn, *args, **kwargs):
            return adapter.attention(attn, *args, **kwargs)
        block.attn.forward = MethodType(attention, block.attn)
    original = visual.forward

    @wraps(original)
    def forward(hidden_states, grid_thw, **kwargs):
        return_dict = kwargs.pop('return_dict', True)
        # The parent text model forwards these capture/cache preferences to
        # visual blocks too; H3 consumes the explicit final/DeepStack outputs.
        kwargs.pop('use_cache', None)
        kwargs.pop('output_hidden_states', None)
        if kwargs.pop('output_attentions', False):
            raise ValueError('H3 vision SP does not return attention matrices')
        if kwargs:
            raise ValueError('Unsupported Qwen3-VL visual arguments: ' + ', '.join(kwargs))
        result = adapter.forward(visual, hidden_states, grid_thw)
        return result if return_dict else result.to_tuple()

    visual.forward = forward
    visual._h3_sp_metrics = adapter.metrics
