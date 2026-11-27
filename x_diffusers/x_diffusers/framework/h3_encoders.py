"""Shared Qwen text and vision encoder adaptation for H3 video and Image."""


# ------------------------------------------------------------------------
# transformer.h3_text_encoder
# ------------------------------------------------------------------------

# Qwen3-VL text encoder adaptation shared by H3 image and video.

import logging

logger = logging.getLogger("infer")


def truncate_text_encoder(pipe, num_keep=50):
    """截断 Qwen3-VL text_encoder 到前 num_keep 层。

    在 load_components() 之后、group offloading 之前调用。
    层 50-63 的权重从不进入 NPU。

    H3 只取 hidden_states[50] (layer 49 输出, pre-norm)，
    后 14 层的权重和计算完全浪费。

    三步:
      1. 删 layers[num_keep:] — ModuleList 只剩 num_keep 层, forward 只跑这些层
      2. norm → Identity — last_hidden_state = layer (num_keep-1) pre-norm 输出
      3. 拦截 output_hidden_states=False — 避免 materialize hidden state tuple
    """
    import torch.nn as nn
    import types

    te = pipe.text_encoder
    # Qwen3-VL structure: text_encoder.model.language_model.layers
    #   Qwen3VLForConditionalGeneration
    #     └── .model = Qwen3VLModel
    #           └── .language_model = Qwen3VLTextModel
    #                 ├── .layers = nn.ModuleList([...])
    #                 ├── .norm = Qwen3VLTextRMSNorm
    #                 └── .rotary_emb
    layers = te.model.language_model.layers
    total_layers = len(layers)

    # 1. 删除后 (total - num_keep) 层 — CPU 内存直接释放，不经过 NPU
    for i in range(total_layers - 1, num_keep - 1, -1):
        del layers[i]

    # 2. final RMSNorm → Identity
    #    H3 要 pre-norm 输出; norm=Identity 后 last_hidden_state = layer (num_keep-1) 输出
    te.model.language_model.norm = nn.Identity()

    # 3. 拦截 output_hidden_states=False，避免 materialize hidden state tuple
    _orig_forward = te.model.forward

    def _patched_forward(self, *args, **kwargs):
        kwargs['output_hidden_states'] = False
        output = _orig_forward(*args, **kwargs)

        class _TruncatedOutput:
            def __init__(self, orig):
                self.last_hidden_state = orig.last_hidden_state
                # hidden_states[num_keep] = last_hidden_state (pre-norm, 因为 norm=Identity)
                self.hidden_states = (None,) * num_keep + (orig.last_hidden_state,)

        return _TruncatedOutput(output)

    te.model.forward = types.MethodType(_patched_forward, te.model)

    # 4. 绕过 get_qwen3vl_prompt_embeds 的 num_layers 验证
    #    forward 遍历 self.layers ModuleList (num_keep 个), 不读 config
    te.config.text_config.num_hidden_layers = num_keep + 1

    logger.info(f"Truncated text_encoder to {num_keep} layers (removed {total_layers - num_keep})")


def enable_text_tp(pipe, tp_size, *, create_manager=False):
    """Shard the 50 Qwen layers used by H3 after component loading."""
    from x_base import ParallelManager, enable_tp
    if create_manager:
        pipe._xgen_image_tp_manager = ParallelManager(1, 1, 1, tp_size=tp_size)
    enable_tp(pipe.text_encoder, tp_size, target_layers=range(50))


# ------------------------------------------------------------------------
# transformer.h3_vision_sp
# ------------------------------------------------------------------------

# Qwen3-VL visual sequence parallelism shared by H3 image and video.
from contextlib import contextmanager
from functools import wraps
from types import MethodType

import torch
import torch.distributed as dist

PAD_KEY = 'h3_reference_vision'


@contextmanager
def full_bf16_accumulation(device=None):
    if device is not None and torch.device(device).type != 'cuda':
        yield
        return
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

        if torch.is_grad_enabled() or pixels.device.type not in ('cuda', 'npu'):
            raise ValueError('H3 vision SP is CUDA/NPU inference-only')
        self.group = self.group_getter()
        peers = dist.get_world_size(self.group)
        if peers not in (2, 4, 8) or model.config.num_heads % peers:
            raise ValueError('H3 vision SP requires 2/4/8 ranks and divisible heads')
        self.metrics['sp'] = peers
        self.metrics['calls'] += 1
        with full_bf16_accumulation(pixels.device):
            modern = hasattr(m, 'get_vision_interpolation_indices_and_weights')
            if modern:
                indices, weights = m.get_vision_interpolation_indices_and_weights(
                    grid, num_grid_per_side=model.num_grid_per_side, mode=model.interpolation_mode,
                    align_corners=model.interpolation_align_corners, spatial_merge_size=model.config.spatial_merge_size,
                    kwargs={})
            else:
                # Transformers 5.14 used corner-major bilinear arrays; preserve
                # its original sum axis rather than changing interpolation.
                indices, weights = m.get_vision_bilinear_indices_and_weights(
                    grid, num_grid_per_side=model.num_grid_per_side,
                    spatial_merge_size=model.config.spatial_merge_size, kwargs={})
            positions = m.get_vision_position_ids(grid, model.spatial_merge_size, kwargs={})
            if modern:
                cu, maxlen = m.get_vision_attention_seqlens(grid, model.config, kwargs={})
            else:
                cu, maxlen = m.get_vision_cu_seqlens(grid, kwargs={}), None
            hidden = model.patch_embed(pixels)
            hidden = hidden + (model.pos_embed(indices) * weights[:, :, None]).sum(1 if modern else 0).to(hidden.dtype)
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
