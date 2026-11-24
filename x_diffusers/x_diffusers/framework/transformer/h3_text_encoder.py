"""Qwen3-VL text encoder adaptation shared by H3 image and video."""

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
