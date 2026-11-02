"""TP 权重分片入口。

enable_tp() 在模型上启用 Tensor Parallelism：
1. 校验 TP group 已由 ParallelManager 初始化
2. 过滤 ViT / Adapter，只处理 LLM Decoder
3. 替换 nn.Linear 为 TP 版本并从原始权重提取分片
"""
import torch
import torch.nn as nn
from typing import Optional, Iterable

from .state import get_tp_world_size
from .layers import ColumnParallelLinear, RowParallelLinear


def enable_tp(model, tp_size: int, target_layers: Optional[Iterable[int]] = None):
    """在模型上启用 Tensor Parallelism

    调用时机: ParallelManager 初始化之后（TP group 已创建），model.from_pretrained() 之后
    前置条件: tensor_parallelism.state.set_tp_group() 已由 ParallelManager 调用

    流程:
      1. 校验 TP group 已初始化
      2. 遍历 LLM Decoder 层，替换 nn.Linear 为 TP 版本
      3. 从原始权重中提取分片
      4. 跳过 ViT、Vision-Language Adapter 等非 LLM 模块

    Args:
        model: 已加载完整权重的模型
        tp_size: TP 并行度（应与 ParallelManager 中一致）
        target_layers: 只对指定层索引做 TP（如 range(50) 表示前 50 层）。
                       None 表示全部 LLM Decoder 层。
    """
    if tp_size <= 1:
        return model

    # 校验 TP group 已由 ParallelManager 初始化
    if get_tp_world_size() != tp_size:
        raise ValueError(
            f"TP group not initialized or size mismatch: "
            f"expected {tp_size}, got {get_tp_world_size()}. "
            f"Ensure ParallelManager is initialized with tp_size={tp_size} before calling enable_tp()."
        )

    # GQA 约束校验
    if hasattr(model, 'config') and hasattr(model.config, 'num_key_value_heads'):
        num_kv_heads = model.config.num_key_value_heads
        if num_kv_heads < tp_size:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be >= tp_size ({tp_size}) for GQA. "
                f"Consider using a smaller tp_size."
            )

    # 模型层替换 + 权重分片
    diag = {'layers_found': 0, 'proj_replaced': 0}

    _replace_and_shard(model, target_layers, diag)

    return model


# Qwen3-VL Attention/MLP 中需要替换的投影层及其 TP 类型
_TP_PROJ_MAP = {
    # Attention projections — gather_output=False: 每 rank 只持有自己的
    # head 分片，不做 all_gather。配合下方 num_heads patching，每 rank
    # 只计算 num_heads//tp_size 个 head 的 attention，scores 矩阵减半。
    # o_proj input_is_parallel=True: attention 输出已是分片的，直接消费。
    'q_proj':    ('column', False),  # gather_output=False
    'k_proj':    ('column', False),
    'v_proj':    ('column', False),
    'o_proj':    ('row',    True),   # input_is_parallel=True
    # MLP projections — standard TP (SiLU/SwiGLU is element-wise,
    # works correctly on sharded gate/up outputs)
    'gate_proj': ('column', False),
    'up_proj':   ('column', False),
    'down_proj': ('row',    True),
}

# 跳过这些模块名前缀（ViT、Vision-Language Adapter 等）
_SKIP_MODULE_PREFIXES = (
    'visual',          # Qwen3VLVisionTransformer / ViT
    'vision',          # 通用 ViT 命名
    'adapter',         # Vision-Language Adapter
    'merger',          # Qwen3VL 的 patch embedding merger
)


def _should_skip(module_name: str) -> bool:
    """判断模块是否应跳过 TP 替换（ViT、Adapter 等）"""
    name_lower = module_name.lower()
    return any(name_lower.startswith(prefix) for prefix in _SKIP_MODULE_PREFIXES)


def _replace_and_shard(model, target_layers: Optional[Iterable[int]] = None, diag: dict = None):
    """递归替换模型中 LLM Decoder 的 Linear 层

    策略:
      - 只处理 LLM Decoder 层（通过模块类型或路径名识别）
      - 跳过 ViT、Vision-Language Adapter
      - 如果指定 target_layers，只替换对应索引的 decoder layer
      - 通过类型检查确保替换目标正确
    """
    target_set = set(target_layers) if target_layers is not None else None

    for name, module in model.named_modules():
        # 跳过 ViT / Adapter
        if _should_skip(name):
            continue

        # 识别 LLM Decoder Layer — 通过模块类名或路径名
        is_decoder_layer = _is_decoder_layer(module, name, target_set)

        if not is_decoder_layer:
            continue

        if diag is not None:
            diag['layers_found'] += 1

        # 在 decoder layer 内部替换 attention 和 MLP 投影
        for child_name, child in module.named_children():
            _try_replace_projections(child, diag)


def _is_decoder_layer(module, name: str, target_set: Optional[set]) -> bool:
    """判断模块是否是目标 LLM Decoder Layer"""
    # 方法 1: 类名匹配
    # 支持: DecoderLayer, DecoderBlock (标准 transformers 命名)
    #       TransformerBlock (ComfyUI 对 Qwen3-VL decoder layer 的命名)
    class_name = type(module).__name__
    if ('DecoderLayer' not in class_name
            and 'DecoderBlock' not in class_name
            and class_name != 'TransformerBlock'):
        return False

    # 方法 2: 层索引过滤（如果指定了 target_layers）
    if target_set is not None:
        # 从模块路径名提取层索引，如 "model.layers.5" → 5
        parts = name.split('.')
        layer_idx = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
                break
        if layer_idx is None or layer_idx not in target_set:
            return False

    return True


def _try_replace_projections(module, diag: dict = None):
    """替换模块中的 q/k/v/o/gate/up/down 投影为 TP 版本

    通过检查模块是否同时拥有多个 _TP_PROJ_MAP 中的属性来确认是 Attention 或 MLP。
    """
    # 检查模块是否包含至少一个可替换的投影层
    has_proj = any(hasattr(module, proj_name) for proj_name in _TP_PROJ_MAP)
    if not has_proj:
        return

    for proj_name, (tp_type, extra_kw) in _TP_PROJ_MAP.items():
        if not hasattr(module, proj_name):
            continue
        orig = getattr(module, proj_name)
        if not isinstance(orig, nn.Linear) or hasattr(orig, '_tp_replaced'):
            continue

        if tp_type == 'column':
            tp_layer = ColumnParallelLinear(
                in_features=orig.in_features,
                out_features=orig.out_features,
                bias=orig.bias is not None,
                gather_output=extra_kw,
                dtype=orig.weight.dtype,
            )
        else:
            tp_layer = RowParallelLinear(
                in_features=orig.in_features,
                out_features=orig.out_features,
                bias=orig.bias is not None,
                input_is_parallel=extra_kw,
                dtype=orig.weight.dtype,
            )

        tp_layer.load_shard(orig.weight.data, orig.bias.data if orig.bias is not None else None)
        tp_layer._tp_replaced = True
        setattr(module, proj_name, tp_layer)

        if diag is not None:
            diag['proj_replaced'] += 1

    # --- num_heads patching ---
    # ColumnParallelLinear 按 head 切分了 q/k/v 权重，但 ComfyUI 的
    # attention 代码用 module.num_heads 做 reshape。如果不 patch，
    # reshape 会用原始的 64 heads 去匹配分片后的 4096 维度，导致报错。
    # patch 后 num_heads=32，reshape 用 32*128=4096，匹配分片输出。
    if not hasattr(module, '_tp_heads_patched'):
        tp_size = get_tp_world_size()
        if hasattr(module, 'num_heads'):
            orig_heads = module.num_heads
            module.num_heads = module.num_heads // tp_size
        # 检查所有常见的 KV head 属性名：
        # - num_key_value_heads: 标准 transformers 命名
        # - num_kv_heads: ComfyUI llama.py 命名
        for _kv_attr in ('num_key_value_heads', 'num_kv_heads'):
            if hasattr(module, _kv_attr):
                orig_kv = getattr(module, _kv_attr)
                setattr(module, _kv_attr, orig_kv // tp_size)
        module._tp_heads_patched = True
