"""
cache_dit (DBCache) 集成模块

提供基于 cache_dit 外部库的 Dual Block Cache 加速，支持通用 pipeline 和 MiniMax-H3。

## 功能

- 从 YAML 加载基础配置，支持环境变量 override（优先级：env > YAML > default）
- 通用 pipeline：直接调用 cache_dit.enable_cache(pipe, ...)
- MiniMax-H3：构造 BlockAdapter 绕过 ModularPipeline 类型检查，并修复 inspect.signature

## 使用方式

```python
from x_base.cache import cache_dit_on_pipe

# 在 general_ability 中调用
if args.cache_dit:
    cache_dit_on_pipe(pipe, args)
```

## 环境变量

| 环境变量 | 类型 | 对应字段 |
|---------|------|---------|
| CACHE_DIT_FN_COMPUTE_BLOCKS | int | Fn_compute_blocks |
| CACHE_DIT_BN_COMPUTE_BLOCKS | int | Bn_compute_blocks |
| CACHE_DIT_MAX_WARMUP_STEPS | int | max_warmup_steps |
| CACHE_DIT_WARMUP_INTERVAL | int | warmup_interval |
| CACHE_DIT_MAX_CACHED_STEPS | int | max_cached_steps |
| CACHE_DIT_MAX_CONTINUOUS_CACHED_STEPS | int | max_continuous_cached_steps |
| CACHE_DIT_RESIDUAL_DIFF_THRESHOLD | float | residual_diff_threshold |
| CACHE_DIT_NUM_INFERENCE_STEPS | int | num_inference_steps |
"""
import os
import functools
import logging

try:
    import cache_dit
    from cache_dit import DBCacheConfig, BlockAdapter, ForwardPattern
    _CACHE_DIT_AVAILABLE = True
except ImportError:
    _CACHE_DIT_AVAILABLE = False

from x_base.config import CACHE_DIT_CONFIG

logger = logging.getLogger(__name__)

# 环境变量 → DBCacheConfig 字段映射
_ENV_MAP = {
    "CACHE_DIT_FN_COMPUTE_BLOCKS": ("Fn_compute_blocks", int),
    "CACHE_DIT_BN_COMPUTE_BLOCKS": ("Bn_compute_blocks", int),
    "CACHE_DIT_MAX_WARMUP_STEPS": ("max_warmup_steps", int),
    "CACHE_DIT_WARMUP_INTERVAL": ("warmup_interval", int),
    "CACHE_DIT_MAX_CACHED_STEPS": ("max_cached_steps", int),
    "CACHE_DIT_MAX_CONTINUOUS_CACHED_STEPS": ("max_continuous_cached_steps", int),
    "CACHE_DIT_RESIDUAL_DIFF_THRESHOLD": ("residual_diff_threshold", float),
    "CACHE_DIT_NUM_INFERENCE_STEPS": ("num_inference_steps", int),
}


def _get_cache_dit_config():
    """Load DBCacheConfig from YAML, then apply environment variable overrides.

    Priority: env var > YAML > dataclass default.
    """
    cache_config = DBCacheConfig(**CACHE_DIT_CONFIG)
    for env_key, (field_name, cast) in _ENV_MAP.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            setattr(cache_config, field_name, cast(env_val))
            logger.info(f"cache_dit env override: {field_name}={cast(env_val)} (from {env_key})")
    return cache_config


def _enable_cache_dit_minimax(pipe):
    """MiniMax-H3 专属 cache_dit 启用逻辑。

    两个问题需要特殊处理：
    1. ModularPipeline 不继承 DiffusionPipeline，enable_cache(pipe) 会 AssertionError。
       → 构造 BlockAdapter 直传，绕过 pipeline 类型检查。
    2. cache_dit 用 functools.partial 替换 transformer.forward，破坏 inspect.signature()。
       MiniMax-H3 denoise step 用 inspect.signature 动态过滤 kwargs，导致 5 个参数丢失。
       → 用 @functools.wraps 包一层恢复原始签名。
    """
    adapter = BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        check_forward_pattern=True,
    )
    cache_dit.enable_cache(adapter, cache_config=_get_cache_dit_config())

    # 修复 inspect.signature：用 @functools.wraps 包一层恢复原始 forward 的参数签名
    _orig_fwd = pipe.transformer._original_forward
    _cached_fwd = pipe.transformer.forward

    @functools.wraps(_orig_fwd)
    def _forward_with_sig(*args, **kwargs):
        return _cached_fwd(*args, **kwargs)

    pipe.transformer.forward = _forward_with_sig


def cache_dit_on_pipe(pipe, args):
    """将 cache_dit (DBCache) 加速应用到 pipeline。

    根据 args.model 选择适配路径：
    - MiniMax-H3：走 BlockAdapter 路径（ModularPipeline 不兼容标准路径）
    - 其他模型：走标准 enable_cache(pipe) 路径

    Args:
        pipe: pipeline 实例
        args: 参数对象，需包含 model、cache_dit 属性
    """
    if not _CACHE_DIT_AVAILABLE:
        raise ImportError(
            "cache_dit is not installed. Install it with: pip install cache_dit"
        )
    if not hasattr(pipe, "_context_manager"):
        if args.model == "MiniMax-H3":
            _enable_cache_dit_minimax(pipe)
        else:
            cache_dit.enable_cache(pipe, cache_config=_get_cache_dit_config())
        logger.info("use cache dit")
    else:
        logger.info("cache dit already enabled, skip re-enabling")
