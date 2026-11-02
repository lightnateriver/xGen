"""
Tensor Parallelism Module

张量并行模块，提供 TP 线性层和权重分片功能。

## API 使用

### 启用 TP
```python
from x_base import enable_tp

# 前置条件: ParallelManager 已初始化 TP group
enable_tp(model, tp_size=2)
```

### TP 线性层
```python
from x_base import ColumnParallelLinear, RowParallelLinear
```
"""

from .state import set_tp_group, get_tp_group, get_tp_world_size, get_tp_rank
from .layers import ColumnParallelLinear, RowParallelLinear
from .shard import enable_tp

__all__ = [
    # State
    "set_tp_group",
    "get_tp_group",
    "get_tp_world_size",
    "get_tp_rank",
    # Layers
    "ColumnParallelLinear",
    "RowParallelLinear",
    # Entry
    "enable_tp",
]
