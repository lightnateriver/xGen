"""TP 全局状态管理。

由 ParallelManager 在初始化时调用 set_tp_group() 写入，
TP 线性层通过 get_tp_group() / get_tp_world_size() / get_tp_rank() 读取。

设计原因：TP 通信组由 ProcessGroupMesh 统一创建（属于 sequence_parallelism 体系），
不依赖 cfg_parallelism 的全局单例 init_TP()。
"""
import logging
import torch.distributed as dist
from typing import Optional
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


_tp_group: Optional[ProcessGroup] = None
_tp_world_size: int = 1
_tp_rank: int = 0


def set_tp_group(group: ProcessGroup, world_size: int, rank: int):
    """由 ParallelManager._init_groups() 调用，设置 TP 通信组。"""
    global _tp_group, _tp_world_size, _tp_rank
    _tp_group = group
    _tp_world_size = world_size
    _tp_rank = rank
    logger.info(f"[TP-DIAG] set_tp_group: world_size={world_size}, rank={rank}, "
                f"group={group}")


def get_tp_group() -> ProcessGroup:
    if _tp_group is None:
        raise ValueError(
            "TP group not initialized. "
            "Call ParallelManager with tp_size > 1 first."
        )
    return _tp_group


def get_tp_world_size() -> int:
    return _tp_world_size


def get_tp_rank() -> int:
    return _tp_rank
