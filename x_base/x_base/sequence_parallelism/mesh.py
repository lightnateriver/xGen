"""
Sequence Parallelism - ProcessGroup Mesh & Parallel Manager

ProcessGroup 网格管理和并行管理器。
"""
from __future__ import annotations

import gc
import itertools
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import GroupMember

from .errors import ParallelConfigError, BackendNotAvailableError


# =============================================================================
# ProcessGroupMesh
# =============================================================================

def _prod(nums: List[int]) -> int:
    return reduce(mul, nums, 1)


class ProcessGroupMesh:
    """ProcessGroup 网格管理器
    
    用 ND 元组表示 ProcessGroup 网格，ND 坐标表示每个进程。
    例如在 3D 网格 (2, 2, 2) 中，坐标 (0, 1, 0) 表示 rank=2 的进程。
    """
    
    def __init__(self, *size: int) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized.")
        
        world_size = dist.get_world_size()
        prod_size = _prod(list(size))
        
        if prod_size != world_size:
            raise RuntimeError(
                f"Product of mesh sizes ({prod_size}) must equal world_size ({world_size})."
            )
        
        self._shape: Tuple[int, ...] = tuple(size)
        self._rank: int = dist.get_rank()
        self._coord: Tuple[int, ...] = self._unravel(self._rank, self._shape)
        self._ranks_to_group: Dict[Tuple[int, ...], Union[ProcessGroup, object]] = {}
        self._group_to_ranks: Dict[ProcessGroup, Tuple[int, ...]] = {}
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @staticmethod
    def _unravel(rank: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(np.unravel_index(rank, shape))
    
    @staticmethod
    def _ravel(coord: Tuple[int, ...], shape: Tuple[int, ...], mode: str = "raise") -> int:
        return int(np.ravel_multi_index(coord, shape, mode=mode))
    
    def _get_group(self, ranks: List[int], backend: Optional[str] = None) -> ProcessGroup:
        ranks = sorted(ranks)
        key = tuple(ranks)
        if key not in self._ranks_to_group:
            group = dist.new_group(ranks, backend=backend)
            self._ranks_to_group[key] = group
            if group is not GroupMember.NON_GROUP_MEMBER:
                self._group_to_ranks[group] = key
        return self._ranks_to_group[key]
    
    def get_group_along_axis(
        self,
        axis: Union[int, List[int]],
        indices_at_axis: Optional[Union[List[int], List[List[int]]]] = None,
        backend: Optional[str] = None,
    ) -> ProcessGroup:
        """获取当前进程沿指定轴的 ProcessGroup"""
        if isinstance(axis, int):
            axis = [axis]
            if indices_at_axis is not None and isinstance(indices_at_axis[0], int):
                indices_at_axis = [indices_at_axis]
        
        if indices_at_axis is None:
            if isinstance(axis, (list, tuple)):
                indices_at_axis = [list(range(self._shape[ax])) for ax in axis]
            else:
                indices_at_axis = list(range(self._shape[axis]))
        
        coords = self._get_coords_along_axis(self._coord, axis, indices_at_axis)
        ranks = tuple(self._ravel(coord, self._shape) for coord in coords)
        
        if ranks not in self._ranks_to_group:
            return self._create_group_along_axis(axis, indices_at_axis, backend)
        return self._ranks_to_group[ranks]
    
    def _get_coords_along_axis(
        self,
        base_coord: Tuple[int, ...],
        axis: List[int],
        indices: List[List[int]]
    ) -> List[Tuple[int, ...]]:
        for ax, idxs in zip(axis, indices):
            if ax < 0 or ax >= len(base_coord):
                raise ValueError(f"Axis {ax} out of bounds for coordinate of dimension {len(base_coord)}")
        coords = [base_coord]
        for ax, idxs in zip(axis, indices):
            new_coords = []
            for coord in coords:
                new_coords.extend([
                    coord[:ax] + (i,) + coord[ax + 1:] for i in idxs
                ])
            coords = new_coords
        return coords
    
    def _create_group_along_axis(
        self,
        axis: List[int],
        indices_at_axis: List[List[int]],
        backend: Optional[str],
    ) -> ProcessGroup:
        reduced_shape = list(self._shape)
        for ax in axis:
            reduced_shape[ax] = 1
        target_group = None
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords = self._get_coords_along_axis(base_coord, axis, indices_at_axis)
            ranks = tuple(self._ravel(c, self._shape) for c in coords)
            group = self._get_group(list(ranks), backend=backend)
            if self._rank in ranks:
                target_group = group
        return target_group
    
    def destroy_mesh_process_groups(self) -> None:
        for group in self._ranks_to_group.values():
            if group is not GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)
        self._ranks_to_group.clear()
        self._group_to_ranks.clear()
        gc.collect()


# =============================================================================
# ParallelConfig & ParallelManager
# =============================================================================

@dataclass
class ParallelConfig:
    """并行配置"""
    dp_size: int = 1
    cp_size: int = 1
    sp_size: int = 1
    tp_size: int = 1
    sp_ulysses_size: Optional[int] = None
    sp_ring_size: Optional[int] = None
    
    def validate(self, world_size: int) -> None:
        total = self.dp_size * self.cp_size * self.sp_size * self.tp_size
        if total != world_size:
            raise ParallelConfigError(
                f"dp({self.dp_size}) * cp({self.cp_size}) * sp({self.sp_size}) * tp({self.tp_size}) = {total}, "
                f"but world_size = {world_size}"
            )
        if self.sp_ulysses_size and self.sp_ring_size:
            if self.sp_ulysses_size * self.sp_ring_size != self.sp_size:
                raise ParallelConfigError(
                    f"ulysses({self.sp_ulysses_size}) * ring({self.sp_ring_size}) "
                    f"!= sp({self.sp_size})"
                )


class ParallelManager:
    """并行管理器"""
    
    def __init__(
        self,
        dp_size: int,
        cp_size: int,
        sp_size: int,
        tp_size: int = 1,
        sp_ulysses_size: Optional[int] = None,
        sp_ring_size: Optional[int] = None,
    ) -> None:
        self._config = ParallelConfig(dp_size, cp_size, sp_size, tp_size, sp_ulysses_size, sp_ring_size)
        self._config.validate(dist.get_world_size())
        self._mesh = ProcessGroupMesh(dp_size, cp_size, sp_size, tp_size)
        self._init_groups()
    
    def _init_groups(self) -> None:
        self.dp_size = self._config.dp_size
        self.dp_group = self._mesh.get_group_along_axis(0)
        self.dp_rank = dist.get_rank(self.dp_group)
        
        self.cp_size = self._config.cp_size
        if self.cp_size > 1:
            self.cp_group = self._mesh.get_group_along_axis(1)
            self.cp_rank = dist.get_rank(self.cp_group)
        else:
            self.cp_group = None
            self.cp_rank = None
        
        self.sp_size = self._config.sp_size
        self.enable_usp = False
        if self.sp_size > 1:
            self.sp_group = self._mesh.get_group_along_axis(2)
            self.sp_rank = dist.get_rank(self.sp_group)
            self._init_ulysses_sp()
        else:
            self.sp_group = None
            self.sp_rank = None

        # TP group (axis=3, 最内层)
        # 直接通过 ProcessGroupMesh 创建，不依赖 cfg_parallelism.init_TP()
        self.tp_size = self._config.tp_size
        if self.tp_size > 1:
            self.tp_group = self._mesh.get_group_along_axis(3)
            self.tp_rank = dist.get_rank(self.tp_group)
            # 同步到 TP 全局状态 (供 ColumnParallelLinear / RowParallelLinear 使用)
            from ..tensor_parallelism.state import set_tp_group
            set_tp_group(
                group=self.tp_group,
                world_size=self.tp_size,
                rank=self.tp_rank,
            )
        else:
            self.tp_group = None
            self.tp_rank = None
    
    def _init_ulysses_sp(self) -> None:
        ulysses = self._config.sp_ulysses_size
        ring = self._config.sp_ring_size
        if ulysses and ring:
            try:
                from yunchang import set_seq_parallel_pg
            except ImportError:
                raise BackendNotAvailableError("yunchang", "pip install yunchang")
            set_seq_parallel_pg(
                sp_ulysses_degree=ulysses,
                sp_ring_degree=ring,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
            self.enable_usp = True
    
    @property
    def config(self) -> ParallelConfig:
        return self._config
    
    def destroy(self) -> None:
        self._mesh.destroy_mesh_process_groups()


# =============================================================================
# 初始化函数
# =============================================================================

def initialize(
    rank: int = 0,
    world_size: int = 1,
    init_method: Optional[str] = None,
    backend: str = "nccl",
) -> None:
    """初始化分布式环境"""
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
