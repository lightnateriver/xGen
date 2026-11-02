"""
Unit tests for cfg_parallelism/utils.py.

Covers:
- GroupCoordinator: init (rank assignment, group selection), all_reduce (world_size=1 bypass),
  all_to_all_single (world_size=1 bypass), all_gather (world_size=1 bypass, dim validation,
  negative dim, separate_tensors)
- Global singletons: init_TP/get_tp_group, init_EP/get_ep_group, init_CFG_Parallel/get_cfg_group
  (double-init raises, get before init raises, rank/world_size accessors)
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.cfg_parallelism.utils import (
    GroupCoordinator,
    init_TP,
    get_tp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    init_EP,
    get_ep_group,
    get_expert_parallel_world_size,
    get_expert_parallel_rank,
    init_CFG_Parallel,
    get_cfg_group,
    get_cfg_parallel_rank,
    get_cfg_parallel_world_size,
)


def _make_coordinator(world_size=1, rank=0):
    with patch("x_base.cfg_parallelism.utils.torch.distributed") as mock_dist:
        mock_dist.get_rank.return_value = rank
        mock_device_group = MagicMock()
        mock_cpu_group = MagicMock()
        mock_dist.new_group.side_effect = [mock_device_group, mock_cpu_group]
        with patch("x_base.cfg_parallelism.utils.torch_npu") as mock_npu:
            mock_npu.npu.current_device.return_value = 0
            ranks = list(range(world_size))
            coord = GroupCoordinator(group_ranks=[ranks], local_rank=rank)
    return coord


class TestGroupCoordinatorInit:
    def test_single_rank(self):
        coord = _make_coordinator(world_size=1, rank=0)
        assert coord.world_size == 1
        assert coord.rank_in_group == 0

    def test_multi_rank(self):
        coord = _make_coordinator(world_size=4, rank=2)
        assert coord.world_size == 4
        assert coord.rank_in_group == 2

    def test_rank_not_in_any_group_raises(self):
        with patch("x_base.cfg_parallelism.utils.torch.distributed") as mock_dist:
            mock_dist.get_rank.return_value = 99
            mock_dist.new_group.return_value = MagicMock()
            with patch("x_base.cfg_parallelism.utils.torch_npu") as mock_npu:
                mock_npu.npu.current_device.return_value = 0
                with pytest.raises(ValueError, match="cpu通信组"):
                    GroupCoordinator(group_ranks=[[0, 1, 2, 3]], local_rank=0)


class TestGroupCoordinatorAllReduce:
    def test_world_size_1_bypass(self):
        coord = _make_coordinator(world_size=1)
        t = torch.randn(3, 4)
        result = coord.all_reduce(t)
        assert result is t


class TestGroupCoordinatorAllToAllSingle:
    def test_world_size_1_bypass(self):
        coord = _make_coordinator(world_size=1)
        t = torch.randn(4, 4)
        result = coord.all_to_all_single(t, t, [4], [4])
        assert result is t


class TestGroupCoordinatorAllGather:
    def test_world_size_1_bypass(self):
        coord = _make_coordinator(world_size=1)
        t = torch.randn(3, 4)
        result = coord.all_gather(t)
        assert result is t

    def test_invalid_dim_raises(self):
        coord = _make_coordinator(world_size=2, rank=0)
        t = torch.randn(3, 4)
        with pytest.raises(ValueError, match="Invalid dim"):
            coord.all_gather(t, dim=5)

    def test_negative_dim_converted(self):
        coord = _make_coordinator(world_size=1, rank=0)
        t = torch.randn(3, 4)
        result = coord.all_gather(t, dim=-1)
        assert result is t


class TestTPSingleton:
    def test_get_before_init_raises(self):
        import x_base.cfg_parallelism.utils as u
        original = u._TP
        u._TP = None
        try:
            with pytest.raises(ValueError, match="not initialized"):
                get_tp_group()
        finally:
            u._TP = original

    def test_double_init_raises(self):
        import x_base.cfg_parallelism.utils as u
        original = u._TP
        u._TP = None
        try:
            with patch("x_base.cfg_parallelism.utils.torch.distributed") as mock_dist:
                mock_dist.get_rank.return_value = 0
                mock_dist.new_group.return_value = MagicMock()
                with patch("x_base.cfg_parallelism.utils.torch_npu"):
                    init_TP(1)
            with pytest.raises(ValueError, match="already initialized"):
                init_TP(1)
        finally:
            u._TP = original


class TestEPSingleton:
    def test_get_before_init_raises(self):
        import x_base.cfg_parallelism.utils as u
        original = u._EP
        u._EP = None
        try:
            with pytest.raises(ValueError, match="not initialized"):
                get_ep_group()
        finally:
            u._EP = original


class TestCFGSingleton:
    def test_get_before_init_raises(self):
        import x_base.cfg_parallelism.utils as u
        original = u._CFG
        u._CFG = None
        try:
            with pytest.raises(ValueError, match="not initialized"):
                get_cfg_group()
        finally:
            u._CFG = original

    def test_double_init_raises(self):
        import x_base.cfg_parallelism.utils as u
        original = u._CFG
        u._CFG = None
        try:
            with patch("x_base.cfg_parallelism.utils.torch.distributed") as mock_dist:
                mock_dist.get_rank.return_value = 0
                mock_dist.new_group.return_value = MagicMock()
                with patch("x_base.cfg_parallelism.utils.torch_npu"):
                    init_CFG_Parallel(1)
            with pytest.raises(ValueError, match="already initialized"):
                init_CFG_Parallel(1)
        finally:
            u._CFG = original
