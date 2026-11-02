"""
Unit tests for sequence_parallelism/mesh.py.

Covers:
- _prod helper
- ProcessGroupMesh: init validation, _unravel/_ravel, shape/rank properties,
  _get_coords_along_axis, get_group_along_axis (cached + creation)
- ParallelConfig: validate (total mismatch, ulysses*ring mismatch, valid config)
- ParallelManager: _init_groups (dp/cp/sp), _init_ulysses_sp (yunchang import error)
- initialize: calls dist.init_process_group when not initialized
"""
from unittest.mock import patch, MagicMock, call

import pytest

from x_base.sequence_parallelism.mesh import (
    _prod,
    ProcessGroupMesh,
    ParallelConfig,
    ParallelManager,
    initialize,
)
from x_base.sequence_parallelism.errors import ParallelConfigError, BackendNotAvailableError


class TestProd:
    def test_empty(self):
        assert _prod([]) == 1

    def test_single(self):
        assert _prod([5]) == 5

    def test_multiple(self):
        assert _prod([2, 3, 4]) == 24


class TestProcessGroupMeshInit:
    def test_dist_not_initialized_raises(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            with pytest.raises(RuntimeError, match="not initialized"):
                ProcessGroupMesh(2, 2)

    def test_product_mismatch_raises(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            mock_dist.get_rank.return_value = 0
            with pytest.raises(RuntimeError, match="Product of mesh sizes"):
                ProcessGroupMesh(2, 2)

    def test_valid_init(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            mock_dist.get_rank.return_value = 3
            mesh = ProcessGroupMesh(2, 2, 2)
            assert mesh.shape == (2, 2, 2)
            assert mesh.rank == 3


class TestProcessGroupMeshUnravelRavel:
    def test_unravel_round_trip(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            mock_dist.get_rank.return_value = 0
            mesh = ProcessGroupMesh(2, 2, 2)
        for rank in range(8):
            coord = mesh._unravel(rank, (2, 2, 2))
            assert mesh._ravel(coord, (2, 2, 2)) == rank

    def test_unravel_known_value(self):
        coord = ProcessGroupMesh._unravel(5, (2, 2, 2))
        assert coord == (1, 0, 1)

    def test_ravel_known_value(self):
        rank = ProcessGroupMesh._ravel((1, 0, 1), (2, 2, 2))
        assert rank == 5


class TestProcessGroupMeshGetCoordsAlongAxis:
    def setup_method(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            mock_dist.get_rank.return_value = 0
            self.mesh = ProcessGroupMesh(2, 2, 2)

    def test_single_axis(self):
        coords = self.mesh._get_coords_along_axis((1, 0, 1), [2], [[0, 1]])
        assert len(coords) == 2
        assert (1, 0, 0) in coords
        assert (1, 0, 1) in coords

    def test_axis_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            self.mesh._get_coords_along_axis((0, 0, 0), [3], [[0]])

    def test_two_axes(self):
        coords = self.mesh._get_coords_along_axis((0, 0, 0), [0, 2], [[0, 1], [0, 1]])
        assert len(coords) == 4


class TestProcessGroupMeshGetGroupAlongAxis:
    def setup_method(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 0
            mock_dist.new_group.return_value = MagicMock()
            self.mesh = ProcessGroupMesh(2, 2)

    def test_returns_group(self):
        mock_group = MagicMock()
        self.mesh._ranks_to_group[(0, 1)] = mock_group
        result = self.mesh.get_group_along_axis(0)
        assert result == mock_group

    def test_creates_group_if_not_cached(self):
        with patch.object(self.mesh, "_create_group_along_axis", return_value=MagicMock()) as mock_create:
            self.mesh.get_group_along_axis(0)
            mock_create.assert_called_once()


class TestProcessGroupMeshDestroy:
    def test_destroy_clears_caches(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 0
            mock_dist.new_group.return_value = MagicMock()
            mock_dist.destroy_process_group = MagicMock()
            mesh = ProcessGroupMesh(2, 2)
            mesh._ranks_to_group[(0, 1)] = MagicMock()
            mesh.destroy_mesh_process_groups()
            assert len(mesh._ranks_to_group) == 0
            assert len(mesh._group_to_ranks) == 0


class TestParallelConfigValidate:
    def test_valid_config(self):
        config = ParallelConfig(dp_size=2, cp_size=1, sp_size=4)
        config.validate(8)

    def test_total_mismatch_raises(self):
        config = ParallelConfig(dp_size=2, cp_size=2, sp_size=4)
        with pytest.raises(ParallelConfigError, match="dp.*cp.*sp"):
            config.validate(8)

    def test_ulysses_ring_mismatch_raises(self):
        config = ParallelConfig(dp_size=1, cp_size=1, sp_size=4, sp_ulysses_size=2, sp_ring_size=3)
        with pytest.raises(ParallelConfigError, match="ulysses"):
            config.validate(4)

    def test_ulysses_ring_valid(self):
        config = ParallelConfig(dp_size=1, cp_size=1, sp_size=4, sp_ulysses_size=2, sp_ring_size=2)
        config.validate(4)


class TestParallelManager:
    def test_init_groups_dp_only(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 0
            mock_dist.new_group.return_value = MagicMock()
            mock_group = MagicMock()
            mock_dist.get_rank.side_effect = [0, 0]
            with patch.object(ProcessGroupMesh, "get_group_along_axis", return_value=mock_group):
                mgr = ParallelManager(dp_size=4, cp_size=1, sp_size=1)
            assert mgr.dp_size == 4
            assert mgr.cp_group is None
            assert mgr.sp_group is None

    def test_ulysses_sp_yunchang_not_available(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 0
            mock_dist.new_group.return_value = MagicMock()
            mock_group = MagicMock()
            with patch.object(ProcessGroupMesh, "get_group_along_axis", return_value=mock_group):
                with patch.dict("sys.modules", {"yunchang": None}):
                    with pytest.raises(BackendNotAvailableError, match="yunchang"):
                        ParallelManager(dp_size=1, cp_size=1, sp_size=4, sp_ulysses_size=2, sp_ring_size=2)


class TestInitialize:
    def test_calls_init_process_group_when_not_initialized(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            initialize(rank=0, world_size=4, backend="nccl")
            mock_dist.init_process_group.assert_called_once()

    def test_skips_when_already_initialized(self):
        with patch("x_base.sequence_parallelism.mesh.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            initialize(rank=0, world_size=4)
            mock_dist.init_process_group.assert_not_called()
