"""
Unit tests for sequence_parallelism/backend.py.

Covers:
- BaseCommBackend ABC instantiation constraints
- TorchDistBackend: name, all_gather (world_size=1), reduce_scatter (world_size=1),
  all_to_all_4d (world_size=1 shortcut + invalid dims), model_sharding
- get_default_backend singleton
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.sequence_parallelism.backend import (
    BaseCommBackend,
    TorchDistBackend,
    get_default_backend,
)


class TestBaseCommBackend:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseCommBackend()

    def test_subclass_must_implement_abstracts(self):
        class Incomplete(BaseCommBackend):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        class Complete(BaseCommBackend):
            @property
            def name(self):
                return "test"
            def all_gather(self, tensor, group, async_op=False):
                pass
            def reduce_scatter(self, tensor, group, async_op=False):
                pass
            def all_to_all_4d(self, tensor, scatter_dim, gather_dim, group, use_sync=False):
                pass
        assert Complete().name == "test"


class TestTorchDistBackendName:
    def test_name(self):
        assert TorchDistBackend().name == "torch_dist"


class TestTorchDistBackendAllGather:
    def test_world_size_1_returns_unsqueezed(self):
        backend = TorchDistBackend()
        t = torch.randn(3, 4)
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.backend.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result, handle = backend.all_gather(t, mock_group)
        assert result.shape == (1, 3, 4)
        assert handle is None
        assert torch.equal(result[0], t)


class TestTorchDistBackendReduceScatter:
    def test_world_size_1_returns_squeezed(self):
        backend = TorchDistBackend()
        t = torch.randn(1, 3, 4)
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.backend.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result, handle = backend.reduce_scatter(t, mock_group)
        assert result.shape == (3, 4)
        assert handle is None
        assert torch.equal(result, t.squeeze(0))


class TestTorchDistBackendAllToAll4d:
    def test_world_size_1_returns_input(self):
        backend = TorchDistBackend()
        t = torch.randn(2, 8, 16)
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.backend.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = backend.all_to_all_4d(t, scatter_dim=2, gather_dim=1, group=mock_group)
        assert torch.equal(result, t)

    def test_invalid_dims_raises(self):
        backend = TorchDistBackend()
        t = torch.randn(2, 8, 16)
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.backend.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 2
            with pytest.raises(ValueError, match="Invalid dims"):
                backend.all_to_all_4d(t, scatter_dim=0, gather_idx=3, group=mock_group)


class TestTorchDistBackendModelSharding:
    def test_splits_params_by_rank(self):
        backend = TorchDistBackend()
        model = torch.nn.Linear(4, 2, bias=False)
        original_numel = sum(p.numel() for p in model.parameters())
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.backend.dist") as mock_dist:
            mock_dist.get_rank.return_value = 0
            mock_dist.get_world_size.return_value = 2
            backend.model_sharding(model, mock_group)
        for _, param in model.named_parameters():
            assert param.numel() == original_numel // 2


class TestGetDefaultBackend:
    def test_returns_torch_dist_backend(self):
        backend = get_default_backend()
        assert isinstance(backend, TorchDistBackend)

    def test_singleton_identity(self):
        b1 = get_default_backend()
        b2 = get_default_backend()
        assert b1 is b2
