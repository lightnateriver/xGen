"""
Unit tests for sequence_parallelism/comm.py.

Covers:
- AllGather: world_size=1 shortcut, overlap validation, forward/backward
- ReduceScatter: world_size=1 shortcut, overlap validation, forward/backward
- AllToAllAutograd: world_size=1 shortcut
- AllToAll4DAutograd: world_size=1 shortcut, invalid dims
- pad_tensor: zero/negative pad, positive pad
- _split_sequence_impl: world_size=1, pad>0, dim not divisible
- _gather_sequence_impl: world_size=1, pad>0
- gather_sequence / split_sequence: world_size=1 pass-through
- all_to_all_before_attn / all_to_all_after_attn: pad=0 path
- batch_func: tensor with batch_size=2, non-tensor, batch_size!=2
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.sequence_parallelism.comm import (
    AllGather,
    ReduceScatter,
    AllToAllAutograd,
    AllToAll4DAutograd,
    pad_tensor,
    _split_sequence_impl,
    _gather_sequence_impl,
    gather_sequence,
    split_sequence,
    all_to_all,
    reduce_scatter as reduce_scatter_fn,
    all_to_all_4d,
    all_to_all_before_attn,
    all_to_all_after_attn,
    batch_func,
)
from x_base.sequence_parallelism.errors import IncompatibleDimensionError


# ============================================================
# AllGather
# ============================================================

class TestAllGather:
    def test_world_size_1_returns_unsqueezed(self):
        t = torch.randn(3, 4)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result, handle = AllGather.forward(None, t, None, False)
        assert result.shape == (1, 3, 4)
        assert handle is None
        assert torch.equal(result[0], t)

    def test_overlap_with_none_ctx_raises(self):
        t = torch.randn(3, 4)
        with pytest.raises(ValueError, match="ctx cannot be None"):
            AllGather.forward(None, t, None, True)

    def test_forward_stores_group_in_ctx(self):
        t = torch.randn(3, 4)
        ctx = MagicMock()
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            AllGather.forward(ctx, t, mock_group, False)
        assert ctx.comm_grp == mock_group

    def test_backward_calls_reduce_scatter(self):
        ctx = MagicMock()
        ctx.comm_grp = MagicMock()
        grad = torch.randn(1, 3, 4)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = AllGather.backward(ctx, grad)
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is None


# ============================================================
# ReduceScatter
# ============================================================

class TestReduceScatter:
    def test_world_size_1_returns_squeezed(self):
        t = torch.randn(1, 3, 4)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result, handle = ReduceScatter.forward(None, t, None, False)
        assert result.shape == (3, 4)
        assert handle is None
        assert torch.equal(result, t.squeeze(0))

    def test_overlap_with_none_ctx_raises(self):
        t = torch.randn(1, 3, 4)
        with pytest.raises(ValueError, match="ctx cannot be None"):
            ReduceScatter.forward(None, t, None, True)

    def test_forward_stores_group_in_ctx(self):
        t = torch.randn(1, 3, 4)
        ctx = MagicMock()
        mock_group = MagicMock()
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            ReduceScatter.forward(ctx, t, mock_group, False)
        assert ctx.comm_grp == mock_group

    def test_backward_calls_all_gather(self):
        ctx = MagicMock()
        ctx.comm_grp = MagicMock()
        grad = torch.randn(3, 4)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = ReduceScatter.backward(ctx, grad)
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is None


# ============================================================
# AllToAllAutograd
# ============================================================

class TestAllToAllAutograd:
    def test_world_size_1_returns_input(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = AllToAllAutograd.forward(MagicMock(), t, None, 2, 1)
        assert torch.equal(result, t)

    def test_forward_saves_dims_in_ctx(self):
        ctx = MagicMock()
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            AllToAllAutograd.forward(ctx, t, None, 2, 1)
        assert ctx.scatter_dim == 2
        assert ctx.gather_dim == 1


# ============================================================
# AllToAll4DAutograd
# ============================================================

class TestAllToAll4DAutograd:
    def test_world_size_1_returns_input(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = AllToAll4DAutograd.forward(MagicMock(), None, t, 2, 1, False)
        assert torch.equal(result, t)

    def test_invalid_dims_raises(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 2
            with pytest.raises(ValueError, match="Invalid dimensions"):
                AllToAll4DAutograd.forward(MagicMock(), None, t, 0, 3, False)

    def test_forward_saves_ctx(self):
        ctx = MagicMock()
        mock_group = MagicMock()
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            AllToAll4DAutograd.forward(ctx, mock_group, t, 2, 1, True)
        assert ctx.group == mock_group
        assert ctx.scatter_dim == 2
        assert ctx.gather_dim == 1
        assert ctx.use_sync is True


# ============================================================
# pad_tensor
# ============================================================

class TestPadTensor:
    def test_zero_pad_returns_input(self):
        t = torch.randn(2, 8, 16)
        result = pad_tensor(t, 1, 0)
        assert result is t

    def test_negative_pad_returns_input(self):
        t = torch.randn(2, 8, 16)
        result = pad_tensor(t, 1, -1)
        assert result is t

    def test_positive_pad_extends_dim(self):
        t = torch.randn(2, 8, 16)
        result = pad_tensor(t, 1, 4)
        assert result.shape == (2, 12, 16)
        assert torch.equal(result[:, :8, :], t)
        assert (result[:, 8:, :] == 0).all()

    def test_pad_dim_0(self):
        t = torch.randn(4, 8)
        result = pad_tensor(t, 0, 2)
        assert result.shape == (6, 8)


# ============================================================
# _split_sequence_impl
# ============================================================

class TestSplitSequenceImpl:
    def test_world_size_1_returns_input(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = _split_sequence_impl(t, None, 1, 0)
        assert result is t

    def test_dim_not_divisible_raises(self):
        t = torch.randn(2, 7, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 2
            mock_dist.get_rank.return_value = 0
            with pytest.raises(IncompatibleDimensionError):
                _split_sequence_impl(t, None, 1, 0)

    def test_with_pad(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 2
            mock_dist.get_rank.return_value = 0
            result = _split_sequence_impl(t, None, 1, 2)
        assert result.shape[1] == 5


# ============================================================
# _gather_sequence_impl
# ============================================================

class TestGatherSequenceImpl:
    def test_world_size_1_returns_input(self):
        t = torch.randn(2, 4, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = _gather_sequence_impl(t, None, 1, 0)
        assert torch.equal(result, t)

    def test_with_pad_removes_padding(self):
        t = torch.randn(2, 4, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 2
            mock_dist.all_gather = MagicMock()
            mock_dist.all_gather.side_effect = lambda tl, tensor, group=None: [
                tl[i].copy_(tensor) for i in range(len(tl))
            ]
            result = _gather_sequence_impl(t, None, 1, 2)
        assert result.shape[1] == 4 * 2 - 2


# ============================================================
# gather_sequence / split_sequence (convenience wrappers)
# ============================================================

class TestGatherSplitSequence:
    def test_gather_world_size_1(self):
        t = torch.randn(2, 4, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = gather_sequence(t, None, dim=1)
        assert torch.equal(result, t)

    def test_split_world_size_1(self):
        t = torch.randn(2, 4, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = split_sequence(t, None, dim=1)
        assert torch.equal(result, t)


# ============================================================
# all_to_all convenience
# ============================================================

class TestAllToAllConvenience:
    def test_delegates_to_autograd(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = all_to_all(t, None, 2, 1)
        assert torch.equal(result, t)


class TestReduceScatterConvenience:
    def test_delegates_to_autograd(self):
        t = torch.randn(1, 3, 4)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result, handle = reduce_scatter_fn(t, None, False)
        assert result.shape == (3, 4)


class TestAllToAll4dConvenience:
    def test_delegates_to_autograd(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            result = all_to_all_4d(t, 2, 1, None)
        assert torch.equal(result, t)


# ============================================================
# all_to_all_before_attn / all_to_all_after_attn
# ============================================================

class TestAllToAllBeforeAttn:
    def test_no_pad(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            with patch("x_base.sequence_parallelism.comm.get_pad_manager") as mock_pm:
                mock_pm.return_value.get_or_default.return_value = 0
                result = all_to_all_before_attn(t, None, 2, 1)
        assert torch.equal(result, t)

    def test_with_pad_trims_output(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            with patch("x_base.sequence_parallelism.comm.get_pad_manager") as mock_pm:
                mock_pm.return_value.get_or_default.return_value = 2
                result = all_to_all_before_attn(t, None, 2, 1)
        assert result.shape[1] == 8 - 2


class TestAllToAllAfterAttn:
    def test_no_pad(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            with patch("x_base.sequence_parallelism.comm.get_pad_manager") as mock_pm:
                mock_pm.return_value.get_or_default.return_value = 0
                result = all_to_all_after_attn(t, None, 1, 2)
        assert torch.equal(result, t)

    def test_with_pad_extends_input(self):
        t = torch.randn(2, 8, 16)
        with patch("x_base.sequence_parallelism.comm.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 1
            with patch("x_base.sequence_parallelism.comm.get_pad_manager") as mock_pm:
                mock_pm.return_value.get_or_default.return_value = 4
                result = all_to_all_after_attn(t, None, 1, 2)
        assert result.shape[1] == 8 + 4


# ============================================================
# batch_func
# ============================================================

class TestBatchFunc:
    def test_tensor_batch_size_2(self):
        t = torch.randn(2, 10)
        result = batch_func(lambda x: x * 2, t)
        assert torch.equal(result[0], t * 2)

    def test_tensor_batch_size_not_2(self):
        t = torch.randn(3, 10)
        result = batch_func(lambda x: x * 2, t)
        assert torch.equal(result[0], t)

    def test_non_tensor_passthrough(self):
        result = batch_func(lambda x: x, 5, "hello", None)
        assert result == [5, "hello", None]

    def test_mixed_args(self):
        t2 = torch.randn(2, 10)
        t3 = torch.randn(3, 10)
        s = 42
        result = batch_func(lambda x: x * 3, t2, t3, s)
        assert torch.equal(result[0], t2 * 3)
        assert torch.equal(result[1], t3)
        assert result[2] == 42
