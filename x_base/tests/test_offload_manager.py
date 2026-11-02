"""
Unit tests for infinite_vram/offload_manager.py.

Covers:
- OffloadManager.__init__: keep_n int->dict normalization, sync_at_layer validation
- _prefetch_factory / _release_factory hook logic (mocked CUDA)
- _maybe_free_prev_layer: boundary & state checks
- _wait_prefetch_if_needed: skip for keep_n layers, wait for event
- _should_prefetch_next: boundary & state checks
- _release_tensor: storage resize
- _restore_all: restore released params
- enable/disable lifecycle
"""
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import torch

from x_base.infinite_vram.offload_manager import OffloadManager


def _make_module():
    m = torch.nn.Linear(4, 2)
    m.index = 0
    m.depth = 3
    return m


def _make_mgr(**kwargs):
    root = torch.nn.Linear(4, 2)
    mods = [_make_module() for _ in range(3)]
    defaults = dict(
        root=root,
        module_groups={"blocks": mods},
        keep_n=1,
    )
    defaults.update(kwargs)
    with patch("x_base.infinite_vram.offload_manager.torch.cuda.Stream", return_value=MagicMock()):
        mgr = OffloadManager(**defaults)
    return mgr


class TestOffloadManagerInit:
    def test_keep_n_int_normalized_to_dict(self):
        mgr = _make_mgr(keep_n=2)
        assert mgr.keep_n == {"blocks": 2}

    def test_keep_n_dict_preserved(self):
        mgr = _make_mgr(keep_n={"blocks": 1})
        assert mgr.keep_n == {"blocks": 1}

    def test_sync_at_layer_without_dist_raises(self):
        with patch("x_base.infinite_vram.offload_manager.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            with pytest.raises(RuntimeError, match="未初始化"):
                OffloadManager(
                    root=torch.nn.Linear(4, 2),
                    module_groups={"blocks": [_make_module()]},
                    keep_n=1,
                    sync_at_layer=True,
                )

    def test_device_from_root(self):
        mgr = _make_mgr()
        assert mgr.device is not None

    def test_enabled_defaults_false(self):
        mgr = _make_mgr()
        assert mgr.enabled is False

    def test_index_depth_set_on_modules(self):
        mods = [_make_module() for _ in range(3)]
        with patch("x_base.infinite_vram.offload_manager.torch.cuda.Stream", return_value=MagicMock()):
            mgr = OffloadManager(
                root=torch.nn.Linear(4, 2),
                module_groups={"blocks": mods},
                keep_n=1,
            )
        for i, m in enumerate(mods):
            assert m.index == i
            assert m.depth == 3


class TestMaybeFreePrevLayer:
    def test_prev_below_keep_n_skips(self):
        mgr = _make_mgr()
        mgr._maybe_free_prev_layer(0, 1, [], [])

    def test_prev_not_needs_free_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod._so_needs_free = False
        mgr._maybe_free_prev_layer(1, 1, [MagicMock(), mod], [[]])
        assert not getattr(mod, "_so_freed", False)

    def test_prev_already_freed_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod._so_needs_free = True
        mod._so_freed = True
        mod._so_compute_evt = MagicMock()
        mgr._maybe_free_prev_layer(1, 1, [MagicMock(), mod], [[]])

    def test_prev_no_compute_evt_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod._so_needs_free = True
        mod._so_freed = False
        mod._so_compute_evt = None
        mgr._maybe_free_prev_layer(1, 1, [MagicMock(), mod], [[]])


class TestWaitPrefetchIfNeeded:
    def test_below_keep_n_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 0
        mgr._wait_prefetch_if_needed(mod, 1)

    def test_no_event_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 1
        mgr._wait_prefetch_if_needed(mod, 1)

    @patch("x_base.infinite_vram.offload_manager.torch.cuda.current_stream", return_value=MagicMock())
    def test_waits_and_clears_event(self, mock_stream):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 1
        mod._so_prefetch_evt = MagicMock()
        mgr._wait_prefetch_if_needed(mod, 1)
        assert mod._so_prefetch_evt is None


class TestShouldPrefetchNext:
    def test_below_keep_n(self):
        mgr = _make_mgr()
        assert mgr._should_prefetch_next(0, 1, []) is False

    def test_beyond_group_length(self):
        mgr = _make_mgr()
        grp = [MagicMock()]
        assert mgr._should_prefetch_next(2, 1, grp) is False

    def test_already_prefetched(self):
        mgr = _make_mgr()
        mod = MagicMock()
        mod._so_prefetch_evt = MagicMock()
        grp = [MagicMock(), mod]
        assert mgr._should_prefetch_next(1, 1, grp) is False

    def test_should_prefetch(self):
        mgr = _make_mgr()
        mod = MagicMock(spec=[])
        grp = [MagicMock(), mod]
        assert mgr._should_prefetch_next(1, 1, grp) is True


class TestReleaseFactory:
    def test_below_keep_n_noop(self):
        mgr = _make_mgr()
        hook = mgr._release_factory("blocks")
        mod = _make_module()
        mod.index = 0
        hook(mod, None, None)
        assert not hasattr(mod, "_so_needs_free")

    def test_above_keep_n_sets_flags(self):
        mgr = _make_mgr()
        hook = mgr._release_factory("blocks")
        mod = _make_module()
        mod.index = 1
        hook(mod, None, None)
        assert mod._so_needs_free is True
        assert mod._so_freed is False
        assert mod._so_compute_evt is not None


class TestReleaseTensor:
    def test_releases_and_marks_flag(self):
        p = torch.nn.Parameter(torch.randn(3, 4))
        OffloadManager._release_tensor(p)
        assert p._released is True
        assert p.data.numel() == 0


class TestRestoreAll:
    def test_restores_released_param(self):
        mgr = _make_mgr()
        mods = mgr.groups["blocks"]
        for idx in range(len(mods)):
            for p in mgr.layer_params["blocks"][idx]:
                p.orig_shape = p.data.shape
                p.p_cpu = p.data.clone()
                OffloadManager._release_tensor(p)

        mgr._restore_all()
        for idx in range(len(mods)):
            for p in mgr.layer_params["blocks"][idx]:
                assert p._released is False
                assert p.data.numel() > 0


class TestEnableDisable:
    def test_enable_sets_flag(self):
        mgr = _make_mgr()
        with patch.object(mgr, "_setup_parameter_offload"):
            with patch("x_base.infinite_vram.offload_manager.torch.cuda.empty_cache"):
                mgr.enable()
        assert mgr.enabled is True

    def test_enable_idempotent(self):
        mgr = _make_mgr()
        mgr.enabled = True
        mgr.enable()
        assert mgr.enabled is True

    def test_disable_restores_and_removes_hooks(self):
        mgr = _make_mgr()
        mgr.enabled = True
        with patch.object(mgr, "_restore_all"):
            with patch.object(mgr, "_remove_hooks"):
                mgr.disable()
        assert mgr.enabled is False

    def test_disable_idempotent(self):
        mgr = _make_mgr()
        mgr.enabled = False
        mgr.disable()
        assert mgr.enabled is False
