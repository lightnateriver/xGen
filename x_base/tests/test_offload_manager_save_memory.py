"""
Unit tests for infinite_vram/offload_manager_save_memory.py.

Covers:
- OffloadManager_For_Save_Memory.__init__: keep_n normalization, sync_at_layer validation
- _maybe_free_prev_layer: boundary & state checks
- _wait_boardcast_if_needed / _wait_h2d_if_needed: skip for keep_n, wait & clear event
- _should_prefetch_next: boundary & state checks
- _release_factory: below keep_n noop, above sets flags
- _release_tensor: storage resize
- _restore_all: restore released params
- enable/disable lifecycle
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.infinite_vram.offload_manager_save_memory import OffloadManager_For_Save_Memory


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
    with patch("x_base.infinite_vram.offload_manager_save_memory.torch.cuda.Stream", return_value=MagicMock()):
        mgr = OffloadManager_For_Save_Memory(**defaults)
    return mgr


class TestInit:
    def test_keep_n_int_normalized(self):
        mgr = _make_mgr(keep_n=2)
        assert mgr.keep_n == {"blocks": 2}

    def test_keep_n_dict_preserved(self):
        mgr = _make_mgr(keep_n={"blocks": 1})
        assert mgr.keep_n == {"blocks": 1}

    def test_sync_at_layer_without_dist_raises(self):
        with patch("x_base.infinite_vram.offload_manager_save_memory.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            with pytest.raises(RuntimeError, match="未初始化"):
                OffloadManager_For_Save_Memory(
                    root=torch.nn.Linear(4, 2),
                    module_groups={"blocks": [_make_module()]},
                    keep_n=1,
                    sync_at_layer=True,
                )

    def test_enabled_defaults_false(self):
        mgr = _make_mgr()
        assert mgr.enabled is False

    def test_index_depth_set(self):
        mods = [_make_module() for _ in range(3)]
        with patch("x_base.infinite_vram.offload_manager_save_memory.torch.cuda.Stream", return_value=MagicMock()):
            mgr = OffloadManager_For_Save_Memory(
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

    def test_not_needs_free_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod._so_needs_free = False
        mgr._maybe_free_prev_layer(1, 1, [MagicMock(), mod], [[]])

    def test_already_freed_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod._so_needs_free = True
        mod._so_freed = True
        mod._so_compute_evt = MagicMock()
        mgr._maybe_free_prev_layer(1, 1, [MagicMock(), mod], [[]])

    def test_no_compute_evt_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod._so_needs_free = True
        mod._so_freed = False
        mod._so_compute_evt = None
        mgr._maybe_free_prev_layer(1, 1, [MagicMock(), mod], [[]])


class TestWaitBoardcastIfNeeded:
    def test_below_keep_n_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 0
        mgr._wait_boardcast_if_needed(mod, 1)

    def test_no_event_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 1
        mgr._wait_boardcast_if_needed(mod, 1)

    @patch("x_base.infinite_vram.offload_manager_save_memory.torch.cuda.current_stream", return_value=MagicMock())
    def test_waits_and_clears(self, mock_stream):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 1
        mod._so_boardcast_evt = MagicMock()
        mgr._wait_boardcast_if_needed(mod, 1)
        assert mod._so_boardcast_evt is None


class TestWaitH2dIfNeeded:
    def test_below_keep_n_skips(self):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 0
        mgr._wait_h2d_if_needed(mod, 1)

    @patch("x_base.infinite_vram.offload_manager_save_memory.torch.cuda.current_stream", return_value=MagicMock())
    def test_waits_and_clears(self, mock_stream):
        mgr = _make_mgr()
        mod = _make_module()
        mod.index = 1
        mod._so_h2d_evt = MagicMock()
        mgr._wait_h2d_if_needed(mod, 1)
        assert mod._so_h2d_evt is None


class TestShouldPrefetchNext:
    def test_below_keep_n(self):
        mgr = _make_mgr()
        assert mgr._should_prefetch_next(0, 1, []) is False

    def test_beyond_length(self):
        mgr = _make_mgr()
        assert mgr._should_prefetch_next(2, 1, [MagicMock()]) is False

    def test_already_has_event(self):
        mgr = _make_mgr()
        mod = MagicMock()
        mod._so_boardcast_evt = MagicMock()
        assert mgr._should_prefetch_next(1, 1, [MagicMock(), mod]) is False

    def test_should_prefetch(self):
        mgr = _make_mgr()
        mod = MagicMock(spec=[])
        assert mgr._should_prefetch_next(1, 1, [MagicMock(), mod]) is True


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
    def test_releases_and_marks(self):
        p = torch.nn.Parameter(torch.randn(3, 4))
        OffloadManager_For_Save_Memory._release_tensor(p)
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
                OffloadManager_For_Save_Memory._release_tensor(p)

        mgr._restore_all()
        for idx in range(len(mods)):
            for p in mgr.layer_params["blocks"][idx]:
                assert p._released is False
                assert p.data.numel() > 0


class TestEnableDisable:
    def test_enable_sets_flag(self):
        mgr = _make_mgr()
        with patch.object(mgr, "_setup_parameter_offload"):
            with patch("x_base.infinite_vram.offload_manager_save_memory.torch.cuda.empty_cache"):
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
