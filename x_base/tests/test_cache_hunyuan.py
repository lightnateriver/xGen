"""
Unit tests for cache/models/hunyuan.py.

Covers:
- _teacache_should_calc: disabled, first/last step, cache hit, cache miss, cnt reset, previous_modulated_input updated
- _teacache_imple: disabled, cache hit (reuse residual), cache miss (compute + store residual)
- teacache_init: sets class attributes on transformer
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.cache.models.hunyuan import (
    _teacache_should_calc,
    _teacache_imple,
    teacache_init,
    REL_L1_THRESH,
)


def _make_self(enable_teacache=True, num_steps=10, cnt=1):
    s = MagicMock()
    s.enable_teacache = enable_teacache
    s.num_steps = num_steps
    s.cnt = cnt
    s.accumulated_rel_l1_distance = 0.0
    s.rel_l1_thresh = REL_L1_THRESH
    s.coefficients = [0.0, 1.0]
    s.previous_modulated_input = torch.ones(2, 8)
    s.previous_residual = torch.ones(2, 8) * 0.1
    s.transformer_blocks = []
    s.single_transformer_blocks = []

    norm_result = (torch.ones(2, 8), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    s.transformer_blocks = [MagicMock()]
    s.transformer_blocks[0].norm1.return_value = norm_result
    return s


class TestTeacacheShouldCalc:
    def test_disabled_always_calc(self):
        s = _make_self(enable_teacache=False)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        assert _teacache_should_calc(s, hs, temb) is True

    def test_first_step_forces_calc(self):
        s = _make_self(cnt=0)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        result = _teacache_should_calc(s, hs, temb)
        assert result is True
        assert s.accumulated_rel_l1_distance == 0

    def test_last_step_forces_calc(self):
        s = _make_self(cnt=9, num_steps=10)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        result = _teacache_should_calc(s, hs, temb)
        assert result is True

    def test_cache_hit_below_threshold(self):
        s = _make_self(cnt=1, num_steps=10)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        s.transformer_blocks[0].norm1.return_value = (
            s.previous_modulated_input.clone(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        result = _teacache_should_calc(s, hs, temb)
        assert result is False

    def test_cache_miss_above_threshold(self):
        s = _make_self(cnt=1, num_steps=10)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        s.transformer_blocks[0].norm1.return_value = (
            torch.zeros(2, 8), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        result = _teacache_should_calc(s, hs, temb)
        assert result is True
        assert s.accumulated_rel_l1_distance == 0

    def test_cnt_resets_at_num_steps(self):
        s = _make_self(cnt=9, num_steps=10)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        _teacache_should_calc(s, hs, temb)
        assert s.cnt == 0

    def test_cnt_increments(self):
        s = _make_self(cnt=1, num_steps=10)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        _teacache_should_calc(s, hs, temb)
        assert s.cnt == 2

    def test_previous_modulated_input_updated(self):
        s = _make_self(cnt=1, num_steps=10)
        hs = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        new_norm = torch.randn(2, 8)
        s.transformer_blocks[0].norm1.return_value = (
            new_norm, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        _teacache_should_calc(s, hs, temb)
        assert torch.equal(s.previous_modulated_input, new_norm)


class TestTeacacheImple:
    def test_disabled_runs_all_blocks(self):
        s = _make_self(enable_teacache=False)
        block = MagicMock(return_value=(torch.randn(2, 8), torch.randn(2, 8)))
        single_block = MagicMock(return_value=(torch.randn(2, 8), torch.randn(2, 8)))
        s.transformer_blocks = [block]
        s.single_transformer_blocks = [single_block]
        hs = torch.randn(2, 8)
        enc = torch.randn(2, 8)
        temb = torch.randn(2, 8)
        out_hs, out_enc = _teacache_imple(s, hs, enc, temb, None, None, True)
        block.assert_called_once()
        single_block.assert_called_once()

    def test_cache_hit_reuses_residual(self):
        s = _make_self(enable_teacache=True)
        hs = torch.zeros(2, 8)
        enc = torch.zeros(2, 8)
        residual = torch.ones(2, 8) * 0.5
        s.previous_residual = residual.clone()
        out_hs, out_enc = _teacache_imple(s, hs, enc, None, None, None, False)
        assert torch.equal(out_hs, residual)

    def test_cache_miss_computes_and_stores(self):
        s = _make_self(enable_teacache=True)
        block = MagicMock(return_value=(torch.ones(2, 8) * 3, torch.ones(2, 8) * 4))
        single_block = MagicMock(return_value=(torch.ones(2, 8) * 5, torch.ones(2, 8) * 6))
        s.transformer_blocks = [block]
        s.single_transformer_blocks = [single_block]
        hs = torch.ones(2, 8)
        enc = torch.ones(2, 8)
        temb = torch.randn(2, 8)
        out_hs, out_enc = _teacache_imple(s, hs, enc, temb, None, None, True)
        block.assert_called_once()
        single_block.assert_called_once()
        assert s.previous_residual is not None


class TestTeacacheInit:
    def test_sets_class_attributes(self):
        pipe = MagicMock()
        pipe.transformer.__class__ = type("MockTransformer", (), {})
        args = MagicMock()
        args.num_inference_steps = 50
        with patch("x_base.cache.models.hunyuan.get_teacache_config", return_value={"coefficients": [1.0, 2.0]}):
            teacache_init(pipe, args)
        cls = pipe.transformer.__class__
        assert cls.enable_teacache is True
        assert cls.cnt == 0
        assert cls.num_steps == 50
        assert cls.rel_l1_thresh == REL_L1_THRESH
        assert cls.accumulated_rel_l1_distance == 0
        assert cls.previous_modulated_input is None
        assert cls.previous_residual is None
        assert cls.coefficients == [1.0, 2.0]
