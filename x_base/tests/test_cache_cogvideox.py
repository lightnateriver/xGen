"""
Unit tests for cache/models/cogvideox.py.

Covers:
- _teacache_should_calc: disabled, first/last step, cache hit, cache miss, cnt reset
- _teacache_imple: cache disabled, cache hit (reuse residual), cache miss (compute + store residual)
- _infer_cogvideox_model_key: 2b default, 5b, 5b-i2v, 1.5, 1.5-i2v
- teacache_init: sets class attributes on transformer
"""
from unittest.mock import patch, MagicMock

import pytest
import torch
import numpy as np

from x_base.cache.models.cogvideox import (
    _teacache_should_calc,
    _teacache_imple,
    _infer_cogvideox_model_key,
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
    s.previous_residual_encoder = torch.ones(2, 8) * 0.1
    s.gradient_checkpointing = False
    s.transformer_blocks = []
    return s


class TestTeacacheShouldCalc:
    def test_disabled_always_calc(self):
        s = _make_self(enable_teacache=False)
        assert _teacache_should_calc(s, torch.randn(2, 8)) is True

    def test_first_step_forces_calc(self):
        s = _make_self(cnt=0)
        emb = torch.randn(2, 8)
        result = _teacache_should_calc(s, emb)
        assert result is True
        assert s.accumulated_rel_l1_distance == 0

    def test_last_step_forces_calc(self):
        s = _make_self(cnt=9, num_steps=10)
        emb = torch.randn(2, 8)
        result = _teacache_should_calc(s, emb)
        assert result is True

    def test_cache_hit_below_threshold(self):
        s = _make_self(cnt=1, num_steps=10)
        emb = s.previous_modulated_input.clone()
        result = _teacache_should_calc(s, emb)
        assert result is False

    def test_cache_miss_above_threshold(self):
        s = _make_self(cnt=1, num_steps=10)
        emb = torch.zeros(2, 8)
        result = _teacache_should_calc(s, emb)
        assert result is True
        assert s.accumulated_rel_l1_distance == 0

    def test_cnt_resets_at_num_steps(self):
        s = _make_self(cnt=9, num_steps=10)
        _teacache_should_calc(s, torch.randn(2, 8))
        assert s.cnt == 0

    def test_cnt_increments(self):
        s = _make_self(cnt=1, num_steps=10)
        _teacache_should_calc(s, s.previous_modulated_input.clone())
        assert s.cnt == 2

    def test_previous_modulated_input_updated(self):
        s = _make_self(cnt=1, num_steps=10)
        emb = torch.randn(2, 8)
        _teacache_should_calc(s, emb)
        assert torch.equal(s.previous_modulated_input, emb)


class TestTeacacheImple:
    def test_disabled_runs_blocks(self):
        s = _make_self(enable_teacache=False)
        block = MagicMock(return_value=(torch.randn(2, 8), torch.randn(2, 8)))
        s.transformer_blocks = [block]
        hs = torch.randn(2, 8)
        enc = torch.randn(2, 8)
        emb = torch.randn(2, 8)
        out_hs, out_enc = _teacache_imple(s, hs, enc, True, emb, None)
        block.assert_called_once()

    def test_cache_hit_reuses_residual(self):
        s = _make_self(enable_teacache=True)
        hs = torch.zeros(2, 8)
        enc = torch.zeros(2, 8)
        residual = torch.ones(2, 8) * 0.5
        s.previous_residual = residual.clone()
        s.previous_residual_encoder = residual.clone()
        out_hs, out_enc = _teacache_imple(s, hs, enc, False, None, None)
        assert torch.equal(out_hs, residual)
        assert torch.equal(out_enc, residual)

    def test_cache_miss_computes_and_stores(self):
        s = _make_self(enable_teacache=True)
        block = MagicMock(return_value=(torch.ones(2, 8) * 3, torch.ones(2, 8) * 4))
        s.transformer_blocks = [block]
        hs = torch.ones(2, 8)
        enc = torch.ones(2, 8)
        emb = torch.randn(2, 8)
        out_hs, out_enc = _teacache_imple(s, hs, enc, True, emb, None)
        block.assert_called_once()
        assert s.previous_residual is not None
        assert s.previous_residual_encoder is not None


class TestInferCogvideoxModelKey:
    def test_default_2b(self):
        args = MagicMock()
        args.model = "CogVideoX"
        assert _infer_cogvideox_model_key(args) == "CogVideoX-2b"

    def test_5b(self):
        args = MagicMock()
        args.model = "CogVideoX-5b"
        assert _infer_cogvideox_model_key(args) == "CogVideoX-5b"

    def test_5b_i2v(self):
        args = MagicMock()
        args.model = "CogVideoX-5b-I2V"
        assert _infer_cogvideox_model_key(args) == "CogVideoX-5b-I2V"

    def test_1_5(self):
        args = MagicMock()
        args.model = "CogVideoX1.5-5B"
        assert _infer_cogvideox_model_key(args) == "CogVideoX1.5-5B"

    def test_1_5_i2v(self):
        args = MagicMock()
        args.model = "CogVideoX1.5-5B-I2V"
        assert _infer_cogvideox_model_key(args) == "CogVideoX1.5-5B-I2V"

    def test_no_model_attr(self):
        args = MagicMock(spec=[])
        assert _infer_cogvideox_model_key(args) == "CogVideoX-2b"


class TestTeacacheInit:
    def test_sets_class_attributes(self):
        pipe = MagicMock()
        pipe.transformer.__class__ = type("MockTransformer", (), {})
        args = MagicMock()
        args.num_inference_steps = 50
        with patch("x_base.cache.models.cogvideox.get_teacache_config", return_value={"coefficients": [1.0, 2.0]}):
            teacache_init(pipe, args)
        cls = pipe.transformer.__class__
        assert cls.enable_teacache is True
        assert cls.cnt == 0
        assert cls.num_steps == 50
        assert cls.rel_l1_thresh == REL_L1_THRESH
        assert cls.accumulated_rel_l1_distance == 0
        assert cls.previous_modulated_input is None
        assert cls.previous_residual is None
        assert cls.previous_residual_encoder is None
        assert cls.coefficients == [1.0, 2.0]
