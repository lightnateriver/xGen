"""
Unit tests for sequence_parallelism/pipeline.py.

Covers:
- Parallelizable / PipelineAdapter protocol conformance checks
- enable_sp: sp_size=None fallback, dp_size computation, ulysses*ring validation,
  world_size divisibility, transformer_2/3 enabling, transformer_3 sp cap
"""
from unittest.mock import patch, MagicMock

import pytest

from x_base.sequence_parallelism.pipeline import (
    Parallelizable,
    PipelineAdapter,
    enable_sp,
)


class TestParallelizableProtocol:
    def test_conforming_class_is_instance(self):
        class Foo:
            def enable_parallel(self, dp_size, sp_size, enable_cp, ulysses_size=None, ring_size=None, **kwargs):
                pass
        assert isinstance(Foo(), Parallelizable)

    def test_non_conforming_class_is_not_instance(self):
        class Bar:
            pass
        assert not isinstance(Bar(), Parallelizable)


class TestPipelineAdapterProtocol:
    def test_conforming_class_is_instance(self):
        class Pipe:
            @property
            def transformer(self):
                return MagicMock()
            @property
            def transformer_2(self):
                return None
            @property
            def transformer_3(self):
                return None
        assert isinstance(Pipe(), PipelineAdapter)


class TestEnableSp:
    def _make_pipeline(self, has_t2=False, has_t3=False):
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock() if has_t2 else None
        pipe.transformer_3 = MagicMock() if has_t3 else None
        return pipe

    def test_sp_size_none_uses_world_size(self):
        pipe = self._make_pipeline()
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 8
            enable_sp(pipe, sp_size=None)
        pipe.transformer.enable_parallel.assert_called_once_with(
            1, 8, enable_cp=False, ulysses_size=None, ring_size=None
        )

    def test_explicit_sp_size_computes_dp(self):
        pipe = self._make_pipeline()
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 8
            enable_sp(pipe, sp_size=4)
        pipe.transformer.enable_parallel.assert_called_once_with(
            2, 4, enable_cp=False, ulysses_size=None, ring_size=None
        )

    def test_ulysses_ring_mismatch_raises(self):
        pipe = self._make_pipeline()
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 8
            with pytest.raises(ValueError, match="must be equal"):
                enable_sp(pipe, sp_size=4, ulysses_size=2, ring_size=3)

    def test_world_size_not_divisible_by_sp_raises(self):
        pipe = self._make_pipeline()
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 7
            with pytest.raises(ValueError, match="divisible"):
                enable_sp(pipe, sp_size=4)

    def test_enable_transformer_2(self):
        pipe = self._make_pipeline(has_t2=True)
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            enable_sp(pipe, sp_size=2, enable_transformer_2=True)
        pipe.transformer_2.enable_parallel.assert_called_once()

    def test_enable_transformer_2_skipped_when_none(self):
        pipe = self._make_pipeline(has_t2=False)
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            enable_sp(pipe, sp_size=2, enable_transformer_2=True)

    def test_enable_transformer_3_caps_sp_at_4(self):
        pipe = self._make_pipeline(has_t3=True)
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 8
            enable_sp(pipe, sp_size=8, enable_transformer_3=True)
        pipe.transformer_3.enable_parallel.assert_called_once_with(
            2, 4, enable_cp=False, ulysses_size=None, ring_size=None
        )

    def test_enable_transformer_3_small_sp_no_cap(self):
        pipe = self._make_pipeline(has_t3=True)
        with patch("x_base.sequence_parallelism.pipeline.dist") as mock_dist:
            mock_dist.get_world_size.return_value = 4
            enable_sp(pipe, sp_size=2, enable_transformer_3=True)
        pipe.transformer_3.enable_parallel.assert_called_once_with(
            2, 2, enable_cp=False, ulysses_size=None, ring_size=None
        )
