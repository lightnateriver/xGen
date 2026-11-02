"""
Unit tests for phaa/utils.py.

Covers:
- check_split_num_correctness: valid/invalid split_num, USP bypass
- phaa_on_pipe: Wan pipeline injection, transformer_2/3, enable_phaa/set_phaa_split_num called
- inject_phaa_for_wan: stream assignment to attn processors
"""
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from x_base.phaa.utils import (
    phaa_on_pipe,
    check_split_num_correctness,
    inject_phaa_for_wan,
)


class TestCheckSplitNumCorrectness:
    def _make_attn(self, heads=8, sp_size=2, enable_usp=False):
        attn = MagicMock()
        attn.heads = heads
        attn.parallel_manager = MagicMock()
        attn.parallel_manager.sp_size = sp_size
        attn.parallel_manager.enable_usp = enable_usp
        return attn

    def test_valid_split_num(self):
        attn = self._make_attn(heads=8, sp_size=2)
        check_split_num_correctness(attn, 2)

    def test_invalid_split_num_raises(self):
        attn = self._make_attn(heads=8, sp_size=2)
        with pytest.raises(ValueError, match="divisible"):
            check_split_num_correctness(attn, 3)

    def test_none_split_num_skips_check(self):
        attn = self._make_attn(heads=8, sp_size=2)
        check_split_num_correctness(attn, None)

    def test_usp_bypasses_check(self):
        attn = self._make_attn(heads=8, sp_size=2, enable_usp=True)
        check_split_num_correctness(attn, 3)

    def test_split_num_one_always_valid(self):
        attn = self._make_attn(heads=7, sp_size=1)
        check_split_num_correctness(attn, 1)


class TestPhaaOnPipe:
    def _make_wan_pipe(self, has_t2=False, has_t3=False):
        pipe = MagicMock()
        pipe.__class__ = type("WanPipeline", (), {})
        pipe.transformer = MagicMock()
        pipe.transformer.blocks = [MagicMock()]
        pipe.transformer_2 = MagicMock() if has_t2 else None
        if has_t2:
            pipe.transformer_2.blocks = [MagicMock()]
        pipe.transformer_3 = MagicMock() if has_t3 else None
        if has_t3:
            pipe.transformer_3.blocks = [MagicMock()]
        return pipe

    @patch("x_base.phaa.utils.inject_phaa_for_wan")
    @patch("x_base.phaa.utils.enable_phaa")
    @patch("x_base.phaa.utils.set_phaa_split_num")
    def test_calls_inject_on_transformer(self, mock_set, mock_enable, mock_inject):
        pipe = self._make_wan_pipe()
        phaa_on_pipe(pipe, phaa_split_num=4)
        mock_inject.assert_called_once_with(pipe.transformer, 4)
        mock_set.assert_called_once_with(4)
        mock_enable.assert_called_once()

    @patch("x_base.phaa.utils.inject_phaa_for_wan")
    @patch("x_base.phaa.utils.enable_phaa")
    @patch("x_base.phaa.utils.set_phaa_split_num")
    def test_transformer_2_injected(self, mock_set, mock_enable, mock_inject):
        pipe = self._make_wan_pipe(has_t2=True)
        phaa_on_pipe(pipe, transformer_2=True, phaa_split_num=2)
        assert mock_inject.call_count == 2

    @patch("x_base.phaa.utils.inject_phaa_for_wan")
    @patch("x_base.phaa.utils.enable_phaa")
    @patch("x_base.phaa.utils.set_phaa_split_num")
    def test_transformer_3_injected(self, mock_set, mock_enable, mock_inject):
        pipe = self._make_wan_pipe(has_t3=True)
        phaa_on_pipe(pipe, transformer_3=True, phaa_split_num=2)
        assert mock_inject.call_count == 2

    @patch("x_base.phaa.utils.inject_phaa_for_wan")
    @patch("x_base.phaa.utils.enable_phaa")
    @patch("x_base.phaa.utils.set_phaa_split_num")
    def test_non_wan_pipe_skips_inject(self, mock_set, mock_enable, mock_inject):
        pipe = MagicMock()
        pipe.__class__ = type("HunyuanPipeline", (), {})
        phaa_on_pipe(pipe, phaa_split_num=4)
        mock_inject.assert_not_called()
        mock_set.assert_called_once_with(4)
        mock_enable.assert_called_once()


class TestInjectPhaaForWan:
    @patch("x_base.phaa.utils.Stream", return_value=MagicMock())
    @patch("x_base.phaa.utils.check_split_num_correctness")
    def test_assigns_comm_stream_to_processors(self, mock_check, mock_stream_cls):
        block = MagicMock()
        transformer = MagicMock()
        transformer.blocks = [block]
        inject_phaa_for_wan(transformer, split_num=2)
        assert block.attn1.processor.comm_stream is mock_stream_cls.return_value
        assert block.attn2.processor.comm_stream is mock_stream_cls.return_value
        assert mock_check.call_count == 2

    @patch("x_base.phaa.utils.Stream", return_value=MagicMock())
    @patch("x_base.phaa.utils.check_split_num_correctness")
    def test_multiple_blocks(self, mock_check, mock_stream_cls):
        blocks = [MagicMock() for _ in range(3)]
        transformer = MagicMock()
        transformer.blocks = blocks
        inject_phaa_for_wan(transformer, split_num=2)
        assert mock_check.call_count == 6
        for block in blocks:
            assert block.attn1.processor.comm_stream is mock_stream_cls.return_value
