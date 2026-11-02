"""
Unit tests for x_diffusers/framework/transformer/attention_backend.py.

Covers:
- _native_npu_attention: return_lse=True raises, no parallel_config path (permute + attention_manager),
  with parallel_config path (delegates to _templated_context_parallel_attention)
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_diffusers.framework.transformer.attention_backend import _native_npu_attention


class TestNativeNpuAttention:
    def test_return_lse_raises(self):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        with pytest.raises(ValueError, match="return_lse"):
            _native_npu_attention(q, k, v, return_lse=True)

    @patch("x_diffusers.framework.transformer.attention_backend.attention_manager")
    @patch("x_diffusers.framework.transformer.attention_backend._maybe_modify_attn_mask_npu")
    def test_no_parallel_config_path(self, mock_mask, mock_mgr):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        mock_out = torch.randn(1, 2, 4, 8)
        mock_mgr.attention.return_value = mock_out
        mock_out.type_as.return_value = torch.randn(1, 4, 2, 8)
        result = _native_npu_attention(q, k, v)
        mock_mgr.attention.assert_called_once()

    @patch("x_diffusers.framework.transformer.attention_backend._templated_context_parallel_attention")
    def test_with_parallel_config(self, mock_cp):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        mock_config = MagicMock()
        mock_cp.return_value = torch.randn(1, 2, 4, 8)
        result = _native_npu_attention(q, k, v, _parallel_config=mock_config)
        mock_cp.assert_called_once()
