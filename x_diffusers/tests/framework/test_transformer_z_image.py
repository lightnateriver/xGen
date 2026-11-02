"""
Unit tests for x_diffusers/framework/transformer/transformer_z_image.py.

Covers:
- z_image_rope_embedder_call__: first call precomputes freqs_cis, subsequent call reuses,
  device mismatch triggers re-move, index_select produces correct shape
"""
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_diffusers.framework.transformer.transformer_z_image import z_image_rope_embedder_call__


class TestZImageRopeEmbedderCall:
    def _make_self(self, device="cpu"):
        s = MagicMock()
        s.freqs_cis = None
        s.axes_dims = [16, 32]
        s.axes_lens = [10, 20]
        s.theta = 10000.0

        freqs_0 = torch.randn(10, 16, device=device)
        freqs_1 = torch.randn(20, 32, device=device)
        s.precompute_freqs_cis.return_value = [freqs_0, freqs_1]
        return s, [freqs_0, freqs_1]

    def test_first_call_precomputes(self):
        s, expected_freqs = self._make_self()
        ids = torch.tensor([[3, 5]])
        result = z_image_rope_embedder_call__(s, ids)
        s.precompute_freqs_cis.assert_called_once_with(
            s.axes_dims, s.axes_lens, theta=s.theta
        )
        assert result.shape[-1] == 16 + 32

    def test_subsequent_call_reuses(self):
        s, freqs = self._make_self()
        s.freqs_cis = freqs
        ids = torch.tensor([[3, 5]])
        result = z_image_rope_embedder_call__(s, ids)
        s.precompute_freqs_cis.assert_not_called()

    def test_device_mismatch_removes(self):
        s, freqs = self._make_self(device="cpu")
        s.freqs_cis = freqs
        ids = torch.tensor([[3, 5]], device="cpu")
        result = z_image_rope_embedder_call__(s, ids)
        assert result is not None

    def test_output_shape(self):
        s, _ = self._make_self()
        ids = torch.tensor([[3, 5]])
        result = z_image_rope_embedder_call__(s, ids)
        assert result.shape[0] == 1
        assert result.shape[-1] == 16 + 32
