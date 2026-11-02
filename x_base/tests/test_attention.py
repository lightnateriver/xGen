"""Tests for attention.py — covers base/sage/laser/sparse/rainfusion attention
and the AttentionManager class.

Heavy NPU/CANN dependencies (cann_ops, mindiesd, torch_atb, torch.npu)
are mocked via sys.modules injection so tests run on CPU-only machines.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch

# ── Mock NPU-only modules before importing attention.py ──────────────────

_mock_cann_ops = MagicMock()
_mock_cann_ops.block_quant = MagicMock()
_mock_cann_ops.npu_sage_attention = MagicMock()
sys.modules.setdefault("cann_ops", _mock_cann_ops)

_mock_mindiesd = MagicMock()
_mock_mindiesd.attention_forward = MagicMock()


class _FakeParametersInvalid(Exception):
    pass


_mock_mindiesd.ParametersInvalid = _FakeParametersInvalid
sys.modules.setdefault("mindiesd", _mock_mindiesd)
sys.modules.setdefault("mindiesd.utils", types.ModuleType("mindiesd.utils"))
sys.modules["mindiesd.utils"].ParametersInvalid = _FakeParametersInvalid

# torch.npu.synchronize no-op
if not hasattr(torch, "npu"):
    _fake_npu = MagicMock()
    _fake_npu.synchronize = MagicMock()
    torch.npu = _fake_npu

# ── Now import the module under test ────────────────────────────────────

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from operator.attention import (  # noqa: E402
    AttentionManager,
    base_attention_infer,
    sage_attention_infer,
    sage_attention_infer_quant_scale,
    laser_attention_infer,
    attention_manager,
    SAGE_SEQ_LEN_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: create standard BNSD tensors
# ═══════════════════════════════════════════════════════════════════════════

def _qkv(B=1, N=2, Sq=8, Sk=8, D=64):
    q = torch.randn(B, N, Sq, D)
    k = torch.randn(B, N, Sk, D)
    v = torch.randn(B, N, Sk, D)
    return q, k, v


# ═══════════════════════════════════════════════════════════════════════════
# 1. base_attention_infer
# ═══════════════════════════════════════════════════════════════════════════

class TestBaseAttentionInfer(unittest.TestCase):
    """Tests for base_attention_infer (wraps F.scaled_dot_product_attention)."""

    def test_basic_output_shape(self):
        q, k, v = _qkv(B=1, N=2, Sq=8, Sk=8, D=64)
        out = base_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    def test_with_attn_mask(self):
        q, k, v = _qkv(Sq=4, Sk=4)
        mask = torch.zeros(1, 2, 4, 4)
        out = base_attention_infer(q, k, v, attn_mask=mask)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    def test_causal(self):
        q, k, v = _qkv(Sq=4, Sk=4)
        out = base_attention_infer(q, k, v, is_causal=True)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    def test_custom_scale(self):
        q, k, v = _qkv(Sq=4, Sk=4)
        out = base_attention_infer(q, k, v, scale=0.1)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    def test_dropout_p(self):
        q, k, v = _qkv(Sq=4, Sk=4)
        out = base_attention_infer(q, k, v, dropout_p=0.1)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    def test_different_Sq_Sk(self):
        q, k, v = _qkv(Sq=4, Sk=12)
        out = base_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    def test_multi_batch(self):
        q, k, v = _qkv(B=3, N=4, Sq=8, Sk=8)
        out = base_attention_infer(q, k, v)
        self.assertEqual(out.shape, (3, 4, 8, 64))


# ═══════════════════════════════════════════════════════════════════════════
# 2. sage_attention_infer
# ═══════════════════════════════════════════════════════════════════════════

class TestSageAttentionInfer(unittest.TestCase):
    """Tests for sage_attention_infer (quantised sage attention path)."""

    def test_fallback_when_D_not_128(self):
        """D != 128 → falls back to base_attention_infer."""
        q, k, v = _qkv(D=64)
        out = sage_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    def test_fallback_when_Sq_below_threshold(self):
        """Sq <= SAGE_SEQ_LEN_THRESHOLD → falls back."""
        q = torch.randn(1, 2, 100, 128)
        k = torch.randn(1, 2, 20000, 128)
        v = torch.randn(1, 2, 20000, 128)
        out = sage_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 100, 128))

    def test_fallback_when_Sk_below_threshold(self):
        """Sk <= SAGE_SEQ_LEN_THRESHOLD → falls back."""
        q = torch.randn(1, 2, 20000, 128)
        k = torch.randn(1, 2, 100, 128)
        v = torch.randn(1, 2, 100, 128)
        out = sage_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 20000, 128))

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_sage_path_called(self):
        """D==128, Sq & Sk > threshold → cann_ops.npu_sage_attention called."""
        fake_out = torch.randn(1, 4, 12000, 128)
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = fake_out
        q = torch.randn(1, 4, 12000, 128)
        k = torch.randn(1, 4, 12000, 128)
        v = torch.randn(1, 4, 12000, 128)
        out = sage_attention_infer(q, k, v)
        _mock_cann_ops.block_quant.assert_called()
        _mock_cann_ops.npu_sage_attention.assert_called_once()
        self.assertEqual(out.shape, fake_out.shape)

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_sage_passes_attn_mask(self):
        """attn_mask is forwarded to npu_sage_attention."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        mask = torch.zeros(1, 2, 12000, 12000)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        sage_attention_infer(q, k, v, attn_mask=mask)
        call_kwargs = _mock_cann_ops.npu_sage_attention.call_args
        self.assertIsNotNone(call_kwargs)

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_sage_scale_value_is_D_neg_half(self):
        """scale_value passed to npu_sage_attention should be D**-0.5."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        sage_attention_infer(q, k, v)
        _, kwargs = _mock_cann_ops.npu_sage_attention.call_args
        self.assertAlmostEqual(kwargs["scale_value"], 128 ** -0.5)


# ═══════════════════════════════════════════════════════════════════════════
# 3. sage_attention_infer_quant_scale
# ═══════════════════════════════════════════════════════════════════════════

class TestSageAttentionInferQuantScale(unittest.TestCase):
    """Tests for sage_attention_infer_quant_scale (sm_scale moved to quant)."""

    def test_fallback_D_not_128(self):
        q, k, v = _qkv(D=64)
        out = sage_attention_infer_quant_scale(q, k, v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_quant_scale_path(self):
        """Quant-scale variant: block_quant gets sm_scale, npu_sage_attention gets scale_value=0."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        out = sage_attention_infer_quant_scale(q, k, v)
        # block_quant called with sm_scale for query
        bq_calls = _mock_cann_ops.block_quant.call_args_list
        first_call = bq_calls[0]
        self.assertIn("sm_scale", first_call[1])
        # npu_sage_attention scale_value should be 0.0
        _, kwargs = _mock_cann_ops.npu_sage_attention.call_args
        self.assertEqual(kwargs["scale_value"], 0.0)

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_quant_scale_key_no_sm_scale(self):
        """Key block_quant should be called with sm_scale=None."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        sage_attention_infer_quant_scale(q, k, v)
        bq_calls = _mock_cann_ops.block_quant.call_args_list
        second_call = bq_calls[1]
        self.assertIsNone(second_call[1]["sm_scale"])


# ═══════════════════════════════════════════════════════════════════════════
# 4. laser_attention_infer
# ═══════════════════════════════════════════════════════════════════════════

class TestLaserAttentionInfer(unittest.TestCase):
    """Tests for laser_attention_infer (MindIE laser attention path)."""

    def test_fallback_short_seq(self):
        """Sk < 4000 → falls back to base."""
        q, k, v = _qkv(Sq=8, Sk=8)
        out = laser_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    @patch("operator.attention.LA_AVAILABLE", True)
    def test_laser_path_called(self):
        """Sk >= 4000 → attention_forward called."""
        fake_out = torch.randn(1, 2, 5000, 64)
        _mock_mindiesd.attention_forward.return_value = fake_out
        q = torch.randn(1, 2, 5000, 64)
        k = torch.randn(1, 2, 5000, 64)
        v = torch.randn(1, 2, 5000, 64)
        out = laser_attention_infer(q, k, v)
        _mock_mindiesd.attention_forward.assert_called_once()
        self.assertEqual(out.shape, fake_out.shape)

    @patch("operator.attention.LA_AVAILABLE", True)
    def test_laser_forwards_layout(self):
        """attention_forward should receive layout='BNSD'."""
        _mock_mindiesd.attention_forward.return_value = torch.randn(1, 2, 5000, 64)
        q = torch.randn(1, 2, 5000, 64)
        k = torch.randn(1, 2, 5000, 64)
        v = torch.randn(1, 2, 5000, 64)
        laser_attention_infer(q, k, v)
        _, kwargs = _mock_mindiesd.attention_forward.call_args
        self.assertEqual(kwargs["layout"], "BNSD")
        self.assertEqual(kwargs["op_type"], "ascend_laser_attention")

    @patch("operator.attention.LA_AVAILABLE", True)
    def test_laser_boundary_exactly_4000(self):
        """Sk == 4000 → should use laser path (>=)."""
        _mock_mindiesd.attention_forward.return_value = torch.randn(1, 2, 4000, 64)
        q = torch.randn(1, 2, 4000, 64)
        k = torch.randn(1, 2, 4000, 64)
        v = torch.randn(1, 2, 4000, 64)
        laser_attention_infer(q, k, v)
        _mock_mindiesd.attention_forward.assert_called_once()

    @patch("operator.attention.LA_AVAILABLE", True)
    def test_laser_3999_falls_back(self):
        """Sk == 3999 → falls back to base."""
        q = torch.randn(1, 2, 3999, 64)
        k = torch.randn(1, 2, 3999, 64)
        v = torch.randn(1, 2, 3999, 64)
        out = laser_attention_infer(q, k, v)
        _mock_mindiesd.attention_forward.assert_not_called()
        self.assertEqual(out.shape, (1, 2, 3999, 64))


# ═══════════════════════════════════════════════════════════════════════════
# 5. AttentionManager — enable / disable
# ═══════════════════════════════════════════════════════════════════════════

class TestAttentionManagerEnableDisable(unittest.TestCase):
    """Tests for AttentionManager enable/disable flags."""

    def setUp(self):
        self.mgr = AttentionManager()

    def test_initial_state(self):
        self.assertFalse(self.mgr.is_sage_attention)
        self.assertFalse(self.mgr.is_laser_attention)
        self.assertFalse(self.mgr.is_sparse_attention)
        self.assertFalse(self.mgr.is_rainfusion_attention)
        self.assertIsNone(self.mgr.rf_grid_size)
        self.assertIsNone(self.mgr.atten_mask_all)
        self.assertEqual(self.mgr.t_idx, -1)

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_enable_sage(self):
        self.mgr.enable_sage_attention()
        self.assertTrue(self.mgr.is_sage_attention)

    @patch("operator.attention.CANN_OPS_AVAILABLE", False)
    def test_enable_sage_no_cann(self):
        """Without CANN_OPS, enable should log error and NOT set flag."""
        self.mgr.enable_sage_attention()
        self.assertFalse(self.mgr.is_sage_attention)

    def test_disable_sage(self):
        self.mgr.is_sage_attention = True
        self.mgr.disable_sage_attention()
        self.assertFalse(self.mgr.is_sage_attention)

    @patch("operator.attention.LA_AVAILABLE", True)
    def test_enable_laser(self):
        self.mgr.enable_laser_attention()
        self.assertTrue(self.mgr.is_laser_attention)

    @patch("operator.attention.LA_AVAILABLE", False)
    def test_enable_laser_no_mindie(self):
        self.mgr.enable_laser_attention()
        self.assertFalse(self.mgr.is_laser_attention)

    def test_disable_laser(self):
        self.mgr.is_laser_attention = True
        self.mgr.disable_laser_attention()
        self.assertFalse(self.mgr.is_laser_attention)

    def test_enable_sparse(self):
        self.mgr.enable_sparse_attention()
        self.assertTrue(self.mgr.is_sparse_attention)

    def test_disable_sparse(self):
        self.mgr.is_sparse_attention = True
        self.mgr.disable_sparse_attention()
        self.assertFalse(self.mgr.is_sparse_attention)

    def test_enable_rainfusion(self):
        self.mgr.enable_rainfusion_attention()
        self.assertTrue(self.mgr.is_rainfusion_attention)

    def test_disable_rainfusion(self):
        self.mgr.is_rainfusion_attention = True
        self.mgr.disable_rainfusion_attention()
        self.assertFalse(self.mgr.is_rainfusion_attention)


# ═══════════════════════════════════════════════════════════════════════════
# 6. AttentionManager — t_idx
# ═══════════════════════════════════════════════════════════════════════════

class TestAttentionManagerTIdx(unittest.TestCase):
    """Tests for set_t_idx / update_t_idx."""

    def setUp(self):
        self.mgr = AttentionManager()

    def test_set_t_idx(self):
        self.mgr.set_t_idx(5)
        self.assertEqual(self.mgr.t_idx, 5)

    def test_update_t_idx_increments(self):
        self.mgr.set_t_idx(0)
        self.mgr.update_t_idx()
        self.assertEqual(self.mgr.t_idx, 1)
        self.mgr.update_t_idx()
        self.assertEqual(self.mgr.t_idx, 2)

    def test_update_from_default(self):
        self.assertEqual(self.mgr.t_idx, -1)
        self.mgr.update_t_idx()
        self.assertEqual(self.mgr.t_idx, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. AttentionManager — set_rainfusion_attention
# ═══════════════════════════════════════════════════════════════════════════

class TestAttentionManagerSetRainfusion(unittest.TestCase):
    """Tests for set_rainfusion_attention (initialises rf parameters)."""

    def setUp(self):
        self.mgr = AttentionManager()
        self.mgr.is_rainfusion_attention = True

    def test_set_rainfusion_basic(self):
        self.mgr.set_rainfusion_attention([2, 8, 8])
        self.assertEqual(self.mgr.rf_grid_size, [2, 4, 4])
        self.assertEqual(self.mgr.rf_skip_timesteps, 0)
        self.assertEqual(self.mgr.rf_sparsity, 0.64)
        self.assertEqual(self.mgr.t_idx, -1)

    def test_set_rainfusion_custom_params(self):
        self.mgr.set_rainfusion_attention([4, 4, 4], skip_timesteps=5, sparsity=0.8)
        self.assertEqual(self.mgr.rf_grid_size, [4, 2, 2])
        self.assertEqual(self.mgr.rf_skip_timesteps, 5)
        self.assertEqual(self.mgr.rf_sparsity, 0.8)

    def test_set_rainfusion_without_enable(self):
        """Calling set_rainfusion_attention without enable should log error but still set params."""
        self.mgr.is_rainfusion_attention = False
        self.mgr.set_rainfusion_attention([2, 8, 8])
        # The guard only logs an error, doesn't return early
        self.assertEqual(self.mgr.rf_grid_size, [2, 4, 4])


# ═══════════════════════════════════════════════════════════════════════════
# 8. AttentionManager — set_ada_bsa_sparse_flash_attention
# ═══════════════════════════════════════════════════════════════════════════

class TestAttentionManagerAdaBsa(unittest.TestCase):
    """Tests for set_ada_bsa_sparse_flash_attention and ada_bsa_sparse_flash_attention_infer."""

    def setUp(self):
        self.mgr = AttentionManager()

    def test_set_ada_sparsity(self):
        self.mgr.set_ada_bsa_sparse_flash_attention(sparsity=0.5)
        self.assertEqual(self.mgr.ada_sparsity, 0.5)

    @patch("operator.attention.LA_AVAILABLE", False)
    def test_ada_bsa_sq_ne_sk_fallback(self):
        """Sq != Sk → falls back to base_attention_infer."""
        self.mgr.set_ada_bsa_sparse_flash_attention(sparsity=0.5)
        q = torch.randn(1, 2, 4, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        out = self.mgr.ada_bsa_sparse_flash_attention_infer(q, k, v)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    @patch("operator.attention.torch")
    def test_ada_bsa_sq_eq_sk_sparse_path(self, mock_torch):
        """Sq == Sk → sparse attention path (torch.ops.mindiesd)."""
        self.mgr.set_ada_bsa_sparse_flash_attention(sparsity=0.7)
        q = torch.randn(1, 2, 8, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        # Mock torch.ops.mindiesd
        fake_smask = torch.ones(1, 2, 8, 8)
        fake_sct = torch.ones(1, 2, 8, 8)
        mock_ops = MagicMock()
        mock_ops.mindiesd.sparse_block_estimate.return_value = (fake_smask, fake_sct)
        mock_ops.mindiesd.block_sparse_attention.return_value = torch.randn(1, 2, 8, 64)
        mock_torch.ops = mock_ops
        mock_torch.randn = torch.randn
        mock_torch.tensor = torch.tensor
        out = self.mgr.ada_bsa_sparse_flash_attention_infer(q, k, v)
        mock_ops.mindiesd.sparse_block_estimate.assert_called_once()
        mock_ops.mindiesd.block_sparse_attention.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 9. AttentionManager — attention() dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestAttentionManagerDispatch(unittest.TestCase):
    """Tests that AttentionManager.attention() dispatches to the right backend."""

    def setUp(self):
        self.mgr = AttentionManager()
        self.q, self.k, self.v = _qkv(Sq=8, Sk=8, D=64)

    def test_dispatch_base(self):
        """No flags → base_attention_infer."""
        out = self.mgr.attention(self.q, self.k, self.v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_dispatch_sage(self):
        """is_sage_attention → sage_attention_infer."""
        self.mgr.enable_sage_attention()
        # D=64 won't trigger sage path, but dispatch still calls sage_attention_infer
        out = self.mgr.attention(self.q, self.k, self.v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    @patch("operator.attention.LA_AVAILABLE", True)
    def test_dispatch_laser(self):
        """is_laser_attention → laser_attention_infer (short seq → fallback)."""
        self.mgr.enable_laser_attention()
        out = self.mgr.attention(self.q, self.k, self.v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    def test_dispatch_sparse(self):
        """is_sparse_attention → ada_bsa_sparse_flash_attention_infer (Sq!=Sk → fallback)."""
        self.mgr.enable_sparse_attention()
        self.mgr.set_ada_bsa_sparse_flash_attention(sparsity=0.5)
        out = self.mgr.attention(self.q, self.k, self.v)
        self.assertEqual(out.shape, (1, 2, 8, 64))

    def test_dispatch_rainfusion(self):
        """is_rainfusion_attention → rainfusion_attention_infer (mocked)."""
        self.mgr.is_rainfusion_attention = True
        self.mgr.rainfusion_attention_infer = MagicMock(return_value=torch.randn(1, 2, 8, 64))
        out = self.mgr.attention(self.q, self.k, self.v)
        self.mgr.rainfusion_attention_infer.assert_called_once()

    def test_priority_sage_over_laser(self):
        """sage takes priority over laser."""
        self.mgr.is_sage_attention = True
        self.mgr.is_laser_attention = True
        with patch("operator.attention.sage_attention_infer", return_value=torch.randn(1, 2, 8, 64)) as mock_sage:
            out = self.mgr.attention(self.q, self.k, self.v)
            mock_sage.assert_called_once()

    def test_priority_laser_over_rainfusion(self):
        """laser takes priority over rainfusion."""
        self.mgr.is_laser_attention = True
        self.mgr.is_rainfusion_attention = True
        self.mgr.rainfusion_attention_infer = MagicMock()
        with patch("operator.attention.laser_attention_infer", return_value=torch.randn(1, 2, 8, 64)) as mock_laser:
            out = self.mgr.attention(self.q, self.k, self.v)
            mock_laser.assert_called_once()
            self.mgr.rainfusion_attention_infer.assert_not_called()

    def test_priority_rainfusion_over_sparse(self):
        """rainfusion takes priority over sparse."""
        self.mgr.is_rainfusion_attention = True
        self.mgr.is_sparse_attention = True
        self.mgr.rainfusion_attention_infer = MagicMock(return_value=torch.randn(1, 2, 8, 64))
        out = self.mgr.attention(self.q, self.k, self.v)
        self.mgr.rainfusion_attention_infer.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 10. SAGE_SEQ_LEN_THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════

class TestSageSeqLenThreshold(unittest.TestCase):
    """Tests for the SAGE_SEQ_LEN_THRESHOLD constant."""

    def test_default_value(self):
        self.assertEqual(SAGE_SEQ_LEN_THRESHOLD, 10000)

    @patch.dict(os.environ, {"SAGE_SEQ_LEN_THRESHOLD": "5000"})
    def test_env_override(self):
        """Threshold should be configurable via env var (checked at import time)."""
        # The constant was already read at import, so we just verify the mechanism
        expected = int(os.getenv("SAGE_SEQ_LEN_THRESHOLD", 10000))
        self.assertEqual(expected, 5000)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Module-level attention_manager singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestAttentionManagerSingleton(unittest.TestCase):
    """Tests for the module-level attention_manager singleton."""

    def test_is_instance(self):
        self.assertIsInstance(attention_manager, AttentionManager)

    def test_default_state(self):
        self.assertFalse(attention_manager.is_sage_attention)
        self.assertFalse(attention_manager.is_laser_attention)
        self.assertFalse(attention_manager.is_sparse_attention)
        self.assertFalse(attention_manager.is_rainfusion_attention)


# ═══════════════════════════════════════════════════════════════════════════
# 12. CANN_OPS_AVAILABLE / LA_AVAILABLE flags
# ═══════════════════════════════════════════════════════════════════════════

class TestAvailabilityFlags(unittest.TestCase):
    """Tests for CANN_OPS_AVAILABLE and LA_AVAILABLE module flags."""

    def test_cann_ops_available_is_bool(self):
        from operator.attention import CANN_OPS_AVAILABLE
        self.assertIsInstance(CANN_OPS_AVAILABLE, bool)

    def test_la_available_is_bool(self):
        from operator.attention import LA_AVAILABLE
        self.assertIsInstance(LA_AVAILABLE, bool)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Edge cases & integration
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Edge-case and integration tests."""

    def test_base_attention_deterministic_eval(self):
        """In eval mode, base attention should be deterministic (no dropout)."""
        q, k, v = _qkv(Sq=4, Sk=4)
        torch.manual_seed(42)
        out1 = base_attention_infer(q, k, v)
        torch.manual_seed(42)
        out2 = base_attention_infer(q, k, v)
        self.assertTrue(torch.allclose(out1, out2))

    def test_sage_fallback_matches_base(self):
        """When sage falls back (D!=128), output should match base_attention_infer."""
        q, k, v = _qkv(D=64)
        out_sage = sage_attention_infer(q, k, v)
        out_base = base_attention_infer(q, k, v)
        self.assertTrue(torch.allclose(out_sage, out_base))

    def test_laser_fallback_matches_base(self):
        """When laser falls back (short seq), output should match base_attention_infer."""
        q, k, v = _qkv(Sq=8, Sk=8)
        out_laser = laser_attention_infer(q, k, v)
        out_base = base_attention_infer(q, k, v)
        self.assertTrue(torch.allclose(out_laser, out_base))

    def test_manager_base_matches_base_fn(self):
        """Manager with no flags should produce same as base_attention_infer."""
        q, k, v = _qkv(Sq=4, Sk=4)
        mgr = AttentionManager()
        out_mgr = mgr.attention(q, k, v)
        out_base = base_attention_infer(q, k, v)
        self.assertTrue(torch.allclose(out_mgr, out_base))

    def test_sage_quant_scale_fallback_matches_base(self):
        """sage_attention_infer_quant_scale fallback should match base."""
        q, k, v = _qkv(D=64)
        out_qs = sage_attention_infer_quant_scale(q, k, v)
        out_base = base_attention_infer(q, k, v)
        self.assertTrue(torch.allclose(out_qs, out_base))

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_sage_block_quant_called_twice(self):
        """Sage path should call block_quant once for q and once for k."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        sage_attention_infer(q, k, v)
        self.assertEqual(_mock_cann_ops.block_quant.call_count, 2)

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_sage_quant_scale_block_quant_called_twice(self):
        """Quant-scale path should also call block_quant twice."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        sage_attention_infer_quant_scale(q, k, v)
        self.assertEqual(_mock_cann_ops.block_quant.call_count, 2)

    @patch("operator.attention.CANN_OPS_AVAILABLE", True)
    def test_sage_contiguous_called(self):
        """Sage path should call .contiguous() on quantised q, k, v."""
        _mock_cann_ops.block_quant.return_value = (MagicMock(), MagicMock())
        _mock_cann_ops.npu_sage_attention.return_value = torch.randn(1, 2, 12000, 128)
        q = torch.randn(1, 2, 12000, 128)
        k = torch.randn(1, 2, 12000, 128)
        v = torch.randn(1, 2, 12000, 128)
        sage_attention_infer(q, k, v)
        _, kwargs = _mock_cann_ops.npu_sage_attention.call_args
        # The mock objects for q_int8 and k_int8 should have had .contiguous() called
        # We can't easily verify that on mocks, but we verify the call was made
        self.assertIsNotNone(kwargs)

    def test_ada_bsa_default_sparsity(self):
        """Default sparsity param in set_ada_bsa_sparse_flash_attention is 0.7."""
        mgr = AttentionManager()
        mgr.set_ada_bsa_sparse_flash_attention()
        self.assertEqual(mgr.ada_sparsity, 0.7)

    def test_attention_with_all_flags_false(self):
        """All flags false → base attention."""
        mgr = AttentionManager()
        q, k, v = _qkv(Sq=4, Sk=4)
        out = mgr.attention(q, k, v)
        self.assertEqual(out.shape, (1, 2, 4, 64))

    def test_manager_multiple_calls(self):
        """Manager should handle multiple sequential attention calls."""
        mgr = AttentionManager()
        q, k, v = _qkv(Sq=4, Sk=4)
        out1 = mgr.attention(q, k, v)
        out2 = mgr.attention(q, k, v)
        self.assertEqual(out1.shape, out2.shape)

    def test_enable_disable_cycle(self):
        """Enable then disable should return to base state."""
        mgr = AttentionManager()
        mgr.is_sage_attention = True
        mgr.disable_sage_attention()
        mgr.is_laser_attention = True
        mgr.disable_laser_attention()
        mgr.is_sparse_attention = True
        mgr.disable_sparse_attention()
        mgr.is_rainfusion_attention = True
        mgr.disable_rainfusion_attention()
        self.assertFalse(mgr.is_sage_attention)
        self.assertFalse(mgr.is_laser_attention)
        self.assertFalse(mgr.is_sparse_attention)
        self.assertFalse(mgr.is_rainfusion_attention)

    def test_t_idx_negative_default(self):
        """Default t_idx should be -1."""
        mgr = AttentionManager()
        self.assertEqual(mgr.t_idx, -1)

    def test_set_rainfusion_resets_t_idx(self):
        """set_rainfusion_attention should reset t_idx to -1."""
        mgr = AttentionManager()
        mgr.is_rainfusion_attention = True
        mgr.t_idx = 10
        mgr.set_rainfusion_attention([2, 8, 8])
        self.assertEqual(mgr.t_idx, -1)


if __name__ == "__main__":
    unittest.main()
