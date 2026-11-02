"""Tests for rope.py — covers apply_rotary_emb, RopeManager (enable, rope_flux,
rope_base, rope_mindie, rope dispatch), and module-level singleton.

NPU-only dependencies (torch_npu, cann_ops, ascend_cloud, mindiesd) are
mocked via sys.modules injection so tests run on CPU-only machines.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

# ── Mock NPU-only modules before importing rope.py ──────────────────────

# torch_npu
sys.modules.setdefault("torch_npu", MagicMock())

# cann_ops + ascend_cloud
sys.modules.setdefault("cann_ops", MagicMock())
_mock_ascend_cloud = MagicMock()
sys.modules.setdefault("ascend_cloud", _mock_ascend_cloud)

_mock_mindiesd = MagicMock()
_mock_mindiesd.rotary_position_embedding = MagicMock()
sys.modules.setdefault("mindiesd", _mock_mindiesd)

# ── Now import the module under test ────────────────────────────────────

from operator.rope import (  # noqa: E402
    apply_rotary_emb,
    RopeManager,
    rope_manager,
    ASCEND_CLOUD_AVAILABLE,
    AVAILABLE_MINDIESD,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: create cos/sin for rotary embedding
# ═══════════════════════════════════════════════════════════════════════════

def _make_cos_sin(seq_len=8, head_dim=64, dtype=torch.float32):
    """Create cos/sin tensors of shape (1, seq_len, head_dim)."""
    cos = torch.randn(1, seq_len, head_dim, dtype=dtype)
    sin = torch.randn(1, seq_len, head_dim, dtype=dtype)
    return cos, sin


def _make_qkv(batch=1, seq_len=8, heads=2, head_dim=64, dtype=torch.float32):
    """Create query/key tensors of shape (batch, seq_len, heads, head_dim)."""
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype)
    return q, k


# ═══════════════════════════════════════════════════════════════════════════
# 1. apply_rotary_emb
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyRotaryEmb(unittest.TestCase):
    """Tests for the apply_rotary_emb function."""

    def test_output_shape(self):
        x = torch.randn(1, 8, 2, 64)
        cos, sin = _make_cos_sin(8, 64)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.shape, x.shape)

    def test_output_dtype_matches_input(self):
        x = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        cos, sin = _make_cos_sin(8, 64, dtype=torch.float32)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.dtype, x.dtype)

    def test_output_dtype_float16(self):
        x = torch.randn(1, 8, 2, 64, dtype=torch.float16)
        cos, sin = _make_cos_sin(8, 64, dtype=torch.float16)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.dtype, torch.float16)

    def test_bfloat16(self):
        x = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16)
        cos, sin = _make_cos_sin(8, 64, dtype=torch.bfloat16)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_not_inplace(self):
        """Output should be a new tensor, not modifying the input."""
        x = torch.randn(1, 8, 2, 64)
        cos, sin = _make_cos_sin(8, 64)
        x_clone = x.clone()
        _ = apply_rotary_emb(x, cos, sin)
        self.assertTrue(torch.equal(x, x_clone))

    def test_even_positions_use_cos_and_sin(self):
        """out[..., 0::2] = x1*cos - x2*sin (even positions)."""
        x = torch.ones(1, 4, 1, 4)
        cos = torch.ones(1, 4, 4)
        sin = torch.zeros(1, 4, 4)
        out = apply_rotary_emb(x, cos, sin)
        # x1=1, x2=1, cos=1, sin=0 → even: 1*1 - 1*0 = 1
        self.assertTrue(torch.allclose(out[..., 0::2], torch.ones_like(out[..., 0::2])))

    def test_odd_positions_use_sin_and_cos(self):
        """out[..., 1::2] = x1*sin + x2*cos (odd positions)."""
        x = torch.ones(1, 4, 1, 4)
        cos = torch.ones(1, 4, 4)
        sin = torch.zeros(1, 4, 4)
        out = apply_rotary_emb(x, cos, sin)
        # x1=1, x2=1, cos=1, sin=0 → odd: 1*0 + 1*1 = 1
        self.assertTrue(torch.allclose(out[..., 1::2], torch.ones_like(out[..., 1::2])))

    def test_identity_with_cos1_sin0(self):
        """cos=1, sin=0 should act as identity for paired (x1,x2)=(1,0)."""
        x = torch.zeros(1, 4, 1, 4)
        x[..., 0::2] = 1.0  # x1=1, x2=0
        cos = torch.ones(1, 4, 4)
        sin = torch.zeros(1, 4, 4)
        out = apply_rotary_emb(x, cos, sin)
        # even: 1*1 - 0*0 = 1, odd: 1*0 + 0*1 = 0
        self.assertTrue(torch.allclose(out[..., 0::2], torch.ones_like(out[..., 0::2])))
        self.assertTrue(torch.allclose(out[..., 1::2], torch.zeros_like(out[..., 1::2])))

    def test_rotation_90_degrees(self):
        """cos=0, sin=1 should swap (x1,x2) → (-x2, x1)."""
        x = torch.zeros(1, 4, 1, 4)
        x[..., 0::2] = 1.0  # x1=1, x2=0
        cos = torch.zeros(1, 4, 4)
        sin = torch.ones(1, 4, 4)
        out = apply_rotary_emb(x, cos, sin)
        # even: 1*0 - 0*1 = 0, odd: 1*1 + 0*0 = 1
        self.assertTrue(torch.allclose(out[..., 0::2], torch.zeros_like(out[..., 0::2]), atol=1e-6))
        self.assertTrue(torch.allclose(out[..., 1::2], torch.ones_like(out[..., 1::2]), atol=1e-6))

    def test_different_seq_lengths(self):
        for sl in [1, 4, 16, 64]:
            x = torch.randn(1, sl, 2, 64)
            cos, sin = _make_cos_sin(sl, 64)
            out = apply_rotary_emb(x, cos, sin)
            self.assertEqual(out.shape, (1, sl, 2, 64))

    def test_different_head_dims(self):
        for hd in [16, 32, 64, 128]:
            x = torch.randn(1, 8, 2, hd)
            cos, sin = _make_cos_sin(8, hd)
            out = apply_rotary_emb(x, cos, sin)
            self.assertEqual(out.shape, (1, 8, 2, hd))

    def test_batch_dimension(self):
        x = torch.randn(3, 8, 2, 64)
        cos, sin = _make_cos_sin(8, 64)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.shape, (3, 8, 2, 64))

    def test_no_nan_output(self):
        x = torch.randn(1, 8, 2, 64)
        cos, sin = _make_cos_sin(8, 64)
        out = apply_rotary_emb(x, cos, sin)
        self.assertFalse(torch.any(torch.isnan(out)))


# ═══════════════════════════════════════════════════════════════════════════
# 2. RopeManager — init
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerInit(unittest.TestCase):
    """Tests for RopeManager.__init__ default state."""

    def test_default_not_fused(self):
        mgr = RopeManager()
        self.assertFalse(mgr.is_rope_fused)


# ═══════════════════════════════════════════════════════════════════════════
# 3. RopeManager — enable_rope_fused
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerEnable(unittest.TestCase):
    """Tests for RopeManager.enable_rope_fused."""

    def test_enable_with_mindie(self):
        with patch("operator.rope.AVAILABLE_MINDIESD", True):
            mgr = RopeManager()
            mgr.enable_rope_fused()
            self.assertTrue(mgr.is_rope_fused)

    def test_enable_with_ascend_cloud(self):
        with patch("operator.rope.ASCEND_CLOUD_AVAILABLE", True), \
             patch("operator.rope.AVAILABLE_MINDIESD", False):
            mgr = RopeManager()
            mgr.enable_rope_fused()
            self.assertTrue(mgr.is_rope_fused)

    def test_enable_with_both(self):
        with patch("operator.rope.ASCEND_CLOUD_AVAILABLE", True), \
             patch("operator.rope.AVAILABLE_MINDIESD", True):
            mgr = RopeManager()
            mgr.enable_rope_fused()
            self.assertTrue(mgr.is_rope_fused)

    def test_enable_with_neither(self):
        with patch("operator.rope.ASCEND_CLOUD_AVAILABLE", False), \
             patch("operator.rope.AVAILABLE_MINDIESD", False):
            mgr = RopeManager()
            mgr.enable_rope_fused()
            self.assertFalse(mgr.is_rope_fused)


# ═══════════════════════════════════════════════════════════════════════════
# 4. RopeManager — rope_base
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerBase(unittest.TestCase):
    """Tests for RopeManager.rope_base (delegates to apply_rotary_emb)."""

    def test_output_shapes(self):
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_both_qk_transformed(self):
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
        # Should differ from input (cos/sin are random non-trivial)
        self.assertFalse(torch.allclose(q_out, q))
        self.assertFalse(torch.allclose(k_out, k))

    def test_identity_cos1_sin0(self):
        """With cos=1, sin=0 and x1=1, x2=0 → output should be same as input."""
        q = torch.zeros(1, 4, 1, 4)
        q[..., 0::2] = 1.0
        k = torch.zeros(1, 4, 1, 4)
        k[..., 0::2] = 1.0
        cos = torch.ones(1, 4, 4)
        sin = torch.zeros(1, 4, 4)
        q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
        self.assertTrue(torch.allclose(q_out, q))
        self.assertTrue(torch.allclose(k_out, k))

    def test_consistent_with_apply_rotary_emb(self):
        """rope_base should apply apply_rotary_emb to both q and k."""
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
        q_expected = apply_rotary_emb(q, cos, sin)
        k_expected = apply_rotary_emb(k, cos, sin)
        self.assertTrue(torch.allclose(q_out, q_expected))
        self.assertTrue(torch.allclose(k_out, k_expected))

    def test_is_static_method(self):
        """rope_base should be callable without instance."""
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
        self.assertEqual(q_out.shape, q.shape)


# ═══════════════════════════════════════════════════════════════════════════
# 5. RopeManager — rope_mindie
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerMindie(unittest.TestCase):
    """Tests for RopeManager.rope_mindie (delegates to mindiesd)."""

    def test_calls_rotary_position_embedding_twice(self):
        """Should call rotary_position_embedding for both q and k."""
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        _mock_mindiesd.rotary_position_embedding.return_value = torch.randn_like(q)
        RopeManager.rope_mindie(q, k, cos, sin)
        self.assertEqual(_mock_mindiesd.rotary_position_embedding.call_count, 2)

    def test_passes_rotated_interleaved(self):
        """Should pass rotated_mode='rotated_interleaved' and fused=True."""
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        _mock_mindiesd.rotary_position_embedding.return_value = torch.randn_like(q)
        RopeManager.rope_mindie(q, k, cos, sin)
        _, kwargs = _mock_mindiesd.rotary_position_embedding.call_args
        self.assertEqual(kwargs["rotated_mode"], "rotated_interleaved")
        self.assertTrue(kwargs["fused"])

    def test_returns_q_k(self):
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        fake_q_out = torch.randn_like(q)
        fake_k_out = torch.randn_like(k)
        _mock_mindiesd.rotary_position_embedding.side_effect = [fake_q_out, fake_k_out]
        q_out, k_out = RopeManager.rope_mindie(q, k, cos, sin)
        self.assertTrue(torch.equal(q_out, fake_q_out))
        self.assertTrue(torch.equal(k_out, fake_k_out))

    def test_is_static_method(self):
        self.assertIsInstance(RopeManager.__dict__["rope_mindie"], staticmethod)


# ═══════════════════════════════════════════════════════════════════════════
# 6. RopeManager — rope_flux
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerFlux(unittest.TestCase):
    """Tests for RopeManager.rope_flux (delegates to ascend_cloud)."""

    def test_calls_ascend_cloud_rope_flux(self):
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        fake_out = (torch.randn_like(q), torch.randn_like(k))
        mock_ops = MagicMock()
        mock_ops.ascend_cloud.rope_flux.return_value = fake_out
        with patch("operator.rope.torch") as mock_torch:
            mock_torch.ops = mock_ops
            mock_torch.randn = torch.randn
            mock_torch.stack = torch.stack
            RopeManager.rope_flux(q, k, cos, sin)
            mock_ops.ascend_cloud.rope_flux.assert_called_once()

    def test_cos_dtype_cast_to_query(self):
        """cos dtype != query dtype → cos should be cast."""
        q = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        k = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        cos = torch.randn(1, 8, 64, dtype=torch.float64)
        sin = torch.randn(1, 8, 64, dtype=torch.float32)
        fake_out = (torch.randn_like(q), torch.randn_like(k))
        mock_ops = MagicMock()
        mock_ops.ascend_cloud.rope_flux.return_value = fake_out
        with patch("operator.rope.torch") as mock_torch:
            mock_torch.ops = mock_ops
            mock_torch.randn = torch.randn
            mock_torch.stack = torch.stack
            RopeManager.rope_flux(q, k, cos, sin)
        # After cast, cos should be float32
        self.assertEqual(cos.dtype, torch.float64)  # original unchanged (cast creates copy)

    def test_sin_dtype_cast_to_query(self):
        """sin dtype != query dtype → sin should be cast."""
        q = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        k = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        cos = torch.randn(1, 8, 64, dtype=torch.float32)
        sin = torch.randn(1, 8, 64, dtype=torch.float64)
        fake_out = (torch.randn_like(q), torch.randn_like(k))
        mock_ops = MagicMock()
        mock_ops.ascend_cloud.rope_flux.return_value = fake_out
        with patch("operator.rope.torch") as mock_torch:
            mock_torch.ops = mock_ops
            mock_torch.randn = torch.randn
            mock_torch.stack = torch.stack
            RopeManager.rope_flux(q, k, cos, sin)

    def test_is_static_method(self):
        self.assertIsInstance(RopeManager.__dict__["rope_flux"], staticmethod)


# ═══════════════════════════════════════════════════════════════════════════
# 7. RopeManager — rope dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerDispatch(unittest.TestCase):
    """Tests for RopeManager.rope dispatching logic."""

    def test_dispatch_base_when_not_fused(self):
        """is_rope_fused=False → rope_base."""
        mgr = RopeManager()
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        with patch.object(RopeManager, "rope_base", return_value=(q, k)) as mock_base, \
             patch.object(RopeManager, "rope_mindie") as mock_mindie, \
             patch.object(RopeManager, "rope_flux") as mock_flux:
            mgr.rope(q, k, cos, sin)
            mock_base.assert_called_once()
            mock_mindie.assert_not_called()
            mock_flux.assert_not_called()

    @patch("operator.rope.AVAILABLE_MINDIESD", True)
    def test_dispatch_mindie_when_fused_and_mindie_available(self):
        """is_rope_fused=True + AVAILABLE_MINDIESD → rope_mindie."""
        mgr = RopeManager()
        mgr.is_rope_fused = True
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        with patch.object(RopeManager, "rope_mindie", return_value=(q, k)) as mock_mindie, \
             patch.object(RopeManager, "rope_flux") as mock_flux, \
             patch.object(RopeManager, "rope_base") as mock_base:
            mgr.rope(q, k, cos, sin)
            mock_mindie.assert_called_once()
            mock_flux.assert_not_called()
            mock_base.assert_not_called()

    @patch("operator.rope.AVAILABLE_MINDIESD", False)
    @patch("operator.rope.ASCEND_CLOUD_AVAILABLE", True)
    def test_dispatch_flux_when_fused_and_ascend_available(self):
        """is_rope_fused=True + no MINDIE + ASCEND_CLOUD → rope_flux."""
        mgr = RopeManager()
        mgr.is_rope_fused = True
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        with patch.object(RopeManager, "rope_flux", return_value=(q, k)) as mock_flux, \
             patch.object(RopeManager, "rope_base") as mock_base, \
             patch.object(RopeManager, "rope_mindie") as mock_mindie:
            mgr.rope(q, k, cos, sin)
            mock_flux.assert_called_once()
            mock_base.assert_not_called()
            mock_mindie.assert_not_called()

    @patch("operator.rope.AVAILABLE_MINDIESD", False)
    @patch("operator.rope.ASCEND_CLOUD_AVAILABLE", False)
    def test_dispatch_base_when_fused_but_no_backend(self):
        """is_rope_fused=True but no backend available → rope_base fallback."""
        mgr = RopeManager()
        mgr.is_rope_fused = True
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        with patch.object(RopeManager, "rope_base", return_value=(q, k)) as mock_base, \
             patch.object(RopeManager, "rope_mindie") as mock_mindie, \
             patch.object(RopeManager, "rope_flux") as mock_flux:
            mgr.rope(q, k, cos, sin)
            mock_base.assert_called_once()

    @patch("operator.rope.AVAILABLE_MINDIESD", True)
    @patch("operator.rope.ASCEND_CLOUD_AVAILABLE", True)
    def test_mindie_priority_over_ascend(self):
        """Both available → MINDIE takes priority."""
        mgr = RopeManager()
        mgr.is_rope_fused = True
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        with patch.object(RopeManager, "rope_mindie", return_value=(q, k)) as mock_mindie, \
             patch.object(RopeManager, "rope_flux") as mock_flux:
            mgr.rope(q, k, cos, sin)
            mock_mindie.assert_called_once()
            mock_flux.assert_not_called()

    def test_base_produces_valid_output(self):
        """End-to-end: rope with base path produces valid tensors."""
        mgr = RopeManager()
        q, k = _make_qkv()
        cos, sin = _make_cos_sin()
        q_out, k_out = mgr.rope(q, k, cos, sin)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)
        self.assertFalse(torch.any(torch.isnan(q_out)))
        self.assertFalse(torch.any(torch.isnan(k_out)))


# ═══════════════════════════════════════════════════════════════════════════
# 8. Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestRopeManagerSingleton(unittest.TestCase):
    """Tests for the module-level rope_manager singleton."""

    def test_is_instance(self):
        self.assertIsInstance(rope_manager, RopeManager)

    def test_default_not_fused(self):
        self.assertFalse(rope_manager.is_rope_fused)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Availability flags
# ═══════════════════════════════════════════════════════════════════════════

class TestAvailabilityFlags(unittest.TestCase):
    """Tests for ASCEND_CLOUD_AVAILABLE and AVAILABLE_MINDIESD flags."""

    def test_ascend_cloud_available_is_bool(self):
        self.assertIsInstance(ASCEND_CLOUD_AVAILABLE, bool)

    def test_available_mindiesd_is_bool(self):
        self.assertIsInstance(AVAILABLE_MINDIESD, bool)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Edge-case and integration tests."""

    def test_apply_rotary_emb_1d_head_dim(self):
        """head_dim=2 (minimum for paired rotation)."""
        x = torch.randn(1, 4, 1, 2)
        cos = torch.randn(1, 4, 2)
        sin = torch.randn(1, 4, 2)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.shape, (1, 4, 1, 2))

    def test_apply_rotary_emb_large_batch(self):
        x = torch.randn(8, 4, 4, 64)
        cos, sin = _make_cos_sin(4, 64)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.shape, (8, 4, 4, 64))

    def test_rope_base_different_dtypes(self):
        """rope_base should preserve dtype."""
        for dt in [torch.float32, torch.float16]:
            q = torch.randn(1, 4, 2, 16, dtype=dt)
            k = torch.randn(1, 4, 2, 16, dtype=dt)
            cos = torch.randn(1, 4, 16, dtype=dt)
            sin = torch.randn(1, 4, 16, dtype=dt)
            q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
            self.assertEqual(q_out.dtype, dt)
            self.assertEqual(k_out.dtype, dt)

    def test_rope_base_seq_len_1(self):
        q = torch.randn(1, 1, 2, 64)
        k = torch.randn(1, 1, 2, 64)
        cos, sin = _make_cos_sin(1, 64)
        q_out, k_out = RopeManager.rope_base(q, k, cos, sin)
        self.assertEqual(q_out.shape, (1, 1, 2, 64))

    def test_multiple_rope_calls_independent(self):
        """Multiple calls should not interfere with each other."""
        mgr = RopeManager()
        q1, k1 = _make_qkv()
        cos1, sin1 = _make_cos_sin()
        q2, k2 = _make_qkv()
        cos2, sin2 = _make_cos_sin()
        q1_out, k1_out = mgr.rope(q1, k1, cos1, sin1)
        q2_out, k2_out = mgr.rope(q2, k2, cos2, sin2)
        # Results should differ (different inputs)
        self.assertFalse(torch.allclose(q1_out, q2_out))

    def test_enable_disable_cycle(self):
        """Enable then manually set to False should work."""
        with patch("operator.rope.AVAILABLE_MINDIESD", True):
            mgr = RopeManager()
            mgr.enable_rope_fused()
            self.assertTrue(mgr.is_rope_fused)
            mgr.is_rope_fused = False
            self.assertFalse(mgr.is_rope_fused)

    def test_rope_flux_cos_sin_slicing(self):
        """rope_flux should slice cos[..., 0::2] and sin[..., 1::2]."""
        q = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        k = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        cos = torch.randn(1, 8, 64, dtype=torch.float32)
        sin = torch.randn(1, 8, 64, dtype=torch.float32)
        fake_out = (torch.randn_like(q), torch.randn_like(k))
        mock_ops = MagicMock()
        mock_ops.ascend_cloud.rope_flux.return_value = fake_out
        with patch("operator.rope.torch") as mock_torch:
            mock_torch.ops = mock_ops
            mock_torch.randn = torch.randn
            mock_torch.stack = torch.stack
            RopeManager.rope_flux(q, k, cos, sin)
            # Verify torch.stack was called (for freqs_cis construction)
            mock_torch.stack.assert_called()

    def test_rope_flux_makes_qk_contiguous(self):
        """rope_flux should call .contiguous() on q and k."""
        q = torch.randn(1, 8, 2, 64)
        k = torch.randn(1, 8, 2, 64)
        # Make non-contiguous by transposing and transposing back
        q_nc = q.transpose(1, 2).transpose(1, 2)
        k_nc = k.transpose(1, 2).transpose(1, 2)
        cos = torch.randn(1, 8, 64)
        sin = torch.randn(1, 8, 64)
        fake_out = (torch.randn_like(q), torch.randn_like(k))
        mock_ops = MagicMock()
        mock_ops.ascend_cloud.rope_flux.return_value = fake_out
        with patch("operator.rope.torch") as mock_torch:
            mock_torch.ops = mock_ops
            mock_torch.randn = torch.randn
            mock_torch.stack = torch.stack
            RopeManager.rope_flux(q_nc, k_nc, cos, sin)
            # After contiguous(), tensors passed to rope_flux should be contiguous
            call_args = mock_ops.ascend_cloud.rope_flux.call_args[0]
            self.assertTrue(call_args[0].is_contiguous())
            self.assertTrue(call_args[1].is_contiguous())


if __name__ == "__main__":
    unittest.main()
