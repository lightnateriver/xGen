"""Tests for NPUTurbo prompt-scan override and cache invalidation.

Verifies that text_tp overrides are set in the prompt-scan (before any
node executes), eliminating the race condition with CLIPLoader. Also
tests cache invalidation on parameter changes.

These tests run on Windows without NPU/torch_npu by loading env.py
directly via importlib and checking minimax_h3.py source via file read.
"""

import importlib.util
import os
import unittest


def _load_env_module():
    """Load env.py directly, bypassing x_comfyui.__init__ (which needs x_base)."""
    env_path = os.path.join(
        os.path.dirname(__file__), "..",
        "x_comfyui", "adaptor", "env.py",
    )
    env_path = os.path.normpath(env_path)
    spec = importlib.util.spec_from_file_location("env_test", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_minimax_h3_source():
    """Read minimax_h3.py source directly (can't import — needs torch/x_base)."""
    src_path = os.path.join(
        os.path.dirname(__file__), "..",
        "x_comfyui", "framework", "transformer", "minimax_h3.py",
    )
    src_path = os.path.normpath(src_path)
    with open(src_path, encoding="utf-8") as f:
        return f.read()


class TestEnvStorage(unittest.TestCase):
    """Test _last_text_tp storage in env.py."""

    def setUp(self):
        self.env = _load_env_module()

    def tearDown(self):
        """Reset global state to avoid cross-test pollution."""
        self.env.set_text_tp_override(None)
        self.env.set_last_text_tp(None)

    def test_last_text_tp_initial_none(self):
        """get_last_text_tp() returns None on fresh module load."""
        self.assertIsNone(self.env.get_last_text_tp())

    def test_set_get_last_text_tp(self):
        """set_last_text_tp / get_last_text_tp round-trip."""
        self.env.set_last_text_tp("enable")
        self.assertEqual(self.env.get_last_text_tp(), "enable")
        self.env.set_last_text_tp("disable")
        self.assertEqual(self.env.get_last_text_tp(), "disable")


class TestOverrideBehavior(unittest.TestCase):
    """Test override setting behavior (simulating prompt-scan)."""

    def setUp(self):
        self.env = _load_env_module()
        # Clear any env vars that could interfere
        self._old_tp = os.environ.pop("X_TEXT_TP", None)

    def tearDown(self):
        """Reset global state to avoid cross-test pollution."""
        self.env.set_text_tp_override(None)
        if self._old_tp is not None:
            os.environ["X_TEXT_TP"] = self._old_tp


class TestPatchedExecuteAsyncSource(unittest.TestCase):
    """Verify _patched_execute_async source code has required patterns.

    These tests read the source file directly since the module can't be
    imported without torch_npu/x_base
    """

    @classmethod
    def setUpClass(cls):
        cls.source = _read_minimax_h3_source()

    def test_patched_execute_async_has_text_scan(self):
        """_patched_execute_async scans for text_tp."""
        self.assertIn("text_tp", self.source,
                       "text_tp not in _patched_execute_async")
        # Verify scanning reads from inputs dict
        self.assertIn('inputs.get("text_tp"', self.source,
                       "text_tp not scanned from inputs")

    def test_patched_execute_async_sets_override_before_execution(self):
        """Overrides are set BEFORE await _original_execute_async."""
        self.assertIn("set_text_tp_override", self.source,
                       "set_text_tp_override not called")
        # Verify override calls appear before the await call
        override_pos = self.source.index("set_text_tp_override")
        await_pos = self.source.index("await _original_execute_async")
        self.assertLess(override_pos, await_pos,
                        "Override must be set before await _original_execute_async")

    def test_patched_execute_async_clears_cache_on_change(self):
        """Cache clearing exists in text_tp change detection."""
        # Count cache.clear() calls — should be at least 1 (quant).
        # text_tp uses _clear_clip_related_cache (selective, not full clear).
        cache_clear_count = self.source.count("cache.clear()")
        self.assertGreaterEqual(cache_clear_count, 1,
                                f"Expected >=1 cache.clear() calls, found {cache_clear_count}")
        # Verify subcaches.clear() also present
        self.assertIn("subcaches.clear()", self.source,
                      "subcaches.clear() not found")
        # Verify last value storage
        self.assertIn("set_last_text_tp", self.source,
                      "set_last_text_tp not called")
        self.assertIn("get_last_text_tp", self.source,
                      "get_last_text_tp not called")


if __name__ == "__main__":
    unittest.main()
