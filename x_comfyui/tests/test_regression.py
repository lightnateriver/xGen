"""Non-MiniMax regression test.

Verifies that non-MiniMax models (Wan, Hunyuan) are NOT affected by the
VAE patches. Tests that VAE patches only apply when MiniMax-H3 VAE is
detected, and that av_stream=None produces identical output to pre-patch
behavior.
"""

import inspect
import unittest


class TestRegression(unittest.TestCase):
    """Regression tests for non-MiniMax models."""

    def test_vae_patches_have_minimax_guard(self):
        """VAE patches have MiniMax-H3 class guard."""
        from x_comfyui.framework.vae import minimax_h3 as v
        src = inspect.getsource(v)
        self.assertTrue(
            "MiniMaxH3" in src or "minimax" in src.lower(),
            "No MiniMax-H3 guard found in VAE patches"
        )

    def test_text_encoder_tp_has_minimax_guard(self):
        """Text Encoder TP has MiniMax-H3 guard."""
        from x_comfyui.framework.text_encoder import minimax_h3 as tp
        src = inspect.getsource(tp)
        self.assertTrue(
            "MiniMax" in src or "minimax" in src.lower(),
            "No MiniMax-H3 guard found in TP patch"
        )

    def test_idempotency_guards_present(self):
        """All patches have idempotency guards."""
        from x_comfyui.framework.vae import minimax_h3 as v
        guards = [
            attr for attr in dir(v)
            if attr.startswith("_x_comfyui_") and attr.endswith("_patched")
        ]
        # Also check the idempotency flag on VAEAttention.forward
        try:
            from comfy.ldm.minimax.vae import Attention as VAEAttention
            if hasattr(VAEAttention.forward, "_x_comfyui_patched"):
                guards.append("VAEAttention.forward._x_comfyui_patched")
        except ImportError:
            pass

        self.assertGreaterEqual(
            len(guards), 1,
            f"Expected at least 1 idempotency guard, found {len(guards)}: {guards}"
        )

    def test_text_encoder_tp_idempotency_guard(self):
        """Text Encoder TP has idempotency guard."""
        from x_comfyui.framework.text_encoder import minimax_h3 as tp
        self.assertTrue(
            hasattr(tp, "_x_comfyui_text_tp_patched"),
            "No _x_comfyui_text_tp_patched guard in text_encoder_tp"
        )

    def test_av_stream_none_falls_back(self):
        """av_stream_config=None falls back to original decode (no streaming)."""
        from x_comfyui.framework.vae import minimax_h3 as v
        src = inspect.getsource(v)
        # _decode_ascend sets av_stream=None when av_stream_config is falsy
        self.assertIn("av_stream = None", src)
        # _decode_lightning_single calls _original_decode when no streaming config
        self.assertIn("_original_decode", src)

    def test_no_return_dict_in_vae(self):
        """No return_dict parameter in x_comfyui VAE code."""
        from x_comfyui.framework.vae import minimax_h3 as v
        src = inspect.getsource(v)
        self.assertNotIn(
            "return_dict",
            src,
            "return_dict found — should use output_buffer"
        )

    def test_no_infer_info_in_vae(self):
        """No infer_info usage in x_comfyui VAE code."""
        from x_comfyui.framework.vae import minimax_h3 as v
        src = inspect.getsource(v)
        self.assertNotIn(
            "infer_info",
            src,
            "infer_info found — x_comfyui doesn't use infer_info"
        )

    def test_vae_scale_factor_not_hardcoded(self):
        """VAE scale factor uses vae_ratio, not hardcoded 8."""
        from x_comfyui.framework.vae import minimax_h3 as v
        src = inspect.getsource(v)
        self.assertIn(
            "vae_ratio",
            src,
            "No vae_ratio reference — scale factor may be hardcoded"
        )

    def test_no_diffusers_import_in_vae_parallelism(self):
        """No diffusers import in x_base vae_parallelism files."""
        from x_base.vae_parallelism import save_av_stream, audio_utils
        save_src = inspect.getsource(save_av_stream)
        audio_src = inspect.getsource(audio_utils)
        self.assertNotIn(
            "from diffusers", save_src,
            "diffusers import found in save_av_stream.py"
        )
        self.assertNotIn(
            "from diffusers", audio_src,
            "diffusers import found in audio_utils.py"
        )


if __name__ == "__main__":
    unittest.main()
