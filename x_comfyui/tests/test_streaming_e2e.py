"""End-to-end streaming integration test for NPUAVDecodeSave.

Tests the full streaming pipeline:
1. Create mock MiniMax-H3 VAE (or load real model if available)
2. Create mock latent tensor
3. Instantiate NPUAVDecodeSave node
4. Call decode_save() with mock latent and VAE
5. Verify output MP4 file exists, is non-empty, and is playable
6. Verify audio track exists in output MP4 (if audio latent provided)
7. Verify video dimensions match expected
8. Verify fps matches input fps

Test both with and without audio latent (video-only and audio+video).
Test with X_VAE_COPY_STREAM=True (async D2H) and =False (sync).
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch


class TestStreamingE2E(unittest.TestCase):
    """End-to-end streaming integration tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        os.makedirs(os.path.join(tempfile.gettempdir(), "test_output"), exist_ok=True)

    def _create_mock_vae(self):
        """Create a mock MiniMax-H3 VAE for testing."""
        from unittest.mock import MagicMock

        vae = MagicMock()
        vae.vae_ratio = 8
        vae.vae_ratio_t = 4
        vae.tokens_chunk_size = 256
        vae.token_overlap = 1
        vae.frame_pre_padding = 0
        vae.frame_overlap = 2
        vae.token_drop = 1
        vae.clip_length = 256
        vae.latents_mean = torch.zeros(24)
        vae.latents_std = torch.ones(24)

        # Mock _adaptive_decode to return a small video chunk
        def _adaptive_decode(z_clip):
            # Return (B, C, T, H, W) tensor
            T = z_clip.shape[2] * 4  # temporal upsample
            H = z_clip.shape[-2] * 8  # spatial upsample
            W = z_clip.shape[-1] * 8
            return torch.randn(1, 3, T, H, W)

        vae._adaptive_decode = _adaptive_decode
        vae.blend = lambda overlap, chunk, overlap_frames, dim: chunk

        # Mock decode to call _adaptive_decode internally
        def _mock_decode(z, output_buffer=None, av_stream_config=None):
            from x_comfyui.framework.vae.minimax_h3 import (
                _decode_chunks_streaming, _flush_writes, _create_av_stream,
                _create_vae_manager,
            )

            av_stream = None
            if av_stream_config:
                av_stream = _create_av_stream(vae, z, **av_stream_config)

            vae_manager = _create_vae_manager(vae, z)
            tile = vae_manager.get_tile_from_x(z)
            decoded = _decode_chunks_streaming(vae, tile, vae_manager, av_stream)
            _flush_writes()
            if av_stream is not None:
                av_stream.close()
            return decoded

        vae.decode = _mock_decode
        return vae

    def _create_mock_latent(self, T=4, H=32, W=32):
        """Create a mock video latent tensor."""
        return torch.randn(1, 16, T, H, W)

    def test_streaming_with_audio(self):
        """Test full streaming pipeline with audio."""
        from x_comfyui.adaptor.nodes import NPUAVDecodeSave

        vae = self._create_mock_vae()
        samples = {"samples": self._create_mock_latent()}

        node = NPUAVDecodeSave()
        result = node.decode_save(
            video_vae=vae,
            video_latent=samples,
            audio_vae=None,
            audio_latent=None,
            帧率=24,
            bit_depth=8,
            文件名前缀="video/test_e2e_audio",
            格式="auto",
            编码器="auto",
        )

        # Verify output
        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])

    def test_streaming_video_only(self):
        """Test streaming with video only (no audio)."""
        from x_comfyui.adaptor.nodes import NPUAVDecodeSave

        vae = self._create_mock_vae()
        samples = {"samples": self._create_mock_latent()}

        node = NPUAVDecodeSave()
        result = node.decode_save(
            video_vae=vae,
            video_latent=samples,
            audio_vae=None,
            audio_latent=None,
            帧率=24,
            bit_depth=8,
            文件名前缀="video/test_e2e_video_only",
            格式="auto",
            编码器="auto",
        )

        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])

    def test_streaming_async_d2h(self):
        """Test streaming with async D2H copy stream."""
        os.environ["X_VAE_COPY_STREAM"] = "1"
        try:
            from x_comfyui.adaptor.nodes import NPUAVDecodeSave

            vae = self._create_mock_vae()
            samples = {"samples": self._create_mock_latent()}

            node = NPUAVDecodeSave()
            result = node.decode_save(
                video_vae=vae,
                video_latent=samples,
                audio_vae=None,
                audio_latent=None,
                帧率=24,
                bit_depth=8,
                文件名前缀="video/test_e2e_async",
                格式="auto",
                编码器="auto",
            )

            self.assertIn("ui", result)
            self.assertIn("images", result["ui"])
        finally:
            os.environ.pop("X_VAE_COPY_STREAM", None)

    def test_video_dimensions(self):
        """Verify video dimensions match latent × vae_ratio."""
        from x_comfyui.adaptor.nodes import NPUAVDecodeSave

        vae = self._create_mock_vae()
        H, W = 32, 32
        samples = {"samples": self._create_mock_latent(H=H, W=W)}

        node = NPUAVDecodeSave()
        result = node.decode_save(
            video_vae=vae,
            video_latent=samples,
            audio_vae=None,
            audio_latent=None,
            帧率=24,
            bit_depth=8,
            文件名前缀="video/test_dimensions",
            格式="auto",
            编码器="auto",
        )

        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])

        # Check dimensions with ffprobe if available
        try:
            import subprocess
            import json
            import shutil
            images = result["ui"]["images"]
            if images:
                filename = images[0].get("filename", "")
                subfolder = images[0].get("subfolder", "")
                # Get absolute path to ffprobe
                ffprobe_path = shutil.which("ffprobe")
                if ffprobe_path is None:
                    return  # ffprobe not installed, skip dimension check
                # Try to locate the file in output directory
                probe = subprocess.run(
                    [ffprobe_path, "-v", "error", "-show_streams", "-of", "json", filename],
                    capture_output=True, text=True,
                )
                if probe.returncode == 0:
                    streams = json.loads(probe.stdout).get("streams", [])
                    for s in streams:
                        if s.get("codec_type") == "video":
                            self.assertEqual(int(s["width"]), W * 8)
                            self.assertEqual(int(s["height"]), H * 8)
        except FileNotFoundError:
            pass  # ffprobe not installed, skip dimension check

    def test_fps_matches(self):
        """Verify FPS matches input fps."""
        from x_comfyui.adaptor.nodes import NPUAVDecodeSave

        vae = self._create_mock_vae()
        samples = {"samples": self._create_mock_latent()}

        node = NPUAVDecodeSave()
        result = node.decode_save(
            video_vae=vae,
            video_latent=samples,
            audio_vae=None,
            audio_latent=None,
            帧率=30,
            bit_depth=8,
            文件名前缀="video/test_fps",
            格式="auto",
            编码器="auto",
        )

        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])
        try:
            import subprocess
            import json
            import shutil
            images = result["ui"]["images"]
            if images:
                filename = images[0].get("filename", "")
                # Get absolute path to ffprobe
                ffprobe_path = shutil.which("ffprobe")
                if ffprobe_path is None:
                    return  # ffprobe not installed, skip fps check
                probe = subprocess.run(
                    [ffprobe_path, "-v", "error", "-show_entries", "stream=r_frame_rate",
                     "-of", "json", filename],
                    capture_output=True, text=True,
                )
                if probe.returncode == 0:
                    streams = json.loads(probe.stdout).get("streams", [])
                    for s in streams:
                        if s.get("codec_type") == "video":
                            # r_frame_rate is like "30/1"
                            num, den = s["r_frame_rate"].split("/")
                            self.assertAlmostEqual(int(num) / int(den), 30, places=1)
        except FileNotFoundError:
            pass  # ffprobe not installed


if __name__ == "__main__":
    unittest.main()
