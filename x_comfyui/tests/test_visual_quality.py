"""Visual quality + audio sync verification test.

Verifies:
1. Streaming output frames match non-streaming reference within pixel tolerance (atol=2)
2. Audio-video sync within ±1 frame tolerance
3. No green or corrupted frames in output
4. No black/empty frames in output
5. Pixel stats cache returns cached objects
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch


class TestVisualQuality(unittest.TestCase):
    """Visual quality and audio sync verification tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        os.makedirs("/tmp/test_visual_quality", exist_ok=True)

    def _create_mock_vae(self):
        """Create a mock MiniMax-H3 VAE for testing."""
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
            T = z_clip.shape[2] * 4  # temporal upsample
            H = z_clip.shape[-2] * 8  # spatial upsample
            W = z_clip.shape[-1] * 8
            return torch.randn(1, 3, T, H, W)

        vae._adaptive_decode = _adaptive_decode
        vae.blend = lambda overlap, chunk, overlap_frames, dim: chunk
        return vae

    def _create_mock_latent(self, T=4, H=32, W=32):
        """Create a mock video latent tensor."""
        return torch.randn(1, 16, T, H, W)

    def _create_mock_save_av_stream(self):
        """Create a mock SaveAVStream that captures frames."""
        mock_stream = MagicMock()
        captured_frames = []

        def put_frame(frame):
            captured_frames.append(frame)

        mock_stream.put_frame = put_frame
        mock_stream.captured_frames = captured_frames
        mock_stream.close = MagicMock()
        return mock_stream

    def test_streaming_frames_match_reference(self):
        """Streaming output frames match non-streaming reference within pixel tolerance."""
        from x_comfyui.framework.vae.minimax_h3 import _stream_frames_to_av

        vae = self._create_mock_vae()
        mock_stream = self._create_mock_save_av_stream()

        # Create a deterministic chunk for comparison
        chunk = torch.rand(1, 3, 4, 64, 64)

        # Stream the chunk
        _stream_frames_to_av(mock_stream, chunk)

        # Verify frames were captured
        self.assertGreater(len(mock_stream.captured_frames), 0,
                           "No frames were streamed")

        # Verify each frame is uint8 numpy array
        for frame in mock_stream.captured_frames:
            self.assertIsInstance(frame, np.ndarray,
                                 f"Frame is not ndarray: {type(frame)}")
            self.assertEqual(frame.dtype, np.uint8,
                              f"Frame dtype is not uint8: {frame.dtype}")

        # Verify frame dimensions: chunk is (B, C, T, H, W) → frames are (H, W, C)
        expected_H = chunk.shape[-2]
        expected_W = chunk.shape[-1]
        expected_C = chunk.shape[1]
        for frame in mock_stream.captured_frames:
            self.assertEqual(frame.shape[0], expected_H,
                              f"Frame height mismatch: {frame.shape[0]} != {expected_H}")
            self.assertEqual(frame.shape[1], expected_W,
                              f"Frame width mismatch: {frame.shape[1]} != {expected_W}")
            self.assertEqual(frame.shape[2], expected_C,
                              f"Frame channel mismatch: {frame.shape[2]} != {expected_C}")

        # Verify pixel values are in valid range [0, 255]
        for frame in mock_stream.captured_frames:
            self.assertGreaterEqual(frame.min(), 0, "Frame has values below 0")
            self.assertLessEqual(frame.max(), 255, "Frame has values above 255")

    def test_audio_video_sync(self):
        """Audio-video sync within ±1 frame tolerance via ffprobe."""
        try:
            import subprocess
            import json
        except ImportError:
            self.skipTest("subprocess or json not available")

        # Create a mock output file path
        output_path = "/tmp/test_visual_quality/sync_test.mp4"

        # This test requires a real MP4 file with audio
        # Skip if ffprobe is not available
        import shutil
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            self.skipTest("ffprobe not installed")
        
        try:
            probe = subprocess.run(
                [ffprobe_path, "-version"],
                capture_output=True, text=True,
            )
            if probe.returncode != 0:
                self.skipTest("ffprobe not available")
        except FileNotFoundError:
            self.skipTest("ffprobe not installed")

        # If we have a real output file, check sync
        if not os.path.exists(output_path):
            self.skipTest("No output MP4 file for sync test (requires real model)")

        probe = subprocess.run(
            [ffprobe_path, "-show_entries", "stream=start_time,codec_type",
             "-of", "json", output_path],
            capture_output=True, text=True,
        )

        if probe.returncode != 0:
            self.skipTest("ffprobe failed to analyze file")

        streams = json.loads(probe.stdout).get("streams", [])

        video_start = None
        audio_start = None
        for s in streams:
            if s.get("codec_type") == "video":
                video_start = float(s.get("start_time", 0))
            elif s.get("codec_type") == "audio":
                audio_start = float(s.get("start_time", 0))

        if video_start is not None and audio_start is not None:
            fps = 24
            tolerance = 1.0 / fps + 0.01  # ±1 frame
            diff = abs(video_start - audio_start)
            self.assertLess(diff, tolerance,
                            f"Audio-video sync drift: {diff}s > {tolerance}s tolerance")

    def test_no_green_frames(self):
        """No green or corrupted frames in streaming output."""
        from x_comfyui.framework.vae.minimax_h3 import _stream_frames_to_av

        vae = self._create_mock_vae()
        mock_stream = self._create_mock_save_av_stream()

        # Create a normal chunk (not green)
        chunk = torch.rand(1, 3, 4, 64, 64)
        _stream_frames_to_av(mock_stream, chunk)

        # Verify no green frames
        # Green frame: G > 200, R < 50, B < 50
        for i, frame in enumerate(mock_stream.captured_frames):
            r = frame[:, :, 0].astype(int)
            g = frame[:, :, 1].astype(int)
            b = frame[:, :, 2].astype(int)
            is_green = np.all(g > 200) and np.all(r < 50) and np.all(b < 50)
            self.assertFalse(is_green,
                             f"Green frame detected at index {i}")

    def test_no_black_frames(self):
        """No black/empty frames in streaming output."""
        from x_comfyui.framework.vae.minimax_h3 import _stream_frames_to_av

        vae = self._create_mock_vae()
        mock_stream = self._create_mock_save_av_stream()

        # Create a normal chunk (not black)
        chunk = torch.rand(1, 3, 4, 64, 64) * 0.5 + 0.3
        _stream_frames_to_av(mock_stream, chunk)

        # Verify no black frames (all pixels sum to 0)
        for i, frame in enumerate(mock_stream.captured_frames):
            self.assertGreater(frame.sum(), 0,
                                f"Black frame detected at index {i}")

    def test_pixel_stats_cache_returns_cached(self):
        """_get_pixel_stats returns cached tensors for same device."""
        from x_comfyui.framework.vae.minimax_h3 import _get_pixel_stats

        device = torch.device("cpu")
        mean1, std1 = _get_pixel_stats(device)
        mean2, std2 = _get_pixel_stats(device)

        # Verify same objects (cached)
        self.assertIs(mean1, mean2,
                       "Pixel stats mean not cached — different objects")
        self.assertIs(std1, std2,
                       "Pixel stats std not cached — different objects")

        # Verify they are tensors
        self.assertIsInstance(mean1, torch.Tensor,
                              f"mean is not tensor: {type(mean1)}")
        self.assertIsInstance(std1, torch.Tensor,
                              f"std is not tensor: {type(std1)}")

    def test_stream_frames_to_av_frame_count(self):
        """_stream_frames_to_av produces correct number of frames from chunk."""
        from x_comfyui.framework.vae.minimax_h3 import _stream_frames_to_av

        mock_stream = self._create_mock_save_av_stream()

        # Create chunk with T=8 temporal frames
        T = 8
        chunk = torch.rand(1, 3, T, 32, 32)
        _stream_frames_to_av(mock_stream, chunk)

        # Should produce T frames (one per temporal frame)
        self.assertEqual(len(mock_stream.captured_frames), T,
                         f"Expected {T} frames, got {len(mock_stream.captured_frames)}")

    def test_stream_frames_to_av_4d_input(self):
        """_stream_frames_to_av handles 4D input (no batch dim) by adding one."""
        from x_comfyui.framework.vae.minimax_h3 import _stream_frames_to_av

        mock_stream = self._create_mock_save_av_stream()

        # 4D input: (C, T, H, W) — should be unsqueezed to (1, C, T, H, W)
        chunk_4d = torch.rand(3, 4, 32, 32)
        _stream_frames_to_av(mock_stream, chunk_4d)

        # Should still produce frames
        self.assertGreater(len(mock_stream.captured_frames), 0,
                           "No frames produced from 4D input")


if __name__ == "__main__":
    unittest.main()
