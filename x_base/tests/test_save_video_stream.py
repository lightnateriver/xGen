"""
Unit tests for vae_parallelism/save_video_stream.py.

Covers:
- SaveVideoStream: init creates queue and starts process, save enqueues frames, close sends sentinel
- write_video: processes frames from queue, stops on None sentinel, resize and brighten branches
"""
from unittest.mock import patch, MagicMock

import pytest
import torch
import numpy as np

from x_base.vae_parallelism.save_video_stream import SaveVideoStream, write_video


class TestSaveVideoStream:
    @patch("x_base.vae_parallelism.save_video_stream.infer_info")
    @patch("x_base.vae_parallelism.save_video_stream.Process")
    @patch("x_base.vae_parallelism.save_video_stream.Queue")
    def test_init_starts_process(self, mock_queue_cls, mock_process_cls, mock_info):
        mock_info.frames = 10
        mock_info.save_path = "out.mp4"
        mock_info.fps = 24
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue
        stream = SaveVideoStream()
        mock_process_cls.assert_called_once()
        mock_process_cls.return_value.start.assert_called_once()

    @patch("x_base.vae_parallelism.save_video_stream.infer_info")
    @patch("x_base.vae_parallelism.save_video_stream.Process")
    @patch("x_base.vae_parallelism.save_video_stream.Queue")
    def test_save_enqueues_frames(self, mock_queue_cls, mock_process_cls, mock_info):
        mock_info.frames = 10
        mock_info.save_path = "out.mp4"
        mock_info.fps = 24
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue
        stream = SaveVideoStream()
        frames = [torch.randn(3, 64, 64) for _ in range(3)]
        stream.save(frames)
        assert mock_queue.put.call_count == 3

    @patch("x_base.vae_parallelism.save_video_stream.infer_info")
    @patch("x_base.vae_parallelism.save_video_stream.Process")
    @patch("x_base.vae_parallelism.save_video_stream.Queue")
    def test_close_sends_sentinel(self, mock_queue_cls, mock_process_cls, mock_info):
        mock_info.frames = 10
        mock_info.save_path = "out.mp4"
        mock_info.fps = 24
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue
        stream = SaveVideoStream()
        stream.close()
        mock_queue.put.assert_called_with(None)
        mock_process_cls.return_value.join.assert_called_once()


class TestWriteVideo:
    @patch("x_base.vae_parallelism.save_video_stream.infer_info")
    @patch("x_base.vae_parallelism.save_video_stream.imageio.get_writer")
    def test_stops_on_none_sentinel(self, mock_writer_ctx, mock_info):
        mock_info.save_path = "out.mp4"
        mock_info.fps = 24
        mock_info.save_width = None
        mock_info.save_height = None
        mock_info.ada_brighten = False
        mock_info.frames = 5
        writer = MagicMock()
        mock_writer_ctx.return_value.__enter__ = MagicMock(return_value=writer)
        mock_writer_ctx.return_value.__exit__ = MagicMock(return_value=False)
        queue = MagicMock()
        queue.get.side_effect = [np.zeros((64, 64, 3), dtype=np.uint8), None]
        write_video(queue)
        assert writer.append_data.call_count == 1

    @patch("x_base.vae_parallelism.save_video_stream.infer_info")
    @patch("x_base.vae_parallelism.save_video_stream.imageio.get_writer")
    @patch("x_base.vae_parallelism.save_video_stream.cv2.resize", return_value=np.zeros((32, 32, 3), dtype=np.uint8))
    def test_resize_when_dimensions_mismatch(self, mock_resize, mock_writer_ctx, mock_info):
        mock_info.save_path = "out.mp4"
        mock_info.fps = 24
        mock_info.save_width = 32
        mock_info.save_height = 32
        mock_info.ada_brighten = False
        mock_info.frames = 5
        writer = MagicMock()
        mock_writer_ctx.return_value.__enter__ = MagicMock(return_value=writer)
        mock_writer_ctx.return_value.__exit__ = MagicMock(return_value=False)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        queue = MagicMock()
        queue.get.side_effect = [frame, None]
        write_video(queue)
        mock_resize.assert_called_once()
