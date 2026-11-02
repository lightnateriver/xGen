"""Streaming audiovisual writer using ffmpeg subprocess.

Like :class:`SaveVideoStream` but writes video frames **and** audio into a
single MP4 in one pass via ffmpeg's dual-input mode (stdin rawvideo + WAV file).

The temporary WAV file is automatically deleted after ``close()`` finishes.
"""

import os
import subprocess

import numpy as np
from torch.multiprocessing import Process, Queue
import logging

from .audio_utils import cleanup_audio

logger = logging.getLogger("save_av")

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")


def write_av(queue, save_path, audio_path, fps, width, height, bit_depth=8, codec=None):
    """Subprocess entry: read frames from queue, pipe to ffmpeg stdin."""
    # Input pixel format: 8-bit = rgb24, 10-bit = rgb48le
    in_pix_fmt = "rgb24" if bit_depth == 8 else "rgb48le"
    # Video codec: use explicit codec or default to libx264
    v_codec = codec or "libx264"
    # Output pixel format: 8-bit = yuv420p, 10-bit = yuv420p10le
    out_pix_fmt = "yuv420p" if bit_depth == 8 else "yuv420p10le"

    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo",
        "-pix_fmt", in_pix_fmt,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",                       # input 0: stdin (video)
    ]
    if audio_path:
        cmd += ["-i", audio_path]             # input 1: WAV file (audio)
    cmd += [
        "-c:v", v_codec,
        "-preset", "fast",
        "-crf", "10",
        "-pix_fmt", out_pix_fmt,
    ]
    if audio_path:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += ["-loglevel", "error", save_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames_num = 0
    while True:
        frame = queue.get()
        if frame is None:
            break
        proc.stdin.write(frame.tobytes())
        frames_num += 1

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        logger.error(f"ffmpeg exited with code {proc.returncode} after {frames_num} frames")
    else:
        logger.info(f"SaveAVStream: wrote {frames_num} frames to {save_path}")


class SaveAVStream:
    """Stream video frames + audio into a single MP4 via ffmpeg.

    The *audio_path* WAV file is treated as temporary — it is deleted
    automatically after ``close()`` completes the merge.

    Args:
        save_path: Output MP4 file path.
        audio_path: Temporary WAV file path (or None for video-only).
        fps: Frames per second.
        width: Output video width in pixels.
        height: Output video height in pixels.
        bit_depth: 8 or 10 (default 8). Affects pixel format.
        codec: Video codec override (e.g. "libx264", "libx265").
               None defaults to "libx264".

    Usage::

        stream = SaveAVStream(save_path, audio_path, fps, width, height)
        for frame in frames:          # frame: uint8 HWC numpy
            stream.put_frame(frame)
        stream.close()                # WAV deleted here
    """

    def __init__(self, save_path, audio_path, fps, width, height,
                 bit_depth=8, codec=None):
        self._audio_path = audio_path
        self.frame_queue = Queue(maxsize=60)
        self.proc = Process(
            target=write_av,
            args=(self.frame_queue, save_path, audio_path, fps, width, height,
                  bit_depth, codec),
        )
        self.proc.start()

    def put_frame(self, frame):
        """Accept a uint8 HWC numpy array and enqueue it."""
        self.frame_queue.put(frame)

    def close(self):
        self.frame_queue.put(None)
        self.proc.join()
        cleanup_audio(self._audio_path)
