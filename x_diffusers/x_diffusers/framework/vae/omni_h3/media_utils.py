# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Modified for xGen: runtime/import adaptation; see UPSTREAM.json.
from __future__ import annotations

import io

import queue

import threading

from collections.abc import Callable, Iterable, Mapping

from fractions import Fraction

from typing import Any, NamedTuple, cast

import av

import numpy as np

import logging

logger = logging.getLogger(__name__)

_CHUNKED_MP4_DONE = object()

class _QueuedChunk(NamedTuple):
    """One queued chunk plus the callback that returns its buffer to its owner."""

    frames: np.ndarray
    on_consumed: Callable[[], None] | None

def _release_chunk(on_consumed: Callable[[], None] | None) -> None:
    """Hand a chunk's buffer back, never letting the callback wedge the worker."""
    if on_consumed is None:
        return
    try:
        on_consumed()
    except BaseException:  # noqa: BLE001
        logger.exception("Chunked MP4 encoder failed to release a consumed chunk")

def normalize_preencode_batch_frames(value: Any) -> int:
    """Validate the request's MP4 transfer/encoding batch threshold."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("preencode_batch_frames must be a positive integer")
    return value

def normalize_video_codec_options(value: Any) -> dict[str, str] | None:
    """Coerce a request's ``video_codec_options`` into PyAV's str->str contract.

    The value reaches a worker straight from ``extra_params``, so reject a
    non-mapping here rather than letting it fail inside the encoder after the
    decode has already run.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("video_codec_options must be a JSON object of encoder option names to values")
    return {str(key): str(item) for key, item in value.items()}

def _validate_video_chunk(chunk: np.ndarray, *, width: int, height: int) -> None:
    """Validate a ``(T, H, W, 3)`` uint8 RGB chunk against a session's frame size."""
    if chunk.ndim != 4 or chunk.shape[-1] != 3:
        raise ValueError("video chunk must have shape (T, H, W, 3)")
    if chunk.dtype != np.uint8:
        raise ValueError("video chunk must have dtype uint8")
    if chunk.shape[1] != height or chunk.shape[2] != width:
        raise ValueError("video chunks in a session must use a consistent frame size")

class ChunkedMP4Encoder:
    """Encode temporal video chunks while the producer is still decoding.

    A bounded queue and one muxing worker provide ordered backpressure. What
    that bounds is pending raw-frame memory: at most ``max_pending`` chunks wait
    to be encoded, so the bound is in chunks rather than bytes and a taller
    frame or a longer chunk raises it. The muxed container is not bounded -- it
    accumulates in memory and grows with output duration. Chunks use the same
    ``(T, H, W, 3)`` uint8 contract as :func:`mux_video_audio_bytes`.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        audio_waveform: np.ndarray | None = None,
        audio_sample_rate: int | None = None,
        max_pending: int = 2,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        crf: str = "18",
        video_codec_options: dict[str, str] | None = None,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_pending)
        self._result: bytes | None = None
        self.frame_count = 0
        self._error: BaseException | None = None
        self._closed = False
        self._aborted = False
        self._input_done = False
        self._state_lock = threading.Lock()

        def run() -> None:
            frames = self._frames()
            try:
                self._result = mux_av_video_audio_bytes(
                    frames,
                    width=self.width,
                    height=self.height,
                    audio_waveform=audio_waveform,
                    fps=fps,
                    audio_sample_rate=audio_sample_rate,
                    video_codec=video_codec,
                    audio_codec=audio_codec,
                    crf=crf,
                    video_codec_options=video_codec_options,
                    encode_audio_first=True,
                )
            except BaseException as exc:
                self._error = exc
                # Closing is what runs the generator's cleanup and releases the
                # chunk it was mid-way through. Waiting for collection would
                # never get there: the traceback just stored on this encoder
                # keeps the abandoned generator alive for the process's life.
                frames.close()
                self._drain_until_done()

        self._thread = threading.Thread(target=run, name="chunked-mp4", daemon=True)
        self._thread.start()

    def _drain_until_done(self) -> None:
        # A flush/mux failure can occur after _frames consumed the sentinel.
        if self._input_done:
            return
        while True:
            entry = self._queue.get()
            if entry is _CHUNKED_MP4_DONE:
                return
            _release_chunk(cast(_QueuedChunk, entry).on_consumed)

    def _frames(self):
        while True:
            entry = self._queue.get()
            if entry is _CHUNKED_MP4_DONE:
                self._input_done = True
                return
            assert isinstance(entry, _QueuedChunk)
            try:
                for frame_data in entry.frames:
                    yield av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    self.frame_count += 1
            finally:
                # Also runs when the muxer raises part-way through this chunk
                # and abandons the generator, so a lent buffer is never
                # stranded in the encoder.
                _release_chunk(entry.on_consumed)

    def _validate_chunk(self, chunk: np.ndarray) -> None:
        _validate_video_chunk(chunk, width=self.width, height=self.height)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise RuntimeError("ChunkedMP4Encoder is already closed")

    def push(self, chunk: np.ndarray, *, on_consumed: Callable[[], None] | None = None) -> None:
        """Queue a chunk; asynchronous failures surface here or at finish().

        ``on_consumed`` fires exactly once after this encoder has read
        ``chunk``, including when the chunk is discarded by an abort or by a
        failed encode. Ownership transfers on call: whether this returns or
        raises, the callback is this encoder's responsibility, so a caller
        lending a pooled host buffer never compensates on the error path.
        """
        try:
            self._validate_chunk(chunk)
            self._raise_if_failed()
        except BaseException:
            _release_chunk(on_consumed)
            raise
        self._queue.put(_QueuedChunk(chunk, on_consumed))
        self._raise_if_failed()

    def _send_done(self) -> None:
        self._queue.put(_CHUNKED_MP4_DONE)

    def finish(self) -> bytes:
        """Flush the muxer and return complete progressive MP4 bytes."""
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._send_done()
        self._thread.join()
        if self._error is not None:
            raise self._error
        if self._aborted:
            raise RuntimeError("ChunkedMP4Encoder was aborted")
        assert self._result is not None
        return self._result

    def abort(self) -> None:
        """Discard pending chunks and join the worker after cancellation.

        Any chunk already being encoded must finish before the worker exits.
        """
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._aborted = True
                while True:
                    try:
                        entry = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if entry is not _CHUNKED_MP4_DONE:
                        _release_chunk(cast(_QueuedChunk, entry).on_consumed)
                self._send_done()
        self._thread.join()

    close = finish

    def __enter__(self) -> ChunkedMP4Encoder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()

def mux_av_video_audio_bytes(
    video_frames: Iterable[av.VideoFrame],
    width: int,
    height: int,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int | None = None,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
    encode_audio_first: bool = False,
) -> bytes:
    """Mux MP4 bytes, optionally encoding ready audio before waiting for video."""
    buf = io.BytesIO()
    with cast(Any, av.open(buf, mode="w", format="mp4")) as container:
        v_stream = cast(
            av.VideoStream,
            container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)),
        )
        v_stream.width = width
        v_stream.height = height
        v_stream.pix_fmt = "yuv420p"

        options: dict[str, object] = {"crf": str(crf)}
        if video_codec_options:
            options.update(video_codec_options)
        v_stream.options = options

        a_stream: av.AudioStream | None = None
        samples: np.ndarray | None = None
        layout: str | None = None
        if audio_waveform is not None:
            effective_audio_sample_rate = 44100 if audio_sample_rate is None else audio_sample_rate
            samples = audio_waveform.astype(np.float32)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
                samples = np.ascontiguousarray(samples.T)
            num_channels = samples.shape[0]
            layout = "stereo" if num_channels >= 2 else "mono"
            a_stream = cast(av.AudioStream, container.add_stream(audio_codec, rate=effective_audio_sample_rate))
            a_stream.layout = layout

        def encode_audio() -> None:
            if a_stream is not None and audio_waveform is not None:
                if samples is None or layout is None:
                    raise ValueError("Audio samples were not prepared for muxing.")
                audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
                audio_frame.sample_rate = effective_audio_sample_rate
                # AAC has a one-frame encoder delay. Mark the input waveform as
                # starting at t=0 so the MP4 muxer writes the corresponding negative
                # priming timestamp instead of exposing the delay as leading silence.
                audio_frame.pts = 0
                audio_frame.time_base = Fraction(1, effective_audio_sample_rate)
                for packet in a_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in a_stream.encode():
                    container.mux(packet)

        if encode_audio_first:
            encode_audio()

        for frame in video_frames:
            for packet in v_stream.encode(frame):
                container.mux(packet)
        for packet in v_stream.encode():
            container.mux(packet)

        if not encode_audio_first:
            encode_audio()

    return buf.getvalue()
