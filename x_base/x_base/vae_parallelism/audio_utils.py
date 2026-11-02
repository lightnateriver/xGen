"""Audio I/O utilities for MiniMax-H3 video+audio generation.

Reusable WAV write/cleanup functions used by both:
- Route A: ``save_minimax_h3_output`` (infer_tools.py) — full decode then save
- Route B: ``SaveAVStream`` + ``MiniMaxH3DecodeStep`` patch (minimax_h3.py) — streaming
"""

import os
import wave

import numpy as np
import torch
import logging

logger = logging.getLogger("audio_utils")


def save_audio_to_wav(audio, sampling_rate, audio_path=None, save_path=None):
    """Save audio tensor to a WAV file.

    Args:
        audio: ``torch.Tensor`` of shape ``(1, C, N)`` or ``(C, N)``, float in ``[-1, 1]``.
        sampling_rate: int, sample rate in Hz.
        audio_path: explicit path for the WAV file. If *None*, derived from *save_path*.
        save_path: video save path, used to derive *audio_path* when *audio_path* is *None*.

    Returns:
        str or None: path to the saved WAV file, or *None* if *audio* is *None*.
    """
    if audio is None or sampling_rate is None:
        return None

    if audio_path is None:
        if save_path is None:
            raise ValueError("Either audio_path or save_path must be provided")
        base = os.path.splitext(save_path)[0]
        audio_path = base + ".wav"

    # --- tensor → numpy ---
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu()
        if audio.dim() == 3:          # (1, C, N) → (C, N)
            audio = audio.squeeze(0)
        audio_np = audio.numpy()
    else:
        audio_np = np.asarray(audio)
        if audio_np.ndim == 3:
            audio_np = audio_np[0]

    # --- float [-1,1] → int16 ---
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    # WAV expects (num_samples, channels)
    audio_int16 = audio_int16.T

    with wave.open(audio_path, "w") as wav_file:
        n_channels = audio_int16.shape[1] if audio_int16.ndim == 2 else 1
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2)       # int16 = 2 bytes
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_int16.tobytes())

    logger.info(f"Saved audio to {audio_path} (sr={sampling_rate})")
    return audio_path


def cleanup_audio(audio_path):
    """Delete a temporary audio file after it has been merged into MP4.

    Args:
        audio_path: path to the WAV file. *None* is a silent no-op.
    """
    if audio_path is None:
        return
    try:
        os.remove(audio_path)
        logger.info(f"Cleaned up temporary audio: {audio_path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to cleanup audio {audio_path}: {e}")
