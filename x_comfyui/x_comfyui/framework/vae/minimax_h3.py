"""NPU adaptation for the MiniMax-H3 VAE in ComfyUI.

All patching happens at **import time**.  No NPUTurbo node call needed.

Patches:
  - ``vae.Attention.forward`` → routes through ``attention_manager``
  - ``MiniMaxH3VideoVAE.decode`` → NPU dispatch (lightning / upstream)
    with streaming output via ``av_stream_config`` dict
  - ``MiniMaxH3VideoVAE.enable_lightning`` → from x_base
  - ``MiniMaxH3VideoVAE`` → add_lightning_init (adds ``lightning`` attribute)

Streaming optimizations (ported from x_diffusers):
  - V1+V2: ``_stream_frames_to_av`` + ``_decode_chunks_streaming``
  - V3: VAEManager spatial tiling
  - V6: Async D2H copy stream (``X_VAE_COPY_STREAM``)
  - V8: Pixel stats cache
  - ``av_stream_config`` dict on ``decode`` for streaming output (VAE manages lifecycle)

Environment variables:
  - ``X_VAE_COPY_STREAM`` — "1" enables async D2H copy stream (default "0")

ComfyUI's MiniMax-H3 VAE (``comfy/ldm/minimax/vae.py``):
  - Input shape: ``(B, S, C)`` — with batch dimension
  - Fused QKV: ``self.to_qkv``
  - RMSNorm: ``self.norm_q``, ``self.norm_k`` (on dim_head, no affine)
  - Output: ``self.to_out`` (single Linear)
  - RoPE: ``comfy.quant_ops.ck.apply_rope_split_half``
  - TransformerBlock: norm1 → attn → scale1 → norm2 → ff → scale2
  - ViT3DDecoder: x_embedder → register_tokens → transformer_blocks → norm_out → proj_out
  - MiniMaxH3VideoVAE: encode/decode with temporal chunking and spatial tiling
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch_npu  # noqa: F401

from x_comfyui.adaptor.env import get_vae_copy_stream
from x_base.vae_parallelism.save_av_stream import SaveAVStream

logger = logging.getLogger("x_comfyui")

# ===================================================================
# Streaming infrastructure (ported from x_diffusers)
# ===================================================================

# ImageNet denormalization constants (module-level to avoid per-call allocation)
_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)
_PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)
_pixel_stats_cache: dict = {}

# Async D2H infrastructure (only used when X_VAE_COPY_STREAM=1)
_copy_stream = None
_write_executor = ThreadPoolExecutor(max_workers=1)


def _get_pixel_stats(device):
    """Cache device-specific ImageNet normalization constants."""
    if device not in _pixel_stats_cache:
        _pixel_stats_cache[device] = (_PIXEL_MEAN.to(device), _PIXEL_STD.to(device))
    return _pixel_stats_cache[device]


def _get_copy_stream():
    """Get or create a separate NPU stream for async D2H copy."""
    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.npu.Stream()
    return _copy_stream


def _flush_writes():
    """Block until all pending async frame-write tasks complete.

    No-op when X_VAE_COPY_STREAM is off (sync path writes inline).
    """
    if not get_vae_copy_stream():
        return
    _write_executor.submit(lambda: None).result()


def _stream_frames_to_av(av_stream, chunk):
    """Convert decoded chunk to uint8 HWC frames and stream to SaveAVStream.

    Default (X_VAE_COPY_STREAM=0): D2H on compute stream, frames written inline.
    Async  (X_VAE_COPY_STREAM=1): D2H on a separate stream, frames written
    in a background thread so compute is not blocked by CPU-side work.
    """
    if chunk.dim() == 4:
        chunk = chunk.unsqueeze(0)
    mean, std = _get_pixel_stats(chunk.device)
    frames = (chunk.float() * std + mean).clamp(0, 1)
    frames = frames.squeeze(0).permute(1, 2, 3, 0)  # (F, H, W, C)
    frames = (frames * 255).to(torch.uint8)

    if get_vae_copy_stream():
        _stream_frames_async(av_stream, frames)
    else:
        """D2H on compute stream, write frames inline."""
        frames_np = frames.cpu().numpy()
        for f in range(frames_np.shape[0]):
            av_stream.put_frame(frames_np[f])


def _stream_frames_async(av_stream, frames):
    """D2H on a separate stream, background thread writes frames."""
    event = torch.npu.Event()
    event.record()
    copy_stream = _get_copy_stream()
    with torch.npu.stream(copy_stream):
        event.wait()
        frames_cpu = frames.cpu()

    def _write():
        copy_stream.synchronize()
        frames_np = frames_cpu.numpy()
        for f in range(frames_np.shape[0]):
            av_stream.put_frame(frames_np[f])

    _write_executor.submit(_write)


def _process_chunk_frame(vae, chunk, overlap, vae_manager, av_stream, rank):
    """Process first frame of a chunk: blend overlap, spatial gather, and stream.
    
    Returns the processed chunk.
    """
    if overlap is not None:
        chunk = vae.blend(overlap, chunk, vae.frame_overlap, dim=-3)
    
    # Spatial gather for lightning mode (collective — all ranks must call)
    if vae_manager is not None:
        chunk = vae_manager.align_out(chunk)
        chunk = vae_manager.collect_out(chunk)
    
    # Stream frames (rank 0 only, av_stream may be None before integration)
    if rank == 0 and av_stream is not None:
        _stream_frames_to_av(av_stream, chunk)
    
    # Barrier: rank 0 streaming must finish before next iteration's collective
    if vae_manager is not None and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    return chunk


def _process_overlap_tail(vae, overlap, vae_manager, av_stream, rank, decoded_chunks):
    """Process remaining overlap frames after chunk loop completes."""
    if overlap is None:
        return
    
    if vae_manager is not None:
        overlap = vae_manager.align_out(overlap)
        overlap = vae_manager.collect_out(overlap)
    
    decoded_chunks.append(overlap)
    
    if rank == 0 and av_stream is not None:
        _stream_frames_to_av(av_stream, overlap)


def _trim_padding_tail(vae, dec, z, pad_tokens, tokens_chunk_size, temporal_ratio):
    """Remove padding frames from decoded output."""
    intra_tail = vae.clip_length % temporal_ratio
    num_tokens_before_pad = z.shape[2] - pad_tokens
    pad_frames = sum(
        intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
        for k in range(pad_tokens)
    )
    return dec[:, :, :-pad_frames]


def _decode_chunks_streaming(vae, z, vae_manager, av_stream):
    """Replicate upstream ``decode_temporal`` chunk loop with streaming to SaveAVStream.

    Decodes the latent ``z`` chunk-by-chunk (each chunk = ``tokens_chunk_size``
    latents → ``chunk_num_frames`` pixel frames). After each chunk is blended,
    if ``vae_manager`` is provided, spatial tiles are gathered across ranks.
    Then frames are streamed to ``av_stream`` (rank 0 only).

    Returns the full decoded tensor (same as upstream ``decode_temporal``).
    """
    # Normalize latents (same as ComfyUI's decode method)
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z)
    z = z * latents_std + latents_mean

    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.token_drop
    temporal_ratio = vae.vae_ratio_t
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    decoded_chunks = []
    overlap = None
    
    for i in range(num_chunks):
        start = i * tokens_chunk_size
        clip = vae._adaptive_decode(z[:, :, start: start + tokens_chunk_size + vae.token_overlap])
        
        for j in range(int(token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip[:, :, frame_start: frame_start + chunk_num_frames]
            chunk = chunk[:, :, vae.frame_pre_padding:]
            
            if j == 0:
                chunk = _process_chunk_frame(vae, chunk, overlap, vae_manager, av_stream, rank)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk

    _process_overlap_tail(vae, overlap, vae_manager, av_stream, rank, decoded_chunks)
    dec = torch.cat(decoded_chunks, dim=2)

    if pad_tokens > 0:
        dec = _trim_padding_tail(vae, dec, z, pad_tokens, tokens_chunk_size, temporal_ratio)
    
    return dec


def _create_av_stream(vae, z, save_path=None, fps=24, audio_path=None,
                      bit_depth=8, codec=None):
    """Create SaveAVStream on rank 0, return None on other ranks.

    Args:
        vae: The VAE model (must have vae_ratio attribute)
        z: The latent tensor
        save_path: Output video file path (required)
        fps: Frames per second (default 24)
        audio_path: Optional WAV file path for audio input
        bit_depth: 8 or 10 (default 8)
        codec: Video codec override (e.g. "libx264", "libx265")
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return None
    if save_path is None:
        return None
    vae_spatial_ratio = vae.vae_ratio
    height = z.shape[-2] * vae_spatial_ratio
    width = z.shape[-1] * vae_spatial_ratio
    return SaveAVStream(save_path, audio_path, fps, width, height,
                        bit_depth=bit_depth, codec=codec)


def _create_vae_manager(vae, z):
    """Create VAEManager for spatial tiling (decode)."""
    from x_base.vae_parallelism.vae_mgr import VAEManager
    return VAEManager(
        z.shape,
        factor=vae.vae_ratio,
        is_encode=False,
        pad_mode="all_sides_valid",
        pad_content="constant",
        use_blend=False,
    )


def patch_vae_classes():
    """Patch VAE classes for MiniMax-H3 NPU acceleration.

    Call this once at import time (from x_comfyui.__init__).
    Silently skips if ComfyUI doesn't have MiniMax-H3 VAE support.
    """
    try:
        from comfy.ldm.minimax.vae import (
            MiniMaxH3VideoVAE,
            Attention as VAEAttention,
            TransformerBlock as VAETransformerBlock,
            ViT3DDecoder,
        )
    except ImportError:
        logger.info("x_comfyui: MiniMax-H3 VAE not found, skipping VAE patches")
        return

    # Idempotency guard
    if getattr(VAEAttention.forward, '_x_comfyui_patched', False):
        logger.info("x_comfyui: VAE patches already applied, skipping")
        return

    from x_base import attention_manager

    # x_base 是硬依赖，直接 import
    from x_base import enable_lightning, add_lightning_init

    # Save original forwards
    _original_encode = MiniMaxH3VideoVAE.encode
    _original_decode = MiniMaxH3VideoVAE.decode

    # -----------------------------------------------------------------------
    # Patch VAE Attention.forward — routes through attention_manager
    # -----------------------------------------------------------------------
    def _npu_vae_attn_forward(self, x, rotary_pos_emb=None):
        """NPU VAE attention: routes through attention_manager."""
        import comfy.rmsnorm
        import comfy.quant_ops
        from x_comfyui.framework.rope_utils import _fused_rope_minimax_h3

        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        # RMSNorm (keep original comfy kernel)
        query = comfy.rmsnorm.rms_norm(query, self.norm_q.weight, self.norm_q.eps)
        key = comfy.rmsnorm.rms_norm(key, self.norm_k.weight, self.norm_k.eps)
        # RMSNorm — comfy.rmsnorm.rms_norm routes through F.rms_norm,
        # which is auto-patched to npu_rms_norm by x_base at import time.

        # RoPE (fused or fallback via rope_manager.rope_rotated_half)
        if rotary_pos_emb is not None:
            rot = rotary_pos_emb.shape[-3] * 2
            query, key = _fused_rope_minimax_h3(query, key, rotary_pos_emb, rot)

        # Attention via attention_manager
        out = attention_manager.attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        out = out.transpose(1, 2).flatten(2, 3).type_as(query).nan_to_num_(0.0)
        return self.to_out(out)

    VAEAttention.forward = _npu_vae_attn_forward

    # -----------------------------------------------------------------------
    # Patch MiniMaxH3VideoVAE — lightning + NPU encode/decode dispatch
    # -----------------------------------------------------------------------
    if not hasattr(MiniMaxH3VideoVAE, "enable_lightning"):
        MiniMaxH3VideoVAE.enable_lightning = enable_lightning
    add_lightning_init(MiniMaxH3VideoVAE)

    # --- NPU encode dispatch ---
    @torch.no_grad()
    def _encode_ascend(self, x, device=None):
        """NPU encode — delegates to upstream encode (no SP needed for CNN encoder)."""
        return _original_encode(self, x, device)

    # --- NPU decode dispatch helpers ---

    def _decode_lightning_single(vae, z, output_buffer=None, av_stream=None, av_stream_config=None):
        """Single-card lightning: VAEManager spatial tiling + streaming.

        When av_stream_config is provided, all ranks use VAEManager for spatial
        parallelism (collective ops require all ranks to participate).
        Only rank 0 streams frames to av_stream (av_stream is None on other ranks).
        """
        if not getattr(vae, "_lightning_enabled", False):
            try:
                vae.enable_lightning()
                vae._lightning_enabled = True
                logger.info("x_comfyui: VAE lightning enabled")
            except Exception as e:
                logger.warning(f"x_comfyui: Failed to enable VAE lightning: {e}")
                vae._lightning_enabled = False

        if av_stream_config:
            # Streaming path: VAEManager + _decode_chunks_streaming
            vae_manager = _create_vae_manager(vae, z)
            tile = vae_manager.get_tile_from_x(z)
            decoded = _decode_chunks_streaming(vae, tile, vae_manager, av_stream)
            _flush_writes()
            return decoded
        else:
            # Non-streaming: original decode
            return _original_decode(vae, z, output_buffer)

    @torch.no_grad()
    def _decode_ascend(self, z, output_buffer=None, av_stream_config=None):
        """NPU decode — delegates to _decode_lightning_single.

        _decode_lightning_single handles:
        - lightning enable (with fallback on failure)
        - VAEManager spatial tiling + streaming (when av_stream_config provided)
        - _original_decode fallback (when no streaming)
        """
        av_stream = None
        if av_stream_config:
            av_stream = _create_av_stream(self, z, **av_stream_config)

        try:
            result = _decode_lightning_single(
                self, z, output_buffer,
                av_stream=av_stream,
                av_stream_config=av_stream_config,
            )
        except Exception:
            if av_stream is not None:
                av_stream.close()
            raise

        if av_stream is not None:
            av_stream.close()
        return result

    # Patch encode/decode
    MiniMaxH3VideoVAE.encode = _encode_ascend
    MiniMaxH3VideoVAE.decode = _decode_ascend

    # Set idempotency flag
    VAEAttention.forward._x_comfyui_patched = True

    logger.info(
        "x_comfyui: MiniMax-H3 VAE patches applied "
        "(attention + lightning + NPU dispatch + streaming)"
    )
