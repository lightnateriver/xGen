"""NPU adaptation for the MiniMax-H3 video and audio VAEs.

The MiniMax-H3 video VAE (``AutoencoderKLMiniMaxH3``) uses causal 3-D
convolutions with reflect spatial padding and per-frame GroupNorm.  On NPU
the upstream encode / decode paths work correctly but cannot take advantage
of VAE parallelism or frame-chunked decoding.  This module patches the VAE
with NPU-friendly encode / decode entry points that mirror the Wan VAE
adaptation: tiled / chunked decode with optional VAE-lightning support.

The audio VAE (``AutoencoderKLMiniMaxH3Audio``) is a DAC-lineage 1-D
convolutional stack that runs efficiently as-is; no adaptation is needed
beyond ensuring the module is moved to the NPU device.

Sequence-parallel (SP) mode
---------------------------
When ``vae_lightning`` is enabled and ``torch.distributed`` is initialised,
the ViT decoder runs with Ulysses-style sequence parallelism.  Each temporal
chunk's token sequence is split across ranks; ``all_to_all`` converts between
sequence-shard and head-shard before / after every attention layer so that the
attention result is mathematically identical to single-card decode.  This
replaces the approximate spatial-tiling approach (``VAEManager``) with an
exact parallel decode — no seams, no blend artifacts.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

# Switch: use a separate NPU stream for D2H copy (async) or block on compute stream (sync).
# Default OFF — sync D2H is stable; async copy stream can cause performance fluctuation.
# Set VAE_COPY_STREAM=1 to enable async D2H on a separate stream.
_USE_COPY_STREAM = os.environ.get("VAE_COPY_STREAM", "0") == "1"

import torch
from x_base.device import device_type

if device_type() == "npu":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
import numpy as np

from diffusers.models.autoencoders import autoencoder_kl_minimax_h3
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoViTDecoder3d,
    MiniMaxH3VideoTransformerBlock,
    DecoderOutput,
    AutoencoderKLOutput,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from x_base.vae_parallelism.vae_mgr import VAEManager
from x_base.vae_parallelism.utils import enable_lightning, add_lightning_init
from x_base.vae_parallelism.save_av_stream import SaveAVStream
from x_base.vae_parallelism.audio_utils import save_audio_to_wav
from x_base.utils.infer_info import infer_info
from x_base.sequence_parallelism import (
    all_to_all_before_attn,
    all_to_all_after_attn,
    gather_sequence,
    split_sequence,
    get_pad_manager,
)

# ---------------------------------------------------------------------------
# Video VAE — NPU encode / decode
# ---------------------------------------------------------------------------

# ===================================================================
# SP INFRASTRUCTURE
# ===================================================================

_VAE_SP_MANAGER = None  # cached ParallelManager for VAE SP
_CURRENT_SP_MANAGER = None  # injected into _sp_decoder_forward during upstream _decode

# SP decode strategy toggle (module-level so Route B can read it):
#   "full":   upstream _decode (temporal chunking + spatial tiling + blend) with SP injected
#   "chunks": temporal chunking + SP inside ViT, streaming output via SaveAVStream
_SP_MODE = "vae_mgr"


def _get_vae_sp_manager():
    """Create or reuse a ParallelManager sized to the current world.

    VAE decode runs *after* the transformer forward, so the same process
    group can be reused safely.
    """
    global _VAE_SP_MANAGER
    if _VAE_SP_MANAGER is not None:
        return _VAE_SP_MANAGER

    from x_base import ParallelManager
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    _VAE_SP_MANAGER = ParallelManager(
        dp_size=1, cp_size=1, sp_size=world_size, tp_size=1,
    )
    return _VAE_SP_MANAGER


# -------------------------------------------------------------------
# SP-aware TransformerBlock forward
# -------------------------------------------------------------------
_original_block_forward = MiniMaxH3VideoTransformerBlock.forward


def _sp_block_forward(
        self: MiniMaxH3VideoTransformerBlock,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        parallel_manager=None,
) -> torch.Tensor:
    """Transformer block with optional Ulysses SP.

    When *parallel_manager* is None or ``sp_size <= 1`` the call is
    identical to the upstream forward (zero communication overhead).

    SP data-flow (per layer)::

        hidden_states: (B, S/P, dim)           ← local sequence shard
        Q/K/V:          (B, S/P, H, D)         ← local projections
        RoPE:           (B, S/P, H, D)         ← on split seq, full heads
        all_to_all:     (B, S, H/P, D)         ← full seq, partial heads
        attention:      (B, S, H/P, D)         ← mathematically exact
        all_to_all:     (B, S/P, H, D)         ← back to seq shard
        FFN:            (B, S/P, dim)          ← local
    """
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    if sp_size <= 1:
        return _original_block_forward(self, hidden_states, rotary_emb)

    sp_group = parallel_manager.sp_group

    # --- Attention ---
    # NPU npu_rms_norm accepts BF16 input + FP32 weight → BF16 output (verified)
    norm_hidden = self.norm1(hidden_states)
    attn = self.attn

    # Q/K/V projections → (B, S/P, H, D)
    query = attn.to_q(norm_hidden).unflatten(-1, (attn.heads, attn.dim_head))
    key = attn.to_k(norm_hidden).unflatten(-1, (attn.heads, attn.dim_head))
    value = attn.to_v(norm_hidden).unflatten(-1, (attn.heads, attn.dim_head))

    # RMSNorm Q/K — BF16 input, affine=False (no weight, falls back to F.rms_norm)
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # RoPE — applied on split sequence with full heads (BEFORE all_to_all)
    # cos/sin shape: (B, S/P, 1, rotary_dim) — broadcasts with (B, S/P, H, D)
    if rotary_emb is not None:
        cos, sin = rotary_emb
        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        rotary_dim = cos.shape[-1]
        query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        query_first, query_second = query_rot.chunk(2, dim=-1)
        key_first, key_second = key_rot.chunk(2, dim=-1)
        query_rotated = torch.cat([-query_second, query_first], dim=-1)
        key_rotated = torch.cat([-key_second, key_first], dim=-1)
        query = torch.cat([query_rot * cos + query_rotated * sin, query_pass], dim=-1)
        key = torch.cat([key_rot * cos + key_rotated * sin, key_pass], dim=-1)

    # Ulysses: (B, S/P, H, D) → (B, S, H/P, D)
    query = all_to_all_before_attn(query, sp_group, scatter_dim=2, gather_dim=1, pad_key="vae_sp")
    key = all_to_all_before_attn(key, sp_group, scatter_dim=2, gather_dim=1, pad_key="vae_sp")
    value = all_to_all_before_attn(value, sp_group, scatter_dim=2, gather_dim=1, pad_key="vae_sp")

    # Attention: (B, S, H/P, D) → reshape (B, H/P, S, D) for SDPA
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    hidden_attn = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    # Back to (B, S, H/P * D)
    hidden_attn = hidden_attn.transpose(1, 2).flatten(2, 3)

    # Ulysses: (B, S, H/P, D) → (B, S/P, H, D) → flatten → (B, S/P, dim)
    hidden_attn = all_to_all_after_attn(
        hidden_attn, sp_group, scatter_dim=1, gather_dim=2, pad_key="vae_sp"
    )

    hidden_attn = attn.to_out[0](hidden_attn)
    hidden_states = hidden_states + hidden_attn * self.scale1

    # --- FFN (local, no communication) ---
    norm_hidden = self.norm2(hidden_states)
    hidden_states = hidden_states + self.ff(norm_hidden) * self.scale2

    return hidden_states


# -------------------------------------------------------------------
# SP-aware ViT Decoder forward
# -------------------------------------------------------------------
_original_decoder_forward = MiniMaxH3VideoViTDecoder3d.forward


def _sp_decoder_forward(
        self: MiniMaxH3VideoViTDecoder3d,
        hidden_states: torch.Tensor,
        parallel_manager=None,
) -> torch.Tensor:
    """ViT decoder forward with optional Ulysses SP.

    *hidden_states*: ``(B, C, T, H, W)`` latent input.

    SP data-flow::

        reshape → tokens (B, T*H*W, C)
        proj_in → (B, N, dim)
        append register + cls → (B, N+5, dim)
        split_sequence → (B, (N+5)/P, dim)          ← each rank
        36× block (SP attention)
        gather_sequence → (B, N+5, dim)
        norm_out → proj_out → unpatchify             ← full sequence
    """
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    # Fall back to module-level injection (used when upstream _decode calls
    # self.decoder() without parallel_manager kwarg).
    if parallel_manager is None:
        parallel_manager = _CURRENT_SP_MANAGER
        sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    if sp_size <= 1:
        return _original_decoder_forward(self, hidden_states)

    sp_group = parallel_manager.sp_group
    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    # --- 1. Reshape: (B,C,T,H,W) → (B, T*H*W, C) ---
    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
        batch_size, num_frames * height * width, num_channels
    )
    num_patches = hidden_states.shape[1]

    # --- 2. proj_in ---
    hidden_states = self.proj_in(hidden_states)

    # --- 3. Append register + cls tokens ---
    register_tokens = self.register_tokens.expand(batch_size, -1, -1)
    cls_token = torch.zeros_like(hidden_states[:, :1, :])
    hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

    # --- 4. RoPE position encoding (full, before split) ---
    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
    suffix_ids = position_ids.new_zeros(
        (batch_size, self.num_register_tokens + 1, 3)
    )
    position_ids = torch.cat([position_ids, suffix_ids], dim=1)
    rotary_emb = self.rope(position_ids)

    # --- 5. Split sequence across SP group ---
    # split_sequence handles padding internally based on the pad amount
    # calculated by PadManager.  No manual padding needed.
    full_seq_len = hidden_states.shape[1]
    pad_mgr = get_pad_manager()
    _pad = pad_mgr.set_from_group("vae_sp", full_seq_len, sp_group)

    hidden_states = split_sequence(hidden_states, sp_group, dim=1, pad=_pad)
    cos = split_sequence(rotary_emb[0], sp_group, dim=1, pad=_pad)
    sin = split_sequence(rotary_emb[1], sp_group, dim=1, pad=_pad)
    rotary_emb = (cos, sin)

    # --- 6. Transformer blocks with SP ---
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, rotary_emb, parallel_manager
            )
        else:
            hidden_states = block(hidden_states, rotary_emb, parallel_manager)

    # --- 7. Gather sequence back (gather_sequence removes padding internally) ---
    hidden_states = gather_sequence(hidden_states, sp_group, dim=1, pad=_pad)

    # --- 8. Output projection + unpatchify (on full sequence) ---
    hidden_states = self.norm_out(hidden_states)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states[:, :num_patches, :]

    patch_size, patch_size_t = self.patch_size, self.patch_size_t
    hidden_states = hidden_states.view(
        batch_size, num_frames, height, width,
        self.out_channels, patch_size_t, patch_size, patch_size,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(
        batch_size, self.out_channels,
        num_frames * patch_size_t, height * patch_size, width * patch_size,
    )


# -------------------------------------------------------------------
# Apply monkey-patches for SP (blocks + decoder)
# -------------------------------------------------------------------
MiniMaxH3VideoTransformerBlock.forward = _sp_block_forward
MiniMaxH3VideoViTDecoder3d.forward = _sp_decoder_forward


# ===================================================================
# Encode / Decode entry points
# ===================================================================

@torch.no_grad()
def _encode_ascend(self, x: torch.Tensor):
    """NPU encode for the MiniMax-H3 video VAE.

    Delegates to the upstream ``_encode``.
    """
    return self._encode(x)


@torch.no_grad()
def encode_ascend(
        self, x: torch.Tensor, return_dict: bool = True
):
    h = self._encode_ascend(x)
    posterior = DiagonalGaussianDistribution(h)
    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)


# ImageNet denormalization constants (module-level to avoid per-call allocation)
_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)
_PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)
_pixel_stats_cache: dict = {}


def _get_pixel_stats(device):
    """Cache device-specific ImageNet normalization constants."""
    if device not in _pixel_stats_cache:
        _pixel_stats_cache[device] = (_PIXEL_MEAN.to(device), _PIXEL_STD.to(device))
    return _pixel_stats_cache[device]


# --- Async D2H infrastructure (only used when VAE_COPY_STREAM=1) ---
_copy_stream = None
_write_executor = ThreadPoolExecutor(max_workers=1)


def _get_copy_stream():
    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.npu.Stream()
    return _copy_stream


def _flush_writes():
    """Block until all pending async frame-write tasks complete.

    No-op when VAE_COPY_STREAM is off (sync path writes inline).
    """
    if not _USE_COPY_STREAM:
        return
    _write_executor.submit(lambda: None).result()


def _stream_frames_to_av(av_stream, chunk):
    """Convert decoded chunk to uint8 HWC frames and stream to SaveAVStream.

    Default (VAE_COPY_STREAM=0): D2H on compute stream, frames written inline.
    Async  (VAE_COPY_STREAM=1): D2H on a separate stream, frames written
    in a background thread so compute is not blocked by CPU-side work.
    """
    if chunk.dim() == 4:
        chunk = chunk.unsqueeze(0)
    mean, std = _get_pixel_stats(chunk.device)
    frames = (chunk.float() * std + mean).clamp(0, 1)
    frames = frames.squeeze(0).permute(1, 2, 3, 0)  # (F, H, W, C)
    frames = (frames * 255).to(torch.uint8)

    if _USE_COPY_STREAM:
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


def _decode_chunks_streaming(vae, z, vae_manager, av_stream, frame_callback=None):
    """Replicate upstream ``_decode`` chunk loop with streaming to SaveAVStream.

    Decodes the latent ``z`` chunk-by-chunk (each chunk = ``tokens_chunk_size``
    latents → ``chunk_num_frames`` pixel frames). After each chunk is blended,
    if ``vae_manager`` is provided, spatial tiles are gathered across ranks.
    Then frames are streamed to ``av_stream`` (rank 0 only).

    Returns the full decoded tensor (same as upstream ``_decode``).
    """
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
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
        clip = vae._decode_clip(z[:, :, start: start + tokens_chunk_size + vae.token_overlap])
        for j in range(int(token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip[:, :, frame_start: frame_start + chunk_num_frames]
            chunk = chunk[:, :, vae.frame_pre_padding:]
            if j == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                # Spatial gather for lightning mode (collective — all ranks must call)
                if vae_manager is not None:
                    chunk = vae_manager.align_out(chunk)
                    chunk = vae_manager.collect_out(chunk)
                decoded_chunks.append(chunk)
                # Stream frames (rank 0 only, av_stream may be None before integration)
                if rank == 0 and av_stream is not None:
                    if frame_callback is None:
                        _stream_frames_to_av(av_stream, chunk)
                    else:
                        frame_callback(chunk)
            else:
                overlap = chunk

    if overlap is not None:
        if vae_manager is not None:
            overlap = vae_manager.align_out(overlap)
            overlap = vae_manager.collect_out(overlap)
        decoded_chunks.append(overlap)
        if rank == 0 and av_stream is not None:
            if frame_callback is None:
                _stream_frames_to_av(av_stream, overlap)
            else:
                frame_callback(overlap)

    dec = torch.cat(decoded_chunks, dim=2)

    if pad_tokens > 0:
        intra_tail = vae.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        dec = dec[:, :, :-pad_frames]
    return dec


# -------------------------------------------------------------------
# SP temporal-chunk decode (mathematically exact parallel decode)
# -------------------------------------------------------------------

def _decode_chunks_sp(vae, z, parallel_manager, av_stream=None):
    """Temporal chunked decode with Ulysses SP + streaming output.

    Same temporal chunking + spatial tiling + blend as upstream
    ``_decode``, but SP is injected into each ViT forward via the
    module-level ``_CURRENT_SP_MANAGER``.

    Each chunk's decoded frames are streamed to ``av_stream`` (rank 0
    only) before the next chunk starts, keeping memory bounded.
    """
    global _CURRENT_SP_MANAGER
    _CURRENT_SP_MANAGER = parallel_manager
    try:
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        temporal_ratio = vae.temporal_compression_ratio
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
            # Upstream _decode_clip does spatial tiling + blend internally.
            # SP is picked up via _CURRENT_SP_MANAGER inside _sp_decoder_forward.
            z_clip = z[:, :, start: start + tokens_chunk_size + vae.token_overlap]
            clip = vae._decode_clip(z_clip)
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start: frame_start + chunk_num_frames]
                chunk = chunk[:, :, vae.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                    if rank == 0 and av_stream is not None:
                        _stream_frames_to_av(av_stream, chunk)
                else:
                    overlap = chunk

        if overlap is not None:
            decoded_chunks.append(overlap)
            if rank == 0 and av_stream is not None:
                _stream_frames_to_av(av_stream, overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        if pad_tokens > 0:
            intra_tail = vae.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec
    finally:
        _CURRENT_SP_MANAGER = None


def _decode_sp_full(vae, z, parallel_manager):
    """Upstream ``_decode`` with SP injected.

    Calls upstream ``_decode`` as-is — temporal chunking + spatial tiling
    + blend are all handled by upstream.  SP is injected transparently:
    the module-level ``_CURRENT_SP_MANAGER`` is picked up by
    ``_sp_decoder_forward`` inside each ``self.decoder(...)`` call.

    Output is identical to single-card upstream ``_decode``, but each
    ViT forward is parallelised across ranks.
    """
    global _CURRENT_SP_MANAGER
    _CURRENT_SP_MANAGER = parallel_manager
    try:
        return vae._decode(z)
    finally:
        _CURRENT_SP_MANAGER = None


def _create_av_stream(vae, z):
    """Create SaveAVStream on rank 0, return None on other ranks."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return None
    audio_path = getattr(infer_info, "audio_path", None)
    vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 8)
    height = z.shape[-2] * vae_spatial_ratio
    width = z.shape[-1] * vae_spatial_ratio
    fps = getattr(infer_info, "fps", None) or 24
    return SaveAVStream(infer_info.save_path, audio_path, fps, width, height)


def _decode_sp(vae, z, parallel_manager):
    """SP multi-card dispatch — selects strategy based on _SP_MODE."""
    if _SP_MODE == "full":
        decoded = _decode_sp_full(vae, z, parallel_manager)
        infer_info._route_b_saved = False
        return decoded

    if _SP_MODE == "vae_mgr":
        vae_manager = VAEManager(
            z.shape,
            factor=vae.spatial_compression_ratio,
            is_encode=False,
            pad_mode=getattr(infer_info, "vae_pad_mode", "all_sides_valid"),
            pad_content=getattr(infer_info, "vae_pad_content", "constant"),
            use_blend=getattr(infer_info, "vae_use_blend", False),
        )
        tile = vae_manager.get_tile_from_x(z)
        av_stream = _create_av_stream(vae, z)
        decoded = _decode_chunks_streaming(vae, tile, vae_manager, av_stream)
        _flush_writes()
        if av_stream is not None:
            av_stream.close()
        infer_info._route_b_saved = True
        return decoded

    # "chunks"
    av_stream = _create_av_stream(vae, z)
    decoded = _decode_chunks_sp(vae, z, parallel_manager, av_stream=av_stream)
    _flush_writes()
    if av_stream is not None:
        av_stream.close()
    infer_info._route_b_saved = True
    return decoded


def _decode_lightning_single(vae, z):
    """Single-card lightning: VAEManager spatial tiling + streaming."""
    vae_manager = VAEManager(
        z.shape,
        factor=vae.spatial_compression_ratio,
        is_encode=False,
        pad_mode=getattr(infer_info, "vae_pad_mode", "all_sides_valid"),
        pad_content=getattr(infer_info, "vae_pad_content", "constant"),
        use_blend=getattr(infer_info, "vae_use_blend", False),
    )
    tile = vae_manager.get_tile_from_x(z)
    av_stream = _create_av_stream(vae, z)
    decoded = _decode_chunks_streaming(vae, tile, vae_manager, av_stream)
    _flush_writes()
    if av_stream is not None:
        av_stream.close()
    infer_info._route_b_saved = True
    return decoded


@torch.no_grad()
def _decode_ascend(self, z: torch.Tensor, return_dict: bool = True):
    """NPU decode dispatch.

    ``vae_lightning`` is the single switch.  Three paths:
    1. Multi-card (lightning + distributed): SP via ``_decode_sp``
    2. Single-card (lightning, no distributed): ``_decode_lightning_single``
    3. Off: upstream ``_decode``
    """
    is_lightning_mode = (
            getattr(infer_info, "vae_lightning", "none")
            in ["decoder", "encoder_and_decoder"]
    )
    is_distributed = torch.distributed.is_initialized()

    if is_lightning_mode and is_distributed:
        decoded = _decode_sp(self, z, _get_vae_sp_manager())
    elif is_lightning_mode:
        decoded = _decode_lightning_single(self, z)
    else:
        decoded = self._decode(z)
        infer_info._route_b_saved = False

    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)


@torch.no_grad()
def decode_ascend(self, z: torch.Tensor, return_dict: bool = True):
    decoded = self._decode_ascend(z).sample
    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)


# -----------------------------------------------------------------------
# Patch VideoProcessor.postprocess_video for MiniMax-H3.
#
# Wan's postprocess_video_ascend (globally monkey-patched onto
# VideoProcessor) returns [None] when lightning=True and
# return_output=False (streaming mode).  That is correct for Wan's
# streaming decode but wrong for MiniMax-H3 SP mode, where the full
# decoded tensor is returned and must be postprocessed normally.
#
# We restore the original upstream postprocess_video logic, bypassing
# Wan's [None] short-circuit.
# -----------------------------------------------------------------------
from diffusers.video_processor import VideoProcessor as _VP


def _mmh3_postprocess_video(self, video, output_type="np", **kwargs):
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = self.postprocess(batch_vid, output_type, **kwargs)
        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


_VP.postprocess_video = _mmh3_postprocess_video

# Monkey-patch the video VAE.
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._encode_ascend = _encode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.encode = encode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._decode_ascend = _decode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.decode = decode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.enable_lightning = enable_lightning
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3 = add_lightning_init(
    AutoencoderKLMiniMaxH3
)

# -----------------------------------------------------------------------
# Patch _decode_video_file: compatibility shim for av >= 12.0.0
#
# PyAV 12.0.0 removed VideoFrame.rotation. diffusers' references.py uses
# frame.rotation in _decode_video_file when loading a reference video
# (ref2va only). Since av.VideoFrame is a Cython extension type, we cannot
# add the property back. Instead, we replace _decode_video_file with a
# version that reads the DISPLAYMATRIX side data directly.
#
# Triggered: ref2va → MiniMaxH3VideoReference.from_file → _decode_video_file
# Unaffected: t2va/fl2va (no reference video, av never called)
# -----------------------------------------------------------------------
import av as _av

if not hasattr(_av.VideoFrame, "rotation"):
    import ctypes as _ctypes
    import math as _math
    import struct as _struct

    import av.sidedata.sidedata as _av_sd
    import numpy as _np

    from diffusers.modular_pipelines.minimax_h3 import references as _refs

    def _read_frame_rotation(frame):
        """Read rotation from DISPLAYMATRIX side data (av >= 12.0.0)."""
        for sd in frame.side_data.keys():
            if sd.type == _av_sd.Type.DISPLAYMATRIX:
                # 3×3 int32 matrix, 16.16 fixed-point, row-major (36 bytes)
                if sd.buffer_size >= 36:
                    raw = _ctypes.string_at(sd.buffer_ptr, 36)
                    matrix = _struct.unpack("<9i", raw)
                    cos_val = matrix[0] / (1 << 16)
                    sin_val = matrix[3] / (1 << 16)
                    angle = -_math.degrees(_math.atan2(sin_val, cos_val))
                    return round(angle / 90.0) * 90.0
        return 0.0

    def _decode_video_file_av12(media):
        """Replacement for references._decode_video_file compatible with av >= 12.0.0."""
        av = _refs._import_av()
        with _refs._local_media_file(media) as path, av.open(path) as container:
            stream = container.streams.video[0]
            frames, rotation = [], 0.0
            for frame in container.decode(stream):
                rotation = _read_frame_rotation(frame)
                frames.append(frame.to_ndarray(format="rgb24"))
            frame_rate = float(stream.average_rate or stream.guessed_rate)
            soundtrack = None
            if container.streams.audio:
                container.seek(0)
                soundtrack = _refs._decode_reference_soundtrack(av, container, container.streams.audio[0])

        if not frames:
            raise ValueError(f"No video frames to decode in {media}.")
        frames = _np.stack(frames)
        turns = round(rotation / 90.0) % 4
        if turns:
            frames = _np.ascontiguousarray(_np.rot90(frames, k=-turns, axes=(1, 2)))
        waveform, sample_rate = soundtrack if soundtrack is not None else (None, None)
        return frames, frame_rate, waveform, sample_rate

    _refs._decode_video_file = _decode_video_file_av12

# -----------------------------------------------------------------------
# Patch MiniMaxH3DecodeStep: Route B swaps VAE decode order (audio first)
# -----------------------------------------------------------------------
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3Blocks,
    MiniMaxH3DecodeStep,
)
from diffusers.modular_pipelines.minimax_h3.decoders import (
    MiniMaxH3VideoDecodeStep,
    MiniMaxH3AudioDecodeStep,
)

_original_decode_step_call = MiniMaxH3DecodeStep.__call__


@torch.no_grad()
def _decode_step_call_route_b(self, pipeline, state):
    is_lightning = (
            getattr(infer_info, "vae_lightning", "none")
            in ["decoder", "encoder_and_decoder"]
    )
    is_distributed = torch.distributed.is_initialized()

    # Non-lightning: original decode order
    # SP "full" mode: no streaming, original decode order
    # SP "chunks" / "vae_mgr" mode + single-card lightning: Route B (audio first → streaming video)
    if not is_lightning:
        return _original_decode_step_call(self, pipeline, state)
    if is_distributed and _SP_MODE == "full":
        return _original_decode_step_call(self, pipeline, state)

    # Route B：先音频后视频
    try:
        # 1. 音频 VAE decode
        audio_block = self.sub_blocks["audio"]
        pipeline, state = audio_block(pipeline, state)

        # 2. 存 WAV → 设 infer_info.audio_path
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        audio = state.get("audio")
        sampling_rate = state.get("sampling_rate")
        if rank == 0 and audio is not None and sampling_rate is not None:
            audio_path = save_audio_to_wav(
                audio, sampling_rate,
                save_path=infer_info.save_path,
            )
            infer_info.audio_path = audio_path
        else:
            infer_info.audio_path = None

        # 3. 视频 VAE decode
        #    _decode_ascend 会读 infer_info.audio_path 创建 SaveAVStream
        #    chunk 流式写入 → close() 自动清理 WAV
        video_block = self.sub_blocks["video"]
        pipeline, state = video_block(pipeline, state)

    finally:
        # 4. 清理（无论成功失败都执行）
        infer_info.audio_path = None

    return pipeline, state


MiniMaxH3DecodeStep.__call__ = _decode_step_call_route_b


# -----------------------------------------------------------------------
# Patch MiniMaxH3Blocks.get_workflow: re-wrap flattened decode blocks
#
# get_execution_blocks() recursively flattens SequentialPipelineBlocks
# containers. MiniMaxH3DecodeStep is dissolved into "decode.video" and
# "decode.audio" leaf blocks, bypassing Route B (which patches
# MiniMaxH3DecodeStep.__call__). This patch intercepts get_workflow's
# return value and re-wraps the leaf blocks back into a
# MiniMaxH3DecodeStep container, restoring Route B in the execution path.
#
# Affected: ref2va (workflow="ref2va" triggers get_workflow)
# Unaffected: t2va/fl2va (workflow=None, get_workflow not called)
# -----------------------------------------------------------------------
_original_get_workflow = MiniMaxH3Blocks.get_workflow


def _get_workflow_rewrap(self, workflow_name):
    blocks = _original_get_workflow(self, workflow_name)

    # Find flattened video/audio leaf blocks
    video_key = audio_key = None
    video_block = audio_block = None
    for key, block in list(blocks.sub_blocks.items()):
        if isinstance(block, MiniMaxH3VideoDecodeStep):
            video_key, video_block = key, block
        elif isinstance(block, MiniMaxH3AudioDecodeStep):
            audio_key, audio_block = key, block

    if video_key and audio_key:
        # Remove flattened leaves
        blocks.sub_blocks.pop(video_key)
        blocks.sub_blocks.pop(audio_key)

        # Re-wrap in MiniMaxH3DecodeStep container (Route B lives here)
        decode_step = MiniMaxH3DecodeStep()
        decode_step.sub_blocks["video"] = video_block
        decode_step.sub_blocks["audio"] = audio_block

        blocks.sub_blocks["decode"] = decode_step

    return blocks


MiniMaxH3Blocks.get_workflow = _get_workflow_rewrap
