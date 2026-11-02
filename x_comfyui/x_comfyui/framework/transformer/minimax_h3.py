"""NPU adaptation for the MiniMax-H3 transformer in ComfyUI.

This module is the ComfyUI equivalent of
``x_diffusers/framework/transformer/minimax_h3.py``.

All patching happens at **import time** — when ComfyUI loads x_comfyui
as a custom node.  No NPUTurbo node call is needed for SP/FSDP/quant;
those are controlled by environment variables set before launch:

  X_SP_SIZE=8       — sequence parallel size
  X_FSDP=true       — enable FSDP
  X_PHAA_SPLIT=4    — PHAA overlap split number
  X_QUANT=int8      — weight/activation quantization
  X_ADALN_CACHE_FILE=/path/to/cache.safetensors  — precomputed AdaLN cache file

The NPUTurbo node provides runtime overrides for 4 NPU acceleration features
(attention mode, PHAA, quant, AdaLN cache).  See ``nodes.py``
for details.

Patches applied:
  1. ``Attention.forward`` → routes through ``attention_manager`` + SP all_to_all
  2. ``AdalnProj.forward`` → AdaLN cache lookup + int8 dtype fallback
  3. ``MiniMaxH3Model._forward`` → SP split/gather around block loop + maybe_install
  4. ``MiniMaxH3Model`` → adds ``set_parallel_manager`` / ``enable_parallel``
  5. ``BaseModel.__init__`` → auto-apply SP + quant after model creation
  6. ``BaseModel.load_model_weights`` → auto-apply FSDP + AdaLN cache after weight loading

ComfyUI's MiniMax-H3 ``Attention`` class (``comfy/ldm/minimax/model.py``):
  - Input shape: ``(S, C)`` — no batch dimension
  - Fused QKV: ``self.qkv_proj`` (single Linear, no bias)
  - RMSNorm: ``self.q_norm``, ``self.k_norm`` (on head_dim)
  - Output: ``self.out_proj`` (single Linear)
  - RoPE: fused with RMSNorm via ``comfy.quant_ops.ck.rms_rope_split_half_``

ComfyUI's ``AdalnProj`` class is used by both DiTBlock (expand=6, has _block_idx)
and FinalLayer (expand=2, no _block_idx).  The unified forward patch checks
``_block_idx`` to distinguish.
"""

import fnmatch
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from safetensors import safe_open
from safetensors.torch import load_file

from x_base import (
    ParallelManager,
    all_to_all_before_attn,
    all_to_all_after_attn,
    attention_manager,
    is_phaa_enabled,
    get_phaa_split_num,
    run_phaa_attention,
    split_sequence,
    gather_sequence,
    set_pad,
    get_pad,
    matmul_manager,
)
from x_base.phaa import enable_phaa, set_phaa_split_num
from x_base.adaln_cache import AdalnScheduleCache
from x_base.config.config_loader import QuantLayerConfig
from x_base.fsdp.fsdp import transformer_lambda_fsdp
from x_base.operator.matmul import WeightQuantLinearModule
from x_comfyui.framework.rope_utils import _fused_rope_minimax_h3

from x_comfyui.adaptor.env import (
    get_sp_size, get_phaa_split, get_quant, get_fsdp_enabled,
    get_adaln_cache_file, get_adaln_transformer_path,
    get_last_quant, set_last_quant,
    get_last_text_tp, set_last_text_tp,
    set_text_tp_override,
)

logger = logging.getLogger("x_comfyui")

# ---------------------------------------------------------------------------
# Import ComfyUI MiniMax-H3 model classes
# ---------------------------------------------------------------------------
import comfy.ldm.minimax.model as mm_model
import comfy.model_base
import comfy.model_management
import comfy.quant_ops
import comfy.ldm.common_dit
import comfy.model_prefetch
import execution

MiniMaxH3Model = mm_model.MiniMaxH3Model
Attention = mm_model.Attention
AdalnProj = mm_model.AdalnProj
DiTBlock = mm_model.DiTBlock
BaseModel = comfy.model_base.BaseModel


# ===========================================================================
# 1. NPU Attention forward — replaces Attention.forward
# ===========================================================================
def _npu_attn_forward(
    self: Attention,
    x: torch.Tensor,
    rope_freqs=None,
    transformer_options: dict = None,
) -> torch.Tensor:
    """NPU attention forward with SP/PHAA support.

    Replaces ``Attention.forward`` to:
      1. Keep the original fused RMSNorm + RoPE (comfy.quant_ops.ck)
      2. Use ``attention_manager.attention`` (sage/laser/sdpa/rainfusion/sparse)
      3. Add SP ``all_to_all`` when ``parallel_manager`` is set
    """
    if transformer_options is None:
        transformer_options = {}

    # --- Existing attention logic ---
    s = x.shape[0]  # (S, C) — no batch dim
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)

    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        rot = rope_freqs.shape[-3] * 2
        # RMSNorm (goes through F.rms_norm → npu_rms_norm via x_base auto-patch)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE (fused or fallback via rope_manager.rope_rotated_half)
        q, k = _fused_rope_minimax_h3(q, k, rope_freqs, rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    v = v.clone()

    # --- Check SP configuration ---
    parallel_manager = getattr(self, "parallel_manager", None)
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    if sp_size > 1:
        sp_group = parallel_manager.sp_group

        # PHAA overlap
        if is_phaa_enabled():
            return _phaa_forward(self, q, k, v, rope_freqs, sp_group, sp_size)

        # Ulysses SP: add batch dim, scatter heads (dim=2), gather sequence (dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        q, k, v = map(
            lambda t: all_to_all_before_attn(t, sp_group, scatter_dim=2, gather_dim=1),
            [q, k, v],
        )
        q = q.transpose(1, 2)  # (1, H/sp, S*sp, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    else:
        q = q.transpose(0, 1).unsqueeze(0)  # (1, heads, S, head_dim)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

    out = attention_manager.attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
    )

    out = out.transpose(1, 2).flatten(2, 3).type_as(q)

    if sp_size > 1:
        out = all_to_all_after_attn(out, sp_group, scatter_dim=1, gather_dim=2)

    return self.out_proj(out.squeeze(0))


# ============================================================================
# 2. PHAA forward — overlap all_to_all with flash attention
# ============================================================================
def _phaa_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rope_freqs,
    sp_group: Any,
    sp_size: int,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    query = query.unsqueeze(0).flatten(-2)
    key = key.unsqueeze(0).flatten(-2)
    value = value.unsqueeze(0).flatten(-2)

    comm_stream = getattr(self, "comm_stream", None)
    if comm_stream is None:
        comm_stream = torch.npu.Stream()
        self.comm_stream = comm_stream

    hidden_states = run_phaa_attention(
        query, key, value,
        comm_stream=comm_stream,
        sp_group=sp_group,
        sp_size=sp_size,
        heads=self.heads,
        phaa_split_num=get_phaa_split_num() or self.heads // sp_size,
        head_dim=head_dim,
        rope_fn=None,
        rotary_emb=rope_freqs,
        attention_mask=None,
        pad_key="comfyui_mmh3_seq",
    )
    return self.out_proj(hidden_states.squeeze(0))


# ===========================================================================
# 3. MiniMaxH3Model method injection
# ===========================================================================
def _set_parallel_manager(self: MiniMaxH3Model, parallel_manager: ParallelManager):
    """Attach ParallelManager so attention forward can read sp_size."""
    self.parallel_manager = parallel_manager
    for block in self.blocks:
        if hasattr(block, "attn"):
            block.attn.parallel_manager = parallel_manager


def _enable_parallel(
    self: MiniMaxH3Model,
    dp_size: int = 1,
    sp_size: int = 1,
    enable_cp: bool = False,
    ulysses_size: int = None,
    ring_size: int = None,
):
    """Enable sequence parallelism on the MiniMax-H3 model."""
    cp_size = 2 if enable_cp and sp_size % 2 == 0 else 1
    if cp_size > 1:
        sp_size = sp_size // 2
    parallel_manager = ParallelManager(
        dp_size, cp_size, sp_size, 1, ulysses_size, ring_size
    )
    self.set_parallel_manager(parallel_manager)


# ===========================================================================
# 4. AdaLN cache: AdalnProj.forward patch (unified for blocks + final layer)
# ===========================================================================
def _patched_adaln_proj_forward(self: AdalnProj, t_emb: torch.Tensor):
    """Patched AdalnProj.forward with AdaLN cache + int8 fallback.

    For DiTBlock adaln_proj (expand=6, has _block_idx):
      - Cache hit → return 6 tensors from cache.lookup_block(idx)
    For FinalLayer adaln_proj (expand=2, no _block_idx):
      - Cache hit → return (shift, scale) from cache.lookup_final()
    On cache miss with file cache loaded → RuntimeError (weights pruned).
    On cache miss without file cache → original logic with int8 dtype fallback.
    """
    cache = getattr(self, "_adaln_cache", None)
    if cache is not None and cache.has_current:
        block_idx = getattr(self, "_block_idx", None)
        if block_idx is not None:
            return cache.lookup_block(block_idx)
        else:
            return cache.lookup_final()

    if cache is not None and cache._file_cache_loaded:
        raise RuntimeError(
            f"AdaLN cache miss — weights have been pruned, cannot compute. "
            f"Block {getattr(self, '_block_idx', 'final')}. "
            f"Check cache file and sampler config."
        )

    # Original logic with int8 dtype fallback
    linear_dtype = self.linear.weight.dtype
    if linear_dtype == torch.int8:
        linear_dtype = torch.bfloat16
    x = self.linear(
        (torch.nn.functional.silu(t_emb) if self.apply_silu else t_emb).to(linear_dtype)
    )
    x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
    return x.chunk(self.expand, dim=-1)


# ===========================================================================
# 5. AdaLN cache: install / release for ComfyUI
# ===========================================================================
def install_adaln_cache_for_comfyui(model, cache_file: str):
    """Create AdalnScheduleCache and inject into model's AdalnProj modules.

    CRITICAL ORDERING: alias model.transformer_blocks = model.blocks
    MUST happen BEFORE creating AdalnScheduleCache, because
    AdalnScheduleCache.__init__ accesses model.transformer_blocks.

    conditioning_type is auto-detected from the cache file:
    if the file contains constant timestep entries (const_timesteps),
    all are pre-merged (conditioning_type="both"); otherwise none
    are pre-merged ("none").

    Args:
        model: MiniMaxH3Model instance.
        cache_file: Path to precomputed safetensors cache file.
    """
    # Auto-detect conditioning_type from cache file
    with safe_open(cache_file, framework="pt") as f:
        has_const = "const_timesteps" in f.keys()
    conditioning_type = "both" if has_const else "none"

    # CRITICAL: alias before cache creation
    if not hasattr(model, "transformer_blocks"):
        model.transformer_blocks = model.blocks

    cache = AdalnScheduleCache(
        model, cache_file=cache_file, conditioning_type=conditioning_type,
    )

    # Inject cache and block index into each block's adaln_proj
    for i, block in enumerate(model.blocks):
        block.adaln_proj._block_idx = i
        block.adaln_proj._adaln_cache = cache

    # Inject cache into final layer's adaln_proj (no _block_idx)
    model.final_layer.adaln_proj._adaln_cache = cache

    # Inject cache into model itself so _patched_forward can call maybe_install
    model._adaln_cache = cache

    logger.info(
        f"x_comfyui: AdaLN cache installed "
        f"(file={cache_file}, conditioning={conditioning_type})"
    )
    return cache


def _try_del_attr(obj, attr):
    """Try to delete an attribute from an object. Returns 1 if deleted, 0 otherwise."""
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        if val is not None:
            delattr(obj, attr)
            return 1
    return 0


def _release_linear_attrs(linear):
    """Release weight/bias/quant attrs from a linear module. Returns freed count."""
    if linear is None:
        return 0
    freed = 0
    for attr in ("weight", "bias", "weight_scale", "weight_quantized"):
        freed += _try_del_attr(linear, attr)
    return freed


def _release_block_adaln_weights(model):
    """Release adaln weights from all transformer blocks. Returns freed count."""
    freed = 0
    for block in model.blocks:
        adaln = getattr(block, "adaln_proj", None)
        if adaln is not None and hasattr(adaln, "linear"):
            freed += _release_linear_attrs(adaln.linear)
    return freed


def _release_final_adaln_weights(model):
    """Release adaln weights from final layer. Returns freed count."""
    freed = 0
    final_adaln = getattr(model.final_layer, "adaln_proj", None)
    if final_adaln is not None and hasattr(final_adaln, "linear"):
        freed += _release_linear_attrs(final_adaln.linear)
    return freed


def _release_time_embedder_weights(model):
    """Release time embedder weights (only for non-curve checkpoints). Returns freed count."""
    if getattr(model, "use_adaln_curves", False):
        return 0
    freed = 0
    time_embedder = getattr(model, "time_embedder", None)
    if time_embedder is None:
        return 0
    for name in ("linear_1", "linear_2"):
        lin = getattr(time_embedder, name, None)
        freed += _release_linear_attrs(lin)
    return freed


def release_adaln_weights_for_comfyui(model: MiniMaxH3Model):
    """Delete adaln-related weight tensors from model to free memory.

    Called after installing cache from a stripped checkpoint.
    The adaln parameters are never used because the cache provides
    all modulation params at runtime.
    """
    freed = _release_block_adaln_weights(model)
    freed += _release_final_adaln_weights(model)
    freed += _release_time_embedder_weights(model)

    logger.info(f"x_comfyui: release_adaln_weights freed {freed} weight tensors")
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()


# ===========================================================================
# 6. SP split/gather: patched _forward with maybe_install
# ===========================================================================
_original_forward = None  # set in patch_model_classes()


def _split_mod_segments(mod_segments, offset, shard_len):
    """Adjust mod_segments for SP shard.

    Each segment is (start, stop, row) where row is a scalar or per-token tensor.
    Straddling segments are clamped: new_start = max(start-offset, 0),
    new_stop = min(stop-offset, shard_len).  Tensor rows are sliced to match.
    """
    new_segments = []
    for a, b, row in mod_segments:
        new_a = max(a - offset, 0)
        new_b = min(b - offset, shard_len)
        if new_b > new_a:
            if isinstance(row, torch.Tensor):
                tensor_start = max(offset - a, 0)
                tensor_len = new_b - new_a
                new_row = row[tensor_start:tensor_start + tensor_len]
            else:
                new_row = row
            new_segments.append((new_a, new_b, new_row))
    return new_segments


def _reorder_mod_segments(mod_segments, reorder_indices, num_ref_rows):
    """Reorder mod_segments to match a reordered sequence.

    After reordering rows so ref_img rows come first, the (start, stop)
    ranges in mod_segments no longer match.  This rebuilds the segment
    list by mapping each old row index to its mod_row value, then
    re-indexing via reorder_indices and merging adjacent runs with
    the same mod_row.
    """
    S = reorder_indices.shape[0]
    row_to_mod = torch.zeros(S, dtype=torch.long, device=reorder_indices.device)
    for start, stop, mod_row in mod_segments:
        row_to_mod[start:stop] = mod_row
    row_to_mod = row_to_mod[reorder_indices]
    # Vectorized boundary detection — O(1) NPU ops instead of O(S) Python loop
    if S <= 1:
        return [(0, S, row_to_mod[0].item())] if S == 1 else []
    # Find positions where the mod value changes
    changed = row_to_mod[1:] != row_to_mod[:-1]  # boolean tensor, no sync
    boundaries = torch.nonzero(changed, as_tuple=False).flatten() + 1  # sync once
    starts = torch.cat([torch.tensor([0], device=reorder_indices.device), boundaries])
    ends = torch.cat([boundaries, torch.tensor([S], device=reorder_indices.device)])
    mod_values = row_to_mod[starts]
    # Convert to Python list — only O(num_segments) iterations, not O(S)
    starts_list = starts.tolist()
    ends_list = ends.tolist()
    mod_list = mod_values.tolist()
    return [(s, e, m) for s, e, m in zip(starts_list, ends_list, mod_list)]


def _compute_timesteps(self, timestep, transformer_options):
    """Compute video and audio timesteps from input."""
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - mm_model.time_shift_sigma(sigma_v, shift_v, shift_a))
    return t_v, t_a, sigma_v


def _install_adaln_cache_table(self, t_v, t_a):
    """Install AdaLN cache table for current timestep."""
    _adaln_cache = getattr(self, "_adaln_cache", None)
    if _adaln_cache is None:
        return
    step_ts = sorted(set([t_v, t_a]))
    const_indices = _adaln_cache._CONST_INDEX_MAP.get(_adaln_cache._conditioning_type, [])
    if const_indices and _adaln_cache._const_timesteps is not None:
        const_ts = _adaln_cache._const_timesteps[const_indices].tolist()
        full_ts = step_ts + const_ts
    else:
        full_ts = step_ts
    ts_tensor = torch.tensor(full_ts, dtype=torch.float32)
    _adaln_cache.maybe_install(ts_tensor)


def _build_layout(self, payload, video_x, audio_x, context):
    """Build or retrieve PackedLayout from payload."""
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = mm_model.PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    return layout, latent_t, lat_h, lat_w


def _compute_seg_t_and_mask_rows(self, payload, t_v, t_a, denoise_mask, audio_denoise_mask,
                                  latent_t, lat_h, lat_w, sigma_v):
    """Compute segment timesteps and per-row timestep tensors."""
    vis_aug = float(payload.get("visual_cond_noise_aug", mm_model.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", mm_model.AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    t_pin_v = max(t_v, mm_model.VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, mm_model.AUDIO_COND_TIMESTEP)
    video_rows_t = _compute_mask_rows(denoise_mask, sigma_v, t_pin_v, latent_t, lat_h, lat_w)
    audio_rows_t = _compute_audio_mask_rows(audio_denoise_mask, t_a, t_pin_a)
    if video_rows_t is not None and video_rows_t.unique().numel() == 1:
        seg_t["video"] = float(video_rows_t[0])
        video_rows_t = None
    if audio_rows_t is not None and audio_rows_t.unique().numel() == 1:
        seg_t["audio"] = float(audio_rows_t[0])
        audio_rows_t = None
    return seg_t, video_rows_t, audio_rows_t


def _compute_mask_rows(denoise_mask, sigma_v, t_pin_v, latent_t, lat_h, lat_w):
    """Compute per-row timesteps from video denoise mask."""
    if denoise_mask is None:
        return None
    m = mm_model.mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
    if m is None:
        return None
    return (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)


def _compute_audio_mask_rows(audio_denoise_mask, t_a, t_pin_a):
    """Compute per-row timesteps from audio denoise mask."""
    if audio_denoise_mask is None:
        return None
    m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
    if bool((m >= 1.0 - 1e-3).all()):
        return None
    sigma_a = 1.0 - t_a
    return (1.0 - m * sigma_a).clamp(max=t_pin_a)


def _build_unique_t_and_mod_segments(layout, seg_t, video_rows_t, audio_rows_t, t_v, t_a, payload):
    """Build unique timestep list and modulation segments."""
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                      | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                      | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}
    text_tags = payload.get("text_token_tags")
    mod_segments = _build_mod_segments(layout, seg_t, t_row, seg_tag, video_rows_t, audio_rows_t, text_tags)
    return unique_t, t_row, mod_segments


def _build_mod_segments(layout, seg_t, t_row, seg_tag, video_rows_t, audio_rows_t, text_tags):
    """Build modulation segments from layout and timesteps."""
    def rows_to_mod_index(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],
                            dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            mod_segments.extend(_build_text_segments(a, b, row_base, text_tags))
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))
    return mod_segments


def _build_text_segments(a, b, row_base, text_tags):
    """Build modulation segments for text tokens with tag runs."""
    tags = text_tags.view(-1).tolist()
    segments = []
    run_start = 0
    for i in range(1, b - a + 1):
        if i == b - a or tags[i] != tags[run_start]:
            segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
            run_start = i
    return segments


def _embed_inputs(self, layout, video_x, audio_x, context, payload, device, dtype, transformer_options):
    """Embed video/audio/text inputs and pack into hidden states tensor."""
    video_embed, audio_embed, text_states = _compute_embeddings(
        self, layout, video_x, audio_x, context, payload, device, dtype, transformer_options)
    h = _pack_hidden_states(self, layout, video_embed, audio_embed, text_states, device, dtype)
    return h


def _compute_embeddings(self, layout, video_x, audio_x, context, payload, device, dtype, transformer_options):
    """Compute video/audio/text embeddings."""
    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = mm_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = mm_model.pack_audio(audio_x.to(torch.float32))
    all_video_rows = _merge_cond_rows(video_rows, img_update, self._cond_video_rows(payload, device), device)
    all_audio_rows = _merge_cond_rows(audio_rows, audio_update, self._cond_audio_rows(payload, device), device)
    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states),
                                          transformer_options=transformer_options)
    return video_embed, audio_embed, text_states


def _merge_cond_rows(base_rows, update_mask, cond_rows, device):
    """Merge conditional rows with base rows using update mask."""
    if cond_rows is None:
        return base_rows
    all_rows = torch.empty(update_mask.shape[0], base_rows.shape[1], dtype=torch.float32, device=device)
    all_rows[~update_mask] = cond_rows
    all_rows[update_mask] = base_rows
    return all_rows


def _pack_hidden_states(self, layout, video_embed, audio_embed, text_states, device, dtype):
    """Pack embeddings into hidden states tensor according to layout segments."""
    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n
    return h


def _compute_t_emb(self, unique_t, device, dtype):
    """Compute time embedding from unique timesteps."""
    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        return _compute_t_emb_curves(self, t_vals, device)
    _adaln_cache = getattr(self, "_adaln_cache", None)
    if _adaln_cache is not None and _adaln_cache.has_current:
        return None  # skip time_embedder when cache is active
    return self.time_embedder(t_vals).to(dtype)


def _compute_t_emb_curves(self, t_vals, device):
    """Compute time embedding using AdaLN curves table."""
    table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


def _compute_reorder_indices(self, layout, sp_size, h, device):
    """Compute reorder indices for ref video KV cache."""
    num_condition_video_rows = sum(b - a for a, b, k in layout.segments if k == "ref_img")
    ref_indices, non_ref_indices = _split_ref_non_ref_indices(layout, device)
    _per_chunk_layout = (sp_size > 1
                         and num_condition_video_rows % sp_size == 0
                         and h.shape[0] % sp_size == 0)
    if _per_chunk_layout:
        reorder_indices, _ref_positions_global = _compute_per_chunk_reorder(
            ref_indices, non_ref_indices, sp_size, h.shape[0], num_condition_video_rows, device)
    else:
        reorder_indices = torch.cat([ref_indices, non_ref_indices])
        _ref_positions_global = torch.arange(num_condition_video_rows, device=device, dtype=torch.long)
    inverse_indices = torch.argsort(reorder_indices)
    self._ref_reorder_indices = reorder_indices
    self._ref_inverse_indices = inverse_indices
    self._ref_cached_num_ref_rows = num_condition_video_rows
    self._per_chunk_layout = _per_chunk_layout
    self._ref_positions_global = _ref_positions_global
    return reorder_indices, inverse_indices, num_condition_video_rows


def _split_ref_non_ref_indices(layout, device):
    """Split layout segments into ref and non-ref indices."""
    ref_indices = []
    non_ref_indices = []
    for a, b, kind in layout.segments:
        if kind == "ref_img":
            ref_indices.extend(range(a, b))
        else:
            non_ref_indices.extend(range(a, b))
    return (torch.tensor(ref_indices, device=device, dtype=torch.long),
            torch.tensor(non_ref_indices, device=device, dtype=torch.long))


def _compute_per_chunk_reorder(ref_indices, non_ref_indices, sp_size, seq_len, num_ref, device):
    """Compute per-chunk interleaved reorder indices for SP."""
    if (seq_len - num_ref) % sp_size != 0:
        raise ValueError(
            f"Per-chunk layout requires (seq_len - num_ref) % sp_size == 0, "
            f"got seq_len={seq_len}, num_ref={num_ref}, sp_size={sp_size}"
        )
    N_ref_per_chunk = num_ref // sp_size
    M_per_chunk = (seq_len - num_ref) // sp_size
    ref_chunks = torch.split(ref_indices, N_ref_per_chunk)
    non_ref_chunks = torch.split(non_ref_indices, M_per_chunk)
    parts = []
    for i in range(sp_size):
        parts.append(ref_chunks[i])
        parts.append(non_ref_chunks[i])
    reorder_indices = torch.cat(parts)
    S_local = seq_len // sp_size
    ref_positions = []
    for i in range(sp_size):
        ref_positions.extend(range(i * S_local, i * S_local + N_ref_per_chunk))
    _ref_positions_global = torch.tensor(ref_positions, device=device, dtype=torch.long)
    return reorder_indices, _ref_positions_global


def _apply_reorder(self, h, rope_freqs, mod_segments, layout, sp_size, device):
    """Apply reorder for ref video KV cache."""
    num_condition_video_rows = sum(b - a for a, b, k in layout.segments if k == "ref_img")
    _cached_reorder = getattr(self, "_ref_reorder_indices", None)
    _cached_num_ref = getattr(self, "_ref_cached_num_ref_rows", None)
    if (_cached_reorder is not None
            and _cached_num_ref == num_condition_video_rows
            and _cached_reorder.shape[0] == h.shape[0]):
        reorder_indices = _cached_reorder
        inverse_indices = getattr(self, "_ref_inverse_indices")
    else:
        reorder_indices, inverse_indices, num_condition_video_rows = \
            _compute_reorder_indices(self, layout, sp_size, h, device)
    h = h[reorder_indices]
    rope_freqs = rope_freqs[:, reorder_indices]
    mod_segments = _reorder_mod_segments(mod_segments, reorder_indices, num_condition_video_rows)
    return h, rope_freqs, mod_segments, inverse_indices, num_condition_video_rows


def _sp_split_before_blocks(self, h, rope_freqs, mod_segments, sp_group):
    """SP split before block loop."""
    set_pad("comfyui_mmh3_seq", h.shape[0], sp_group)
    _pad = get_pad("comfyui_mmh3_seq")
    h = split_sequence(h, sp_group, dim=0, pad=_pad)
    shard_len = h.shape[0]
    offset = sp_group.rank() * shard_len
    rope_freqs = split_sequence(rope_freqs, sp_group, dim=1, pad=_pad)
    mod_segments = _split_mod_segments(mod_segments, offset, shard_len)
    return h, rope_freqs, mod_segments, _pad


def _run_block_loop(self, h, t_emb, mod_segments, rope_freqs, device, transformer_options):
    """Run the transformer block loop."""
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        h = _run_single_block(block, h, t_emb, mod_segments, rope_freqs, transformer_options, blocks_replace, i, prefetch_queue, device)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return h


def _run_single_block(block, h, t_emb, mod_segments, rope_freqs, transformer_options, blocks_replace, i, prefetch_queue, device):
    """Run a single transformer block with optional replacement."""
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
    if ("double_block", i) in blocks_replace:
        return _run_replaced_block(block, h, t_emb, mod_segments, rope_freqs, transformer_options, blocks_replace, i)
    return block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)


def _run_replaced_block(block, h, t_emb, mod_segments, rope_freqs, transformer_options, blocks_replace, i):
    """Run a block with replacement from patches_replace."""
    def block_wrap(args):
        return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                             transformer_options=args["transformer_options"])}
    return blocks_replace[("double_block", i)](
        {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
         "transformer_options": transformer_options},
        {"original_block": block_wrap})["img"]


def _compute_final_output(self, h, t_emb, layout, seg_t, t_row, video_rows_t, audio_rows_t,
                           latent_t, lat_h, lat_w, orig_t, orig_h, orig_w, video_x, audio_x):
    """Compute final layer output and unpatchify."""
    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")

    def rows_to_mod_index_final(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],
                            dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    video_seg = _build_video_seg(va, vb, video_rows_t, rows_to_mod_index_final, t_row, seg_t)
    audio_seg = _build_audio_seg(aa, ab, audio_rows_t, rows_to_mod_index_final, t_row, seg_t)
    v, a = self.final_layer(h, t_emb, video_seg, audio_seg)
    video_out = mm_model.unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = mm_model.unpack_audio(a)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def _build_video_seg(va, vb, video_rows_t, rows_to_mod_index, t_row, seg_t):
    """Build video segment for final layer."""
    if video_rows_t is not None:
        return (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)
    return (va, vb, t_row[seg_t["video"]])


def _build_audio_seg(aa, ab, audio_rows_t, rows_to_mod_index, t_row, seg_t):
    """Build audio segment for final layer."""
    if audio_rows_t is not None:
        return (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)
    return (aa, ab, t_row[seg_t["audio"]])


def _patched_forward(self: MiniMaxH3Model, x, timestep, context,
                     transformer_options=None, minimax_payload=None,
                     denoise_mask=None, audio_denoise_mask=None, **kwargs):
    """Patched _forward with AdaLN cache maybe_install + SP split/gather."""
    if transformer_options is None:
        transformer_options = {}
    payload = minimax_payload or {}
    t_v, t_a, sigma_v = _compute_timesteps(self, timestep, transformer_options)
    _install_adaln_cache_table(self, t_v, t_a)
    parallel_manager = getattr(self, "parallel_manager", None)
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1
    if sp_size <= 1:
        return _original_forward(
            self, x, timestep, context, transformer_options,
            minimax_payload, denoise_mask, audio_denoise_mask, **kwargs
        )
    sp_group = parallel_manager.sp_group if parallel_manager is not None else None
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    device = video_x.device
    dtype = context.dtype
    layout, latent_t, lat_h, lat_w = _build_layout(self, payload, video_x, audio_x, context)
    seg_t, video_rows_t, audio_rows_t = _compute_seg_t_and_mask_rows(
        self, payload, t_v, t_a, denoise_mask, audio_denoise_mask, latent_t, lat_h, lat_w, sigma_v)
    unique_t, t_row, mod_segments = _build_unique_t_and_mod_segments(
        layout, seg_t, video_rows_t, audio_rows_t, t_v, t_a, payload)
    h = _embed_inputs(self, layout, video_x, audio_x, context, payload, device, dtype, transformer_options)
    t_emb = _compute_t_emb(self, unique_t, device, dtype)
    rope_freqs = mm_model.rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
    if sp_size > 1:
        h, rope_freqs, mod_segments, _pad = _sp_split_before_blocks(self, h, rope_freqs, mod_segments, sp_group)
    else:
        _pad = None
    h = _run_block_loop(self, h, t_emb, mod_segments, rope_freqs, device, transformer_options)
    if sp_size > 1:
        h = gather_sequence(h, sp_group, dim=0, pad=_pad)
    return _compute_final_output(self, h, t_emb, layout, seg_t, t_row, video_rows_t, audio_rows_t,
                                  latent_t, lat_h, lat_w, orig_t, orig_h, orig_w, video_x, audio_x)


# ===========================================================================
# 7. Post-load hooks: quant, FSDP, AdaLN stripping
# ===========================================================================

def _cast_quant_params_to_input_device(module, input):
    """forward_pre_hook: cast WeightQuantLinearModule params to input device.

    ComfyUI's lowvram/dynamic offload may leave quantized params on CPU.
    cast_to is a no-op when devices already match.
    """
    dev = input[0].device
    module.weight.data = comfy.model_management.cast_to(module.weight.data, device=dev)
    module.antiquant_scale.data = comfy.model_management.cast_to(module.antiquant_scale.data, device=dev)
    if module.bias is not None:
        module.bias.data = comfy.model_management.cast_to(module.bias.data, device=dev)


def _apply_sp_and_quant(base_model: BaseModel):
    """Apply SP and AdaLN cache to diffusion_model after creation.

    Called from patched BaseModel.__init__.  Reads environment variables.
    Only affects MiniMax-H3 models; silently skips others.

    Quantization is NOT applied here — it runs later in _apply_quant()
    from _patched_load_weights, after checkpoint weights are loaded,
    so WeightQuantLinearModule can compute correct antiquant_scale
    from real weights (aligned with x_diffusers' approach).
    """
    diffusion_model = base_model.diffusion_model
    model_type = type(diffusion_model).__name__
    if "MiniMax" not in model_type:
        return

    # --- SP ---
    sp_size = get_sp_size()
    if sp_size > 1:
        if dist.is_initialized():
            phaa_split = get_phaa_split()
            if phaa_split > 1:
                enable_phaa()
                set_phaa_split_num(phaa_split)

            parallel_manager = ParallelManager(
                dp_size=1, cp_size=1, sp_size=sp_size, tp_size=1
            )
            diffusion_model.set_parallel_manager(parallel_manager)
            logger.info(
                f"x_comfyui: SP enabled (sp_size={sp_size}, phaa_split={phaa_split})"
            )

    # --- AdaLN cache ---
    cache_file = get_adaln_cache_file()
    if cache_file:
        install_adaln_cache_for_comfyui(diffusion_model, cache_file)
        diffusion_model._npu_adaln_strip = True
        logger.info(
            "x_comfyui: AdaLN cache installed "
            "(adaln keys will be stripped at weight loading)"
        )


def _apply_quant(base_model: BaseModel):
    """Apply int8 quantization to diffusion_model after weight loading.

    Called from patched BaseModel.load_model_weights, AFTER checkpoint
    weights are loaded.  This ensures WeightQuantLinearModule.__init__
    receives actual checkpoint weights and computes correct antiquant_scale.

    Aligns with x_diffusers' approach (load_pipe.py line 414-416) where
    quant is applied after the model is fully loaded.
    """
    diffusion_model = base_model.diffusion_model
    model_type = type(diffusion_model).__name__
    if "MiniMax" not in model_type:
        return

    quant = get_quant()
    if quant == "int8":
        _cfg_path = os.path.join(os.path.dirname(__file__), "quant_config_comfyui.yaml")
        _quant_config = QuantLayerConfig.from_yaml(_cfg_path)
        base_model.diffusion_model = matmul_manager.enable_online_dynamic_quant(
            diffusion_model, quant_config=_quant_config, dtype="w8a8"
        )
        logger.info("x_comfyui: int8 quantization enabled (after weight loading)")

        # Register forward_pre_hook to cast quantized params to input device.
        for module in base_model.diffusion_model.modules():
            if isinstance(module, WeightQuantLinearModule):
                module.register_forward_pre_hook(_cast_quant_params_to_input_device)
        logger.info("x_comfyui: device-cast hooks registered for quantized modules")


def _apply_fsdp(base_model: BaseModel):
    """Apply FSDP to diffusion_model after weight loading.

    Called from patched BaseModel.load_model_weights.  Reads X_FSDP env var.
    Only affects MiniMax-H3 models; silently skips others.
    """
    diffusion_model = base_model.diffusion_model
    model_type = type(diffusion_model).__name__
    if "MiniMax" not in model_type:
        return

    if not get_fsdp_enabled():
        return

    if not dist.is_initialized():
        logger.warning("x_comfyui: FSDP requested but torch.distributed not initialised")
        return

    quant = get_quant()
    device_id = torch.npu.current_device()
    base_model.diffusion_model = transformer_lambda_fsdp(
        diffusion_model,
        device_id,
        is_need_dtype=(quant != "int8"),
    )
    logger.info("x_comfyui: FSDP enabled")


def _clear_clip_related_cache(executor, prompt):
    """Clear cache entries only for CLIPLoader nodes and their dependents.

    Unlike clearing the entire cache, this preserves VAE/UNet cache entries,
    so only the text encoder and nodes depending on it are reloaded.
    Used for text_tp changes.
    """
    cache = executor.caches.outputs
    if not getattr(cache, "initialized", False):
        return

    # 1. Find all CLIPLoader node IDs
    clip_loader_ids = set()
    for node_id, node_def in prompt.items():
        if isinstance(node_def, dict) and node_def.get("class_type") == "CLIPLoader":
            clip_loader_ids.add(node_id)

    if not clip_loader_ids:
        return

    # 2. Find all transitive dependents (BFS)
    affected_ids = set(clip_loader_ids)
    changed = True
    while changed:
        changed = False
        for node_id, node_def in prompt.items():
            if node_id in affected_ids:
                continue
            if not isinstance(node_def, dict):
                continue
            for input_val in node_def.get("inputs", {}).values():
                if isinstance(input_val, list) and len(input_val) == 2:
                    ancestor_id = str(input_val[0])
                    if ancestor_id in affected_ids:
                        affected_ids.add(node_id)
                        changed = True
                        break

    # 3. Clear cache entries for affected nodes
    for node_id in affected_ids:
        key = cache.cache_key_set.get_data_key(node_id)
        if key is not None and key in cache.cache:
            del cache.cache[key]
        subcache_key = cache.cache_key_set.get_subcache_key(node_id)
        if subcache_key is not None and subcache_key in cache.subcaches:
            del cache.subcaches[subcache_key]

    logger.info(
        f"x_comfyui: cleared cache for {len(affected_ids)} nodes "
        f"(CLIPLoader + dependents)"
    )


# ===========================================================================
# 8. Public API: patch_model_classes()
# ===========================================================================
def patch_model_classes():
    """Patch ComfyUI model classes for MiniMax-H3 NPU acceleration.

    Call this once at import time (from x_comfyui.__init__).
    Idempotent — safe to call multiple times.

    Patches:
      - ``Attention.forward`` → NPU attention with SP/PHAA
      - ``AdalnProj.forward`` → AdaLN cache lookup + int8 dtype fallback
      - ``MiniMaxH3Model._forward`` → SP split/gather + maybe_install
      - ``MiniMaxH3Model`` → adds set_parallel_manager / enable_parallel
      - ``BaseModel.__init__`` → auto-apply SP + AdaLN cache
      - ``BaseModel.load_model_weights`` → auto-apply quant + FSDP + release adaln weights
      - ``PromptExecutor.execute_async`` → detect quant change, clear cache for reload
    """
    # Idempotency guard
    if getattr(Attention, "_x_comfyui_patched", False):
        return
    Attention._x_comfyui_patched = True

    # Patch Attention.forward (class-level)
    Attention.forward = _npu_attn_forward

    # Patch AdalnProj.forward (class-level, unified for blocks + final)
    AdalnProj.forward = _patched_adaln_proj_forward

    # Patch MiniMaxH3Model._forward (SP split/gather + maybe_install)
    global _original_forward
    _original_forward = MiniMaxH3Model._forward
    MiniMaxH3Model._forward = _patched_forward

    # Patch MiniMaxH3Model methods
    MiniMaxH3Model.set_parallel_manager = _set_parallel_manager
    MiniMaxH3Model.enable_parallel = _enable_parallel

    # Patch BaseModel.__init__ for auto SP + quant + AdaLN cache
    _original_base_init = BaseModel.__init__

    def _patched_base_init(self, *args, **kwargs):
        _original_base_init(self, *args, **kwargs)
        if hasattr(self, "diffusion_model"):
            _apply_sp_and_quant(self)

    BaseModel.__init__ = _patched_base_init

    # Patch BaseModel.load_model_weights for auto FSDP + adaln stripping
    _original_load_weights = BaseModel.load_model_weights

    def _patched_load_weights(self, sd, unet_prefix="", assign=False):
        # If AdaLN cache is active, strip adaln weights to save ~26GB.
        diffusion_model = getattr(self, "diffusion_model", None)
        if (diffusion_model is not None
                and getattr(diffusion_model, "_npu_adaln_strip", False)):
            transformer_path = get_adaln_transformer_path()
            if transformer_path:
                # Strategy 1: load pre-stripped checkpoint
                if os.path.isfile(transformer_path):
                    sd = load_file(transformer_path)
                elif os.path.isdir(transformer_path):
                    sd = {}
                    for fname in sorted(os.listdir(transformer_path)):
                        if fname.endswith(".safetensors"):
                            sd.update(load_file(
                                os.path.join(transformer_path, fname)))
                else:
                    raise FileNotFoundError(
                        f"X_ADALN_TRANSFORMER_PATH not found: "
                        f"{transformer_path}")
                logger.info(
                    f"x_comfyui: loaded stripped checkpoint from "
                    f"{transformer_path} ({len(sd)} keys)"
                )
            else:
                # Strategy 2: filter adaln keys from original state dict
                ADALN_PATTERNS = [
                    "*blocks.*.adaln_proj.linear.weight",
                    "*blocks.*.adaln_proj.linear.bias",
                    "*final_layer.adaln_proj.linear.weight",
                    "*final_layer.adaln_proj.linear.bias",
                ]
                if not getattr(diffusion_model, "use_adaln_curves", False):
                    ADALN_PATTERNS.extend([
                        "*time_embedder.linear_1.weight",
                        "*time_embedder.linear_1.bias",
                        "*time_embedder.linear_2.weight",
                        "*time_embedder.linear_2.bias",
                    ])
                sd = {k: v for k, v in sd.items()
                      if not any(fnmatch.fnmatch(k, p)
                                 for p in ADALN_PATTERNS)}
                logger.info(
                    f"x_comfyui: filtered adaln keys from state dict "
                    f"({len(sd)} keys remaining)"
                )

        result = _original_load_weights(self, sd, unet_prefix, assign)
        # Apply quantization AFTER weight loading
        if not getattr(self, "_npu_quant_applied", False):
            _apply_quant(self)
            self._npu_quant_applied = True
        if not getattr(self, "_npu_fsdp_applied", False):
            _apply_fsdp(self)
            self._npu_fsdp_applied = True
        # Release any remaining adaln weights after loading
        diffusion_model = getattr(self, "diffusion_model", None)
        if diffusion_model is not None and hasattr(diffusion_model, "_adaln_cache"):
            release_adaln_weights_for_comfyui(diffusion_model)
        return result

    BaseModel.load_model_weights = _patched_load_weights

    # Patch PromptExecutor.execute_async to detect quant/text changes between runs.
    _original_execute_async = execution.PromptExecutor.execute_async

    async def _patched_execute_async(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        if extra_data is None:
            extra_data = {}
        if execute_outputs is None:
            execute_outputs = []
        # Scan prompt for NPUTurbo values before any node executes.
        prompt_quant = None
        prompt_text_tp = None
        for node_id, node_def in prompt.items():
            if isinstance(node_def, dict) and node_def.get("class_type") == "NPUTurbo":
                inputs = node_def.get("inputs", {})
                prompt_quant = inputs.get("quant", "disable")
                prompt_text_tp = inputs.get("text_tp", "disable")
                logger.info(
                    f"x_comfyui: NPUTurbo found in prompt (node {node_id}): "
                    f"quant={prompt_quant}, "
                    f"text_tp={prompt_text_tp}"
                )

        if prompt_quant is None:
            logger.debug("x_comfyui: NPUTurbo not found in prompt, using env var defaults")

        if prompt_quant is not None:
            last_quant = get_last_quant()
            if last_quant is not None and last_quant != prompt_quant:
                self.caches.outputs.cache.clear()
                self.caches.outputs.subcaches.clear()
                logger.info(
                    f"x_comfyui: quant changed ({last_quant} → {prompt_quant}), "
                    f"clearing model cache for reload"
                )
            set_last_quant(prompt_quant)

        # Set text_tp override before any node executes.
        if prompt_text_tp is not None:
            set_text_tp_override(prompt_text_tp == "enable")
            logger.info(f"x_comfyui: text_tp override set to {prompt_text_tp == 'enable'}")
            last_text_tp = get_last_text_tp()
            if last_text_tp is not None and last_text_tp != prompt_text_tp:
                _clear_clip_related_cache(self, prompt)
                logger.info(
                    f"x_comfyui: text_tp changed ({last_text_tp} → {prompt_text_tp}), "
                    f"clearing CLIP cache for reload"
                )
            set_last_text_tp(prompt_text_tp)

        await _original_execute_async(self, prompt, prompt_id, extra_data, execute_outputs)

    execution.PromptExecutor.execute_async = _patched_execute_async

    logger.info("x_comfyui: MiniMax-H3 model patches applied (Attention, AdalnProj, _forward, BaseModel)")

