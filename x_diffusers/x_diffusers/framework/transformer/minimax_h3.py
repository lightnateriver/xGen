"""NPU adaptation for the MiniMax-H3 transformer.

This module replaces the upstream ``MiniMaxH3AttnProcessor`` with
``AscendMiniMaxH3AttnProcessor``, which calls ``attention_manager.attention``
directly — the same x_base interface used by the Wan and HunyuanVideo
adaptations.  This gives MiniMax-H3 immediate access to all x_base attention
backends (sage, laser, rainfusion, ada_sparse) without going through
``dispatch_attention_fn``.

Sequence-parallel (SP) support is added via the same
``all_to_all_before_attn`` / ``all_to_all_after_attn`` pair that the Wan
adaptation uses.  When ``sp_size == 1`` the processor is a straight pass-through
to ``attention_manager.attention``.
"""

import logging
import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
from x_base.device import device_type

if device_type() == "npu":
    import torch_npu  # noqa: F401  – registers NPU kernels
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
import torch.nn as nn

from diffusers.models.transformers import transformer_minimax_h3
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    MiniMaxH3TransformerOutput,
    _apply_rotary_emb,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.utils import apply_lora_scale

from x_base import (
    ParallelManager,
    all_to_all_before_attn,
    all_to_all_after_attn,
    attention_manager,
    gather_sequence,
    get_pad,
    rope_manager,
    set_pad,
    split_sequence,
    is_phaa_enabled,
    get_phaa_split_num,
    run_phaa_attention,
)


def _rope_minimax_h3(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for MiniMax-H3: partial rotation (96/128 channels) with rotate_half convention.

    MiniMax-H3 rotates only the first ``rotary_dim`` channels (96 of 128) using
    the rotate_half convention.  The remaining 32 channels pass through unchanged.

    Routes through ``rope_manager`` for the fused NPU path when available,
    falling back to the reference ``_apply_rotary_emb`` implementation otherwise.
    """
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts.
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    if rope_manager.is_rope_fused:
        # Try fused mindie backend with rotate_half mode.
        try:
            from x_base.operator.rope import AVAILABLE_MINDIESD, rotary_position_embedding

            if AVAILABLE_MINDIESD:
                q_rot = rotary_position_embedding(
                    q_rot.contiguous(), cos, sin, rotated_mode="rotated_half", fused=True
                )
                k_rot = rotary_position_embedding(
                    k_rot.contiguous(), cos, sin, rotated_mode="rotated_half", fused=True
                )
                return (
                    torch.cat([q_rot, q_pass], dim=-1),
                    torch.cat([k_rot, k_pass], dim=-1),
                )
        except Exception:
            pass  # Fall through to base implementation.

    # Base implementation: rotate_half convention (reference).
    query_out = _apply_rotary_emb(query, cos, sin)
    key_out = _apply_rotary_emb(key, cos, sin)
    return query_out, key_out


class AscendMiniMaxH3AttnProcessor:
    """NPU attention processor for MiniMax-H3 with optional SP support.

    When ``attn.parallel_manager`` is set and ``sp_size > 1``, the heads are
    scattered across the SP group (Ulysses pattern) before attention and
    gathered back afterwards.  Otherwise the call is forwarded directly to
    ``attention_manager.attention`` — the same x_base interface that the Wan
    and HunyuanVideo adaptations use — so that sage / laser / rainfusion /
    ada_sparse attention are all available through the global
    ``attention_manager`` configuration.
    """

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- projections (identical to the upstream processor) ---
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        normalized = None
        fused_rope = getattr(attn, "_h3_fused_partial_rope", False)
        if getattr(attn, "_h3_fused_qk_rms", False):
            from x_base.operator.h3_qk_rms import try_h3_qk_rms

            normalized = try_h3_qk_rms(
                query, key, attn.norm_q, attn.norm_k, rotary_emb if fused_rope else None,
            )
        if normalized is None:
            query, key = attn.norm_q(query), attn.norm_k(key)
        else:
            query, key = normalized

        # RoPE applied before SP branch (original position).
        # all_to_all doesn't change (position, head) mapping, so RoPE stays valid.
        if rotary_emb is not None and not (fused_rope and normalized is not None):
            rotated = None
            if fused_rope:
                from x_base.operator.h3_partial_rope import try_h3_partial_rope

                rotated = try_h3_partial_rope(query, key, *rotary_emb)
            query, key = rotated if rotated is not None else _rope_minimax_h3(query, key, *rotary_emb)
        parallel_manager = getattr(attn, "parallel_manager", None)
        sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

        if sp_size > 1:
            if is_phaa_enabled():
                # PHAA path: RoPE applied inside run_phaa_attention via rope_fn
                return self._phaa_parallel(
                    attn=attn,
                    query=query,
                    key=key,
                    value=value,
                    rotary_emb=rotary_emb,
                    attention_mask=attention_mask,
                    sp_group=parallel_manager.sp_group,
                    sp_size=sp_size,
                )
            # Ulysses SP: scatter heads (dim=2), gather sequence (dim=1).
            attn_heads = attn.heads // sp_size
            query, key, value = map(
                lambda x: all_to_all_before_attn(
                    x, parallel_manager.sp_group, scatter_dim=2, gather_dim=1
                ),
                [query, key, value],
            )
        else:
            attn_heads = attn.heads

        # Transpose to (B, H, S, D) for attention dispatch.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)


        if getattr(attn, '_h3_attention_backend', None) is not None:
            hidden_states = attn._h3_attention_backend(
                query, key, value, attn_mask=attention_mask)
        elif getattr(attn, "_h3_turbo_sla", False):
            from x_base.operator.h3_sla import sparse_attention
            # A full real sequence is published by the transformer before SP
            # adds tail padding. Text-refiner attention is never SP-sharded.
            used_len = getattr(attn, "_h3_sla_used_len", query.shape[-2])
            if attention_mask is not None:
                raise ValueError(
                    "Turbo-SLA reference does not accept arbitrary attention masks. "
                    "Use a real-token H3 packed sequence; SP tail padding is stripped separately."
                )
            hidden_states = sparse_attention(query, key, value, used_len=used_len)
        else:
            hidden_states = attention_manager.attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if sp_size > 1:
            hidden_states = all_to_all_after_attn(
                hidden_states,
                parallel_manager.sp_group,
                scatter_dim=1,
                gather_dim=2,
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _phaa_parallel(
        self,
        attn: MiniMaxH3Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sp_group=None,
        sp_size: int = 1,
    ) -> torch.Tensor:
        """PHAA for MiniMax-H3: overlap all2all with flash attention.

        Thin wrapper around ``run_phaa_attention``. MiniMax-H3 has no img
        attention (extra_attn_fn=None) and uses partial-rotation RoPE.

        ``run_phaa_attention`` handles unequal chunk sizes internally:
        56 heads, 8 NPUs, phaa_num=2 → 7 heads/dev → chunks [4, 3].
        """
        head_dim = query.shape[-1]

        # Flatten to (B, S, heads * head_dim) for run_phaa_attention
        query = query.flatten(-2)
        key = key.flatten(-2)
        value = value.flatten(-2)

        hidden_states = run_phaa_attention(
            query, key, value,
            comm_stream=self.comm_stream,
            sp_group=sp_group,
            sp_size=sp_size,
            heads=attn.heads,
            phaa_split_num=get_phaa_split_num() or attn.heads // sp_size,
            head_dim=head_dim,
            rope_fn=None,  # RoPE already applied before SP branch
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            pad_key="mmh3_seq",
            attention_fn=getattr(attn, '_h3_attention_backend', None),
            fp8_transport=getattr(attn, '_h3_phaa_fp8', False),
            gather_priority=getattr(attn, '_h3_phaa_gather_priority', False),
            tail_first=getattr(attn, '_h3_phaa_tail_first', False),
            device_join=getattr(attn, '_h3_phaa_device_join', False),
            async_options=getattr(attn, '_h3_phaa_async_options', None),
            scatter_group=getattr(attn, '_h3_phaa_scatter_group', None),
            scatter_stream=getattr(attn, '_h3_phaa_scatter_stream', None),
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# Monkey-patch: replace default processor with AscendMiniMaxH3AttnProcessor.
# Attention backend selection happens through global attention_manager, not set_attention_backend.
MiniMaxH3Attention._default_processor_cls = AscendMiniMaxH3AttnProcessor
MiniMaxH3Attention._available_processors = [
    MiniMaxH3AttnProcessor,
    AscendMiniMaxH3AttnProcessor,
]

logger = logging.getLogger(__name__)


def _set_parallel_manager(self, parallel_manager):
    """Attach a ``ParallelManager`` so the processor can read ``sp_size``."""
    self.parallel_manager = parallel_manager
    for block in getattr(self, "transformer_blocks", []):
        if hasattr(block, "attn"):
            block.attn.parallel_manager = parallel_manager
    # Token refiner operates on text embeddings before sequence splitting —
    # it must NOT use SP, so we leave its attention without a parallel_manager.


MiniMaxH3Transformer3DModel = transformer_minimax_h3.MiniMaxH3Transformer3DModel
MiniMaxH3Transformer3DModel.set_parallel_manager = _set_parallel_manager


def _enable_parallel(self, dp_size, sp_size, enable_cp=False, ulysses_size=None, ring_size=None):
    """Enable sequence parallelism on the MiniMax-H3 transformer.

    Mirrors ``WanTransformer3DModel.enable_parallel``: creates a
    ``ParallelManager`` and propagates it to every attention module via
    ``set_parallel_manager``.
    """
    cp_size = 2 if enable_cp and sp_size % 2 == 0 else 1
    if cp_size > 1:
        sp_size = sp_size // 2
    parallel_manager = ParallelManager(dp_size, cp_size, sp_size, 1, ulysses_size, ring_size)
    self.set_parallel_manager(parallel_manager)


MiniMaxH3Transformer3DModel.enable_parallel = _enable_parallel


def _build_packed_sequence(
    text_embeds: torch.Tensor,
    video_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    text_idx: torch.Tensor,
    video_idx: torch.Tensor,
    audio_idx: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Build packed hidden_states from text/video/audio embeds."""
    hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    hidden_states = hidden_states.index_copy(1, text_idx, text_embeds)
    hidden_states = hidden_states.index_copy(1, video_idx, video_embeds.to(text_embeds.dtype))
    hidden_states = hidden_states.index_copy(1, audio_idx, audio_embeds.to(text_embeds.dtype))
    return hidden_states


def _split_sequence_for_sp(
    parallel_manager,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    adaln_indices: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> tuple:
    """Split tensors across SP group. Returns (sp_group, pad, hidden_states, rotary_emb, adaln_indices, attention_mask)."""
    if parallel_manager is None or parallel_manager.sp_size < 2:
        return None, None, hidden_states, rotary_emb, adaln_indices, attention_mask

    sp_group = parallel_manager.sp_group
    set_pad("mmh3_seq", hidden_states.shape[1], sp_group)
    _pad = get_pad("mmh3_seq")

    hidden_states = split_sequence(hidden_states, sp_group, dim=1, pad=_pad)
    cos, sin = rotary_emb
    cos = split_sequence(cos, sp_group, dim=0, pad=_pad)
    sin = split_sequence(sin, sp_group, dim=0, pad=_pad)
    rotary_emb = (cos, sin)
    adaln_indices = split_sequence(adaln_indices, sp_group, dim=0, pad=_pad)
    if attention_mask is not None:
        attention_mask = split_sequence(attention_mask, sp_group, dim=0, pad=_pad)
        attention_mask = split_sequence(attention_mask, sp_group, dim=1, pad=_pad)
    return sp_group, _pad, hidden_states, rotary_emb, adaln_indices, attention_mask


def _prepare_timestep_adaln_mask(
    self,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute timestep embedding, adaln_indices, and attention mask."""
    table = getattr(self, '_h3_packed_adaln_table', None)
    if table is not None:
        from x_base.operator.h3_nvfp4 import interpolate_h3_adaln

        temb = interpolate_h3_adaln(timestep, table)
    else:
        temb = self.time_proj(timestep)
        _adaln_cache = getattr(self, '_adaln_cache', None)
        if _adaln_cache is None or not _adaln_cache.has_current:
            temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

    attention_mask = None
    is_pad = token_tags < 0
    if bool(is_pad.any()):
        attention_mask = is_pad[None, :] == is_pad[:, None]

    return temb, adaln_indices, attention_mask


def _gather_and_output(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    sp_group,
    pad,
    timestep_indices: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    return_dict: bool,
) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
    """Gather sequence and compute output heads."""
    if sp_group is not None:
        hidden_states = gather_sequence(hidden_states, sp_group, dim=1, pad=pad)

    hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
    video_output = self.proj_out(hidden_states).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
    if not return_dict:
        return (video_output, audio_output)
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def _compute_rope(
    self,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE, with optional table cache."""

    return self.rope(position_ids)


def _project_modalities(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project video, audio, and text modalities. Returns (video_embeds, audio_embeds, text_embeds)."""
    video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
    text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
    text_embeds = self.token_refiner(text_embeds)
    return video_embeds, audio_embeds, text_embeds


def _run_transformer_blocks(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run transformer blocks with optional gradient checkpointing."""
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
            )
        else:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    return hidden_states


# Sequence-level SP: split/gather sequence, rotary_emb, adaln_indices,
# and attention_mask around the block loop when sp_size > 1.
_original_mmh3_forward = MiniMaxH3Transformer3DModel.forward


@apply_lora_scale("attention_kwargs")
def _mmh3_forward_with_sp(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:

    # Update denoise step counter for rainfusion skip-timestep logic
    attention_manager.update_t_idx()

    parallel_manager = getattr(self, "parallel_manager", None)
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    # Publish real length before the single-card early return as well as SP.
    for block in self.transformer_blocks:
        if getattr(block.attn, "_h3_turbo_sla", False):
            block.attn._h3_sla_used_len = position_ids.shape[0]

    # --- AdaLN schedule cache: build/install table for this timestep ---
    _adaln_cache = getattr(self, "_adaln_cache", None)
    if _adaln_cache is not None:
        _adaln_cache.maybe_install(timestep)

    # --- No SP: delegate to the original forward (undecorated, LoRA already handled) ---
    if sp_size < 2 and not hasattr(self, '_h3_packed_adaln_table'):
        return _original_mmh3_forward.__wrapped__(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    # --- SP path: full forward with split / gather around the block loop ---

    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
    sequence_length = position_ids.shape[0]
    if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
        raise ValueError(
            "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
            f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
        )

    # RoPE table cache: reuse across denoise steps when position_ids unchanged
    rotary_emb = _compute_rope(self, position_ids)

    # 1. Project each modality and scatter into the packed sequence buffer.
    video_embeds, audio_embeds, text_embeds = _project_modalities(
        self, hidden_states, audio_hidden_states, encoder_hidden_states
    )

    hidden_states = _build_packed_sequence(
        text_embeds, video_embeds, audio_embeds, text_indices, video_indices, audio_indices, sequence_length
    )

    # 2. Timestep embedding, AdaLN indices, and padding mask.
    temb, adaln_indices, attention_mask = _prepare_timestep_adaln_mask(
        self, timestep, timestep_indices, token_tags,
    )

    # --- Split sequence across SP group before the block loop ---
    sp_group, _pad, hidden_states, rotary_emb, adaln_indices, attention_mask = _split_sequence_for_sp(
        parallel_manager, hidden_states, rotary_emb, adaln_indices, attention_mask
    )
    # Publish the meaningful sequence length (without SP padding) so that
    # RainFusion sparse attention can slice off the zero-padding rows before
    # dispatching to the rf_v2 kernel and pad the output back afterwards.
    if attention_manager.is_rainfusion_attention:
        attention_manager.set_rainfusion_seq_len(sequence_length)

    # Set _block_idx for AdaLN cache lookup
    for i, block in enumerate(self.transformer_blocks):
        block._block_idx = i

    # 3. Transformer blocks
    hidden_states = _run_transformer_blocks(
        self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
    )

    return _gather_and_output(
        self, hidden_states, temb, sp_group, _pad,
        timestep_indices, video_indices, audio_indices, return_dict,
    )


MiniMaxH3Transformer3DModel.forward = _mmh3_forward_with_sp


_original_adaln_forward = MiniMaxH3AdaLayerNormModulation.forward


def _patched_adaln_forward(self, temb):
    # Cache hit: skip SiLU + Linear + view + chunk
    cache = getattr(self, '_adaln_cache', None)
    if cache is not None and cache.has_current:
        return cache.lookup_block(self._block_idx)

    # File cache mode: weights are stripped, cannot fall back to computation
    if cache is not None and cache._file_cache_loaded:
        raise RuntimeError(
            f"AdaLN cache miss — weights have been pruned, cannot compute. "
            f"Block {self._block_idx}. Check cache file and sampler config."
        )

    # Runtime cache mode: original logic (weights still exist)
    linear_dtype = self.linear.weight.dtype
    if linear_dtype == torch.int8:
        linear_dtype = torch.bfloat16
    if getattr(self, '_h3_packed_adaln', False):
        temb = self.linear(temb).to(torch.bfloat16)
    else:
        temb = self.linear(nn.functional.silu(temb).to(linear_dtype))
    temb = temb.view(-1, 6 * self.hidden_size)
    return temb.chunk(6, dim=-1)


MiniMaxH3AdaLayerNormModulation.forward = _patched_adaln_forward


# Monkey-patch AdaLayerNormOut.forward for AdaLN schedule cache.


def _patched_adaln_out_forward(self, hidden_states, temb, timestep_indices):
    cache = getattr(self, '_adaln_cache', None)
    if cache is not None and cache.has_current:
        shift, scale = cache.lookup_final()
    elif cache is not None and cache._file_cache_loaded:
        raise RuntimeError(
            "AdaLN cache miss — weights have been pruned, cannot compute. "
            "Final layer. Check cache file and sampler config."
        )
    else:
        out_dtype = self.linear.weight.dtype
        if out_dtype == torch.int8:
            out_dtype = torch.bfloat16
        if getattr(self, '_h3_packed_adaln', False):
            modulation = self.linear(temb).to(hidden_states.dtype)
        else:
            modulation = self.linear(nn.functional.silu(temb).to(out_dtype))
        shift, scale = modulation.chunk(2, dim=-1)
    hidden_states = self.norm(hidden_states)
    if (getattr(self, '_h3_fused_indexed_affine', False)
            and hidden_states.is_cuda and not torch.is_grad_enabled()):
        from x_base.operator.h3_indexed_affine import _apply

        return _apply(hidden_states, scale, shift, timestep_indices, modulation=True)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


MiniMaxH3AdaLayerNormOut.forward = _patched_adaln_out_forward


_orig_seq_call = SequentialPipelineBlocks.__call__


@torch.no_grad()
def _timed_seq_call(self, pipeline, state):
    if os.environ.get("DEBUG_TIME") and self.model_name == "minimax-h3":
        is_dist = dist.is_initialized()
        is_rank0 = not is_dist or dist.get_rank() == 0
        accelerator = torch.get_device_module(device_type())
        for block_name, block in self.sub_blocks.items():
            if is_dist:
                dist.barrier()
            accelerator.synchronize()
            t0 = time.time()
            pipeline, state = block(pipeline, state)
            accelerator.synchronize()
            if is_dist:
                dist.barrier()
            if is_rank0:
                print(f"[DEBUG_TIME] {block_name}: {time.time() - t0:.3f}s")
        return pipeline, state
    return _orig_seq_call(self, pipeline, state)


SequentialPipelineBlocks.__call__ = _timed_seq_call
