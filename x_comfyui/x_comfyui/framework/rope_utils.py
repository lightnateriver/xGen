"""ComfyUI rope_freqs conversion + thin RoPE helper for MiniMax-H3.

ComfyUI passes rope_freqs as 2x2 rotation matrices with shape
[1, S, 1, 48, 2, 2] where the last two dims are [[c, -s], [s, c]].
mindie's rotary_position_embedding expects separate cos/sin tensors
with shape [..., rotary_dim].  This module handles the conversion and
delegates to rope_manager.rope_rotated_half() which uses the
rotated_half convention (correct for MiniMax-H3).

The 48-dim cos/sin is expanded to 96 via torch.cat([cos, cos], dim=-1),
matching upstream diffusers' rope.forward() which does
torch.cat((freqs, freqs), dim=-1).
"""

import torch

from x_base.operator.rope import rope_manager


def _comfyui_rope_freqs_to_cos_sin(rope_freqs):
    """Convert ComfyUI rope_freqs to cos/sin tensors.

    Args:
        rope_freqs: [1, S, 1, 48, 2, 2] — 2x2 rotation matrices [[c, -s], [s, c]]

    Returns:
        (cos, sin): each [1, S, 1, 96] — expanded via torch.cat([x, x], dim=-1)
    """
    # Extract cos (top-left) and sin (bottom-left) from 2x2 matrix
    cos = rope_freqs[..., 0, 0]  # [1, S, 1, 48]
    sin = rope_freqs[..., 1, 0]  # [1, S, 1, 48]

    # Expand 48 → 96 (rotary_dim) via cat, matching upstream diffusers
    cos = torch.cat([cos, cos], dim=-1)  # [1, S, 1, 96]
    sin = torch.cat([sin, sin], dim=-1)  # [1, S, 1, 96]

    return cos, sin


def _fused_rope_minimax_h3(q, k, rope_freqs, rot_dim):
    """Apply RoPE to q/k using rope_manager.rope_rotated_half().

    Handles partial rotary: only the first rot_dim channels are rotated,
    the remaining channels pass through unchanged.

    Args:
        q: [B, S, H, D] or [1, S, H, D] — full query tensor
        k: [B, S, H, D] or [1, S, H, D] — full key tensor
        rope_freqs: [1, S, 1, 48, 2, 2] — ComfyUI rotation matrices
        rot_dim: int — number of channels to rotate (e.g., 96)

    Returns:
        (q_out, k_out): same shape as input, with first rot_dim channels rotated
    """
    # Convert rope_freqs to cos/sin
    cos, sin = _comfyui_rope_freqs_to_cos_sin(rope_freqs)  # [1, S, 1, 96]

    # Split into rotary and pass-through parts
    q_rot = q[..., :rot_dim]
    k_rot = k[..., :rot_dim]
    q_pass = q[..., rot_dim:]
    k_pass = k[..., rot_dim:]

    # Apply RoPE via rope_manager (handles fused/fallback internally)
    q_rot, k_rot = rope_manager.rope_rotated_half(q_rot, k_rot, cos, sin)

    # Concatenate back to full dim
    q_out = torch.cat([q_rot, q_pass], dim=-1)
    k_out = torch.cat([k_rot, k_pass], dim=-1)

    return q_out, k_out
