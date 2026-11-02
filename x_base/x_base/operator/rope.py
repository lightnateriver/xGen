import torch
import torch_npu
import logging

logger = logging.getLogger("attention")

try:
    import cann_ops
    import ascend_cloud

    ASCEND_CLOUD_AVAILABLE = True
except ImportError:
    logger.error("Not install CANN-OPS or ascend_cloud")
    ASCEND_CLOUD_AVAILABLE = False

try:
    from mindiesd import rotary_position_embedding

    AVAILABLE_MINDIESD = True
except ImportError:
    logger.warning("Not install MINDIE")
    AVAILABLE_MINDIESD = False


def apply_rotary_emb(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
):
    x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[..., 0::2]
    sin = sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class RopeManager:
    def __init__(self):
        self.is_rope_fused = False

    def enable_rope_fused(self):
        if not ASCEND_CLOUD_AVAILABLE and not AVAILABLE_MINDIESD:
            logger.error("Not install CANN-OPS or MINDIE. Now using base rope")
        else:
            self.is_rope_fused = True

    @staticmethod
    def rope_flux(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        query = query.contiguous()
        key = key.contiguous()
        if cos.dtype != query.dtype:
            cos = cos.to(dtype=query.dtype)
        if sin.dtype != query.dtype:
            sin = sin.to(dtype=query.dtype)
        cos = cos[..., 0::2]
        sin = sin[..., 1::2]
        row1 = torch.stack([cos, sin], dim=-1)
        row2 = torch.stack([-sin, cos], dim=-1)
        freqs_cis = torch.stack([row1, row2], dim=-1)
        return torch.ops.ascend_cloud.rope_flux(query, key, freqs_cis)

    @staticmethod
    def rope_base(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

    @staticmethod
    def rope_mindie(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        query = rotary_position_embedding(query, cos, sin, rotated_mode="rotated_interleaved", fused=True)
        key = rotary_position_embedding(key, cos, sin, rotated_mode="rotated_interleaved", fused=True)
        return query, key

    @staticmethod
    def rope_rotated_half_base(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        """Apply RoPE using rotated_half convention (split in half, not interleaved)."""
        d = query.shape[-1]
        q1, q2 = query[..., :d // 2], query[..., d // 2:]
        k1, k2 = key[..., :d // 2], key[..., d // 2:]

        if cos.dtype != query.dtype:
            cos = cos.to(dtype=query.dtype)
        if sin.dtype != query.dtype:
            sin = sin.to(dtype=query.dtype)

        cos_half = cos[..., :d // 2]
        sin_half = sin[..., :d // 2]

        q_out = torch.empty_like(query)
        k_out = torch.empty_like(key)
        q_out[..., :d // 2] = q1 * cos_half - q2 * sin_half
        q_out[..., d // 2:] = q1 * sin_half + q2 * cos_half
        k_out[..., :d // 2] = k1 * cos_half - k2 * sin_half
        k_out[..., d // 2:] = k1 * sin_half + k2 * cos_half
        return q_out, k_out

    @staticmethod
    def rope_rotated_half_mindie(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        query = rotary_position_embedding(query, cos, sin, rotated_mode="rotated_half", fused=True)
        key = rotary_position_embedding(key, cos, sin, rotated_mode="rotated_half", fused=True)
        return query, key

    def rope_rotated_half(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        if self.is_rope_fused and AVAILABLE_MINDIESD:
            return self.rope_rotated_half_mindie(query, key, cos, sin)
        elif self.is_rope_fused and ASCEND_CLOUD_AVAILABLE:
            return self.rope_rotated_half_base(query, key, cos, sin)
        else:
            return self.rope_rotated_half_base(query, key, cos, sin)

    def rope(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
    ):
        if self.is_rope_fused and AVAILABLE_MINDIESD:
            return self.rope_mindie(query, key, cos, sin)
        elif self.is_rope_fused and ASCEND_CLOUD_AVAILABLE:
            return self.rope_flux(query, key, cos, sin)
        else:
            return self.rope_base(query, key, cos, sin)


rope_manager = RopeManager()
