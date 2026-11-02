import os
import torch
import torch.nn.functional as F
import logging
from ascendx_video.base.operator.attention import sage_attention_infer

logging.basicConfig(level=logging.INFO)

try:
    from mindiesd import attention_forward
    MINDIE_AVAILABLE = True
except ImportError:
    logging.error("Not install MINDIE")
    MINDIE_AVAILABLE = False


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    attention_mode='sdpa',
):
    if "sageattn" in attention_mode:
        if not (q.dtype == k.dtype == v.dtype):
            return sage_attention_infer(
                q.transpose(1, 2),
                k.transpose(1, 2).to(q.dtype),
                v.transpose(1, 2).to(q.dtype)
            ).transpose(1, 2).contiguous()
        else:
            return sage_attention_infer(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2)
            ).transpose(1, 2).contiguous()
    elif "laser" in attention_mode and q.shape[1] == k.shape[1]:
        return attention_forward(q, k, v, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
    else:
        if not (q.dtype == k.dtype == v.dtype):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype)).transpose(1, 2).contiguous()
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()
