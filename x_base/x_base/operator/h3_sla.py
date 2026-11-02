"""Portable BF16 reference for LightX2V H3 Turbo-SLA's ``sage2`` routing.

This computes only selected KV blocks, without a token-by-token dense mask.
It reproduces the 5090 router (128 query / 64 key tokens), but deliberately
does not emulate Sage2 INT8 rounding. CPU is supported for numerical tests;
on Ascend the same torch operations dispatch through torch_npu. This is a
correctness backend, not a fused performance kernel.
"""

import math

import torch
import torch.nn.functional as F


def _pool(x, block_size):
    length = x.shape[-2]
    padded = F.pad(x, (0, 0, 0, (-length) % block_size))
    pooled = padded.float().reshape(*x.shape[:-2], -1, block_size, x.shape[-1]).sum(-2)
    counts = (length - torch.arange(pooled.shape[-2], device=x.device) * block_size).clamp(
        min=1, max=block_size
    )
    return (pooled / counts[:, None]).to(x.dtype)


def block_routes(query, key, sparsity=0.85, block_q=128, block_k=64):
    """Mean-pool, smooth K, then floor the keep budget as in LightX2V.

    Routing is performed in query-block chunks by ``sparse_attention``; this
    helper is also independently usable for comparisons with the CUDA router.
    """
    if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError("H3 SLA sparsity must be finite and in [0, 1).")
    if block_q < 1 or block_k < 1 or query.shape[-2] < 1 or key.shape[-2] < 1:
        raise ValueError("H3 SLA requires nonempty sequences and positive block sizes.")
    # Match the materialized subtraction's rounding before FP32 reduction.
    centered = (key.float() - key.mean(dim=-2, keepdim=True).float()).to(key.dtype)
    q_pool, k_pool = _pool(query, block_q), _pool(centered, block_k)
    scores = q_pool @ k_pool.transpose(-1, -2)
    keep = max(1, min(k_pool.shape[-2], int((1.0 - sparsity) * k_pool.shape[-2])))
    return torch.topk(scores, keep, dim=-1, sorted=False).indices


def sparse_attention(query, key, value, *, used_len=None, sparsity=0.85,
                     block_q=128, block_k=64, query_chunk_blocks=1, scale=None):
    """Selected-block softmax attention for B,H,S,D tensors.

    SP padding is stripped BEFORE pooling/top-k. The output padding is zero.
    All real packed rows participate, including text/audio, matching the
    published LightX2V configuration (no invented modality exemptions).
    """
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("H3 SLA requires identical B,H,S,D self-attention tensors.")
    if query.dtype != key.dtype or query.dtype != value.dtype or not query.is_floating_point():
        raise ValueError("H3 SLA Q/K/V must share a floating-point dtype.")
    if query.device != key.device or query.device != value.device:
        raise ValueError("H3 SLA Q/K/V must share a device.")
    if not isinstance(query_chunk_blocks, int) or query_chunk_blocks < 1:
        raise ValueError("query_chunk_blocks must be a positive integer.")
    batch, heads, length, dim = query.shape
    used_len = length if used_len is None else used_len
    if not isinstance(used_len, int) or not 0 < used_len <= length:
        raise ValueError("used_len must describe the real nonempty sequence before SP padding.")
    q, k, v = (x[:, :, :used_len] for x in (query, key, value))
    output = torch.zeros_like(query)
    offsets = torch.arange(block_k, device=query.device)
    chunk_tokens = block_q * query_chunk_blocks
    # Center over the FULL real sequence; never independently center a KV shard.
    centered = (k.float() - k.mean(dim=-2, keepdim=True).float()).to(k.dtype)
    if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError("H3 SLA sparsity must be finite and in [0, 1).")
    if block_q < 1 or block_k < 1:
        raise ValueError("H3 SLA block sizes must be positive.")
    k_pool = _pool(centered, block_k)
    keep = max(1, min(k_pool.shape[-2], int((1.0 - sparsity) * k_pool.shape[-2])))
    for start in range(0, used_len, chunk_tokens):
        stop = min(start + chunk_tokens, used_len)
        q_chunk = q[:, :, start:stop]
        q_pool = _pool(q_chunk, block_q)
        routes = torch.topk(q_pool @ k_pool.transpose(-1, -2), keep, dim=-1, sorted=False).indices
        indices = (routes[..., None] * block_k + offsets).flatten(-2)
        valid = indices < used_len
        indices = indices.clamp(max=used_len - 1)
        blocks = routes.shape[-2]
        # Advanced indexing gathers only the selected keys, not an S x S tensor.
        b = torch.arange(batch, device=q.device)[:, None, None, None]
        h = torch.arange(heads, device=q.device)[None, :, None, None]
        selected_k, selected_v = k[b, h, indices], v[b, h, indices]
        padded_q = F.pad(q_chunk, (0, 0, 0, (-q_chunk.shape[-2]) % block_q))
        blocked_q = padded_q.reshape(batch, heads, blocks, block_q, dim)
        attended = F.scaled_dot_product_attention(
            blocked_q.reshape(batch * heads * blocks, 1, block_q, dim),
            selected_k.reshape(batch * heads * blocks, 1, -1, dim),
            selected_v.reshape(batch * heads * blocks, 1, -1, dim),
            attn_mask=valid.reshape(batch * heads * blocks, 1, 1, -1),
            dropout_p=0.0, is_causal=False, scale=scale,
        )
        output[:, :, start:stop] = attended.reshape(batch, heads, -1, dim)[:, :, :stop - start]
    return output
