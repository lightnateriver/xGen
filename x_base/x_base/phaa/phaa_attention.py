"""PHAA (Parallel Head All2All) core attention algorithm.

Extracted from Wan's ``AscendWanAttnProcessor2_0._phaa_parallel`` to enable
reuse across model architectures (Wan, MiniMax-H3, …).

The algorithm splits the attention head dimension into chunks and uses a
dedicated ``comm_stream`` so that all2all communication overlaps with flash
attention computation.

Supports **unequal chunk sizes**: when ``heads // sp_size`` is not divisible
by ``phaa_split_num``, the remainder is distributed across the first chunks.
Example: 7 heads/dev, phaa_num=2 → chunks [4, 3].

Model-specific differences are injected via parameters:
    * ``rope_fn``      – RoPE application (or None to skip).
    * ``pad_key``      – Key for ``get_pad()`` (Wan: "pad", MiniMax-H3: "mmh3_seq").
    * ``extra_attn_fn`` – Optional callback for extra attention (e.g. Wan i2v img attention).

⚠️  Circular-dependency constraint (P1-1):
    All imports MUST come from concrete sub-modules, NOT from ``x_base`` top-level.
    Otherwise: x_base/__init__.py → phaa/__init__.py → phaa_attention.py → x_base  →  ImportError.

Note (P1-4):
    ``extra_attn_fn`` signature does not include ``attention_mask``.
    Callers that need it (e.g. Wan img attention) should capture it via closure.
"""

from contextlib import nullcontext

import torch

# ✅ Import from sub-modules to avoid circular dependency (P1-1)
from x_base.phaa.globals import get_phaa_split_num          # P0-2: not get_pad
from x_base.sequence_parallelism import get_pad             # P0-2: correct source
from x_base.operator import attention_manager               # P0-3: explicit import


def run_phaa_attention(
    query: torch.Tensor,         # (B, S, H) where H = heads * head_dim
    key: torch.Tensor,           # (B, Skv, H)
    value: torch.Tensor,         # (B, Skv, H)
    *,
    comm_stream,                  # accelerator communication stream
    sp_group,                     # SP process group
    sp_size: int,                # W_S
    heads: int,                   # total attention heads
    phaa_split_num: int,          # fa_split_num
    head_dim: int,                # D — per-head dim, caller computes externally (P1-3)
    # --- model-specific hooks (all optional) ---
    rope_fn=None,                 # callable(q, k, rotary_emb) -> (q, k)
    rotary_emb=None,              # tuple(cos, sin) or None
    attention_mask=None,          # tensor or None
    pad_key="pad",               # str, for get_pad()
    extra_attn_fn=None,           # callable(query_reshaped, head_offset, heads_per_split) -> chunk or None
    k_ref=None,                   # (B, heads, S_ref, D) cached reference K for prefix attention
    v_ref=None,                   # (B, heads, S_ref, D) cached reference V for prefix attention
    num_ref_global=0,             # number of ref tokens in the global (post-all2all) sequence
    attention_fn=None,            # optional model-local operator; receives local head offset
    fp8_transport=False,
    gather_priority=False,
    scatter_stream=None,
    scatter_group=None,
    tail_first=False,
    device_join=False,
    async_options=None,
) -> torch.Tensor:
    """PHAA core: split heads into chunks, overlap all2all with flash attention.

    Returns:
        hidden_states: (B, S, heads * head_dim)
    """
    # --- helpers ---
    def apply_pad(tensor, pad, dim=2):
        if pad > 0:
            pad_size = list(tensor.shape)
            pad_size[dim] = pad
            tensor = torch.cat(
                [tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
        return tensor

    async_exchange = async_options is not None
    direct_gather = async_exchange and async_options.direct_gather
    direct_return = async_exchange and async_options.direct_return
    fused_return_pack = async_exchange and async_options.fused_return_pack

    def reshape_for_attention(x):
        if direct_gather:
            return x
        # (W_S, S, B, N, D) -> (B, N, W_S*S, D)
        x = x.flatten(0, 1).permute(1, 2, 0, 3)
        if get_pad(pad_key) > 0:
            x = x.narrow(2, 0, x.shape[2] - get_pad(pad_key))
        return x

    def reshape_after_attention(x):
        # (B, N, W_S*S, D) -> (W_S, S, B, N*D)
        x = apply_pad(x, pad=get_pad(pad_key), dim=2)
        return x.unflatten(2, (W_S, -1)).permute(2, 3, 0, 1, 4).flatten(-2, -1)

    # --- prepare shapes ---
    W_S = sp_size
    fa_split_num = phaa_split_num
    B, Sq, H = query.shape
    Skv = key.shape[1]
    D = head_dim

    # Compute unequal chunk sizes: distribute remainder across first chunks.
    # e.g. heads_per_device=7, fa_split_num=2 → [4, 3]
    heads_per_device = heads // W_S
    base = heads_per_device // fa_split_num
    remainder = heads_per_device % fa_split_num
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(fa_split_num)]

    chunk_offsets = [sum(chunk_sizes[:index]) for index in range(fa_split_num)]
    order = list(range(fa_split_num))
    if tail_first:
        if not fp8_transport or fa_split_num != 4:
            raise ValueError("PHAA tail-first requires FP8 transport and four chunks")
        order = [3, 0, 1, 2]

    # --- prepare events ---
    accelerator = torch.cuda if query.device.type == "cuda" else torch.npu
    exchange = torch.distributed.all_to_all_single
    if fp8_transport:
        if query.device.type != "cuda":
            raise ValueError("PHAA FP8 transport requires CUDA")
        from .fp8_transport import fp8_all_to_all, fp8_all_to_all_many

        def exchange(output, tensor, *, group):
            fp8_all_to_all(output, tensor, D, group)

    if gather_priority and not fp8_transport:
        raise ValueError("PHAA gather priority requires FP8 transport")
    if device_join and (not fp8_transport or scatter_stream is not None or extra_attn_fn is not None):
        raise ValueError("PHAA device join requires single-stream FP8 transport without extra attention")
    if async_exchange:
        if not device_join:
            raise ValueError("PHAA async exchange requires device join")
        from .fp8_async_transport import finish_fp8_exchange, submit_fp8_exchange
    if (scatter_stream is None) != (scatter_group is None) or (
            scatter_stream is not None and not fp8_transport):
        raise ValueError("Separate PHAA scatter requires FP8, a stream and a process group")
    output_stream = scatter_stream if scatter_stream is not None else comm_stream
    output_group = scatter_group if scatter_group is not None else sp_group

    if async_exchange:
        pending_gathers = {}
        pending_returns = []
    else:
        all2all_before_attn_is_finished = [accelerator.Event() for _ in range(fa_split_num)]
    fa_is_finished = [accelerator.Event() for _ in range(fa_split_num)]
    has_extra = extra_attn_fn is not None
    if has_extra:
        fa_extra_is_finished = [accelerator.Event() for _ in range(fa_split_num)]

    # --- prepare per-chunk buffers (variable size) ---
    def gather_shape(seq, chunk_heads):
        if direct_gather:
            return (B, chunk_heads, W_S * seq - get_pad(pad_key), D)
        return (W_S, seq, B, chunk_heads, D)

    query_a2a_bufs = [torch.empty(gather_shape(Sq, cs), device=query.device, dtype=query.dtype)
                     for cs in chunk_sizes]
    key_a2a_bufs = [torch.empty(gather_shape(Skv, cs), device=key.device, dtype=key.dtype)
                   for cs in chunk_sizes]
    value_a2a_bufs = [torch.empty(gather_shape(Skv, cs), device=value.device, dtype=value.dtype)
                     for cs in chunk_sizes]
    if direct_return:
        output = torch.empty((B, Sq, H), device=query.device, dtype=query.dtype)
    else:
        output_a2a_bufs = [torch.empty((W_S, Sq, B, cs * D), device=query.device, dtype=query.dtype)
                          for cs in chunk_sizes]
    hidden_states_buf = [None] * fa_split_num

    hidden_states_extra_buf = [None] * fa_split_num
    output_extra_a2a_bufs = None
    if has_extra:
        output_extra_a2a_bufs = [
            torch.empty((W_S, Sq, B, cs * D), device=query.device, dtype=query.dtype)
            for cs in chunk_sizes]

    # --- prepare input: split head dim into per-chunk tensors ---
    # (B, S, H) → (B, S, heads, D) → (B, S, W_S, heads_per_device, D)
    query = query.unflatten(-1, (heads, D)).unflatten(2, (W_S, heads_per_device))
    key = key.unflatten(-1, (heads, D)).unflatten(2, (W_S, heads_per_device))
    value = value.unflatten(-1, (heads, D)).unflatten(2, (W_S, heads_per_device))

    query_chunks = []
    key_chunks = []
    value_chunks = []
    c_offset = 0
    for c in range(fa_split_num):
        cs = chunk_sizes[c]
        for chunks, tensor in ((query_chunks, query), (key_chunks, key), (value_chunks, value)):
            view = tensor[:, :, :, c_offset:c_offset + cs, :].permute(2, 1, 0, 3, 4)
            chunks.append(view if fp8_transport else view.contiguous())
        c_offset += cs

    def gather_chunk(index):
        outputs = (query_a2a_bufs[index], key_a2a_bufs[index], value_a2a_bufs[index])
        inputs = (query_chunks[index], key_chunks[index], value_chunks[index])
        if fp8_transport:
            if async_exchange:
                pending_gathers[index] = submit_fp8_exchange(
                    outputs, inputs, D, sp_group, row_tiling=async_options.row_tiling, direct_gather=direct_gather,
                )
            else:
                fp8_all_to_all_many(outputs, inputs, D, sp_group)
            if device_join:
                for tensor in inputs:
                    tensor.record_stream(comm_stream)
        else:
            for output, tensor in zip(outputs, inputs):
                exchange(output, tensor, group=sp_group)
        if not async_exchange:
            all2all_before_attn_is_finished[index].record(comm_stream)

    # --- 1. first chunk all2all on comm_stream ---
    comm_stream.wait_stream(accelerator.current_stream())
    if scatter_stream is not None:
        scatter_stream.wait_stream(accelerator.current_stream())
    with accelerator.stream(comm_stream):
        for index in (order if gather_priority else order[:1]):
            gather_chunk(index)
    # Keep compute-stream send buffers alive until the communication stream
    # completes. async_op=False does not imply device completion on the host.

    # --- 2. main loop ---
    for position, step in enumerate(order):
        head_offset = chunk_offsets[step]
        cs = chunk_sizes[step]

        # 2.1 compute attention
        if async_exchange:
            finish_fp8_exchange(pending_gathers.pop(step))
        else:
            accelerator.current_stream().wait_event(all2all_before_attn_is_finished[step])
        query_ = reshape_for_attention(query_a2a_bufs[step])
        key_ = reshape_for_attention(key_a2a_bufs[step])
        value_ = reshape_for_attention(value_a2a_bufs[step])
        if rope_fn is not None and rotary_emb is not None:
            query_, key_ = rope_fn(query_, key_, *rotary_emb)

        if k_ref is not None:
            # all2all 后 Q/K/V 是 (B, cs, S_full, D)，[0:num_ref_global] 是 zeros，[num_ref_global:] 是 non-ref
            query_ = query_[:, :, num_ref_global:, :]
            key_ = key_[:, :, num_ref_global:, :]
            value_ = value_[:, :, num_ref_global:, :]
            # 切缓存 ref 对应 head chunk
            k_ref_chunk = k_ref[:, head_offset:head_offset + cs, :, :]
            v_ref_chunk = v_ref[:, head_offset:head_offset + cs, :, :]
            # 拼接 ref + non-ref
            key_ = torch.cat([k_ref_chunk, key_], dim=2)
            value_ = torch.cat([v_ref_chunk, value_], dim=2)

        if attention_fn is None:
            hidden_states_buf[step] = attention_manager.attention(
                query_, key_, value_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            ).type_as(query)
        else:
            hidden_states_buf[step] = attention_fn(
                query_, key_, value_, head_offset=head_offset, attn_mask=attention_mask
            ).type_as(query)
        if k_ref is not None:
            S_full = hidden_states_buf[step].shape[2] + num_ref_global
            out_full = torch.zeros(
                hidden_states_buf[step].shape[0], hidden_states_buf[step].shape[1],
                S_full, hidden_states_buf[step].shape[3],
                device=hidden_states_buf[step].device, dtype=hidden_states_buf[step].dtype,
            )
            out_full[:, :, num_ref_global:, :] = hidden_states_buf[step]
            hidden_states_buf[step] = out_full
        if not fused_return_pack:
            hidden_states_buf[step] = reshape_after_attention(hidden_states_buf[step])
        fa_is_finished[step].record()

        # 2.2 extra attention (e.g., Wan img attention)
        if has_extra:
            hidden_states_extra_buf[step] = extra_attn_fn(query_, head_offset, cs)
            if hidden_states_extra_buf[step] is not None:
                hidden_states_extra_buf[step] = reshape_after_attention(
                    hidden_states_extra_buf[step]).type_as(query)
                fa_extra_is_finished[step].record()

        # [MEM-2] free a2a input buffers — attention & extra attention are done
        # query_/key_/value_ are views of a2a_bufs[step], must release first
        query_ = key_ = value_ = None
        query_a2a_bufs[step] = key_a2a_bufs[step] = value_a2a_bufs[step] = None

        # 2.3 communication on comm_stream
        with accelerator.stream(comm_stream):
            # next chunk all2all
            if not gather_priority and position + 1 < fa_split_num:
                gather_chunk(order[position + 1])
            scatter_context = accelerator.stream(scatter_stream) if scatter_stream is not None else nullcontext()
            with scatter_context:
                output_stream.wait_event(fa_is_finished[step])
                if async_exchange:
                    pending_returns.append(submit_fp8_exchange(
                        None if direct_return else (output_a2a_bufs[step],),
                        (hidden_states_buf[step],), D, output_group, row_tiling=async_options.row_tiling,
                        return_shard_seq=Sq if fused_return_pack else 0,
                    ))
                else:
                    exchange(output_a2a_bufs[step], hidden_states_buf[step], group=output_group)
                if device_join:
                    hidden_states_buf[step].record_stream(comm_stream)
                if has_extra and hidden_states_extra_buf[step] is not None:
                    output_stream.wait_event(fa_extra_is_finished[step])
                    torch.distributed.all_to_all_single(
                        output_extra_a2a_bufs[step], hidden_states_extra_buf[step],
                        group=output_group, async_op=False)
        # Send/output references stay alive through the final stream join.

    # --- 3. finalize ---
    # Device join orders output reads on the compute stream. The default
    # host join also keeps send buffers alive without allocator records.
    if async_exchange:
        for step, pending in zip(order, pending_returns):
            if direct_return:
                layout = (Sq, B, W_S, heads_per_device, chunk_sizes[step], chunk_offsets[step])
                finish_fp8_exchange(pending, output=output, return_layout=layout)
            else:
                finish_fp8_exchange(pending)
    elif device_join:
        accelerator.current_stream().wait_stream(comm_stream)
    else:
        comm_stream.synchronize()
    if scatter_stream is not None:
        scatter_stream.synchronize()
    query_chunks.clear()
    key_chunks.clear()
    value_chunks.clear()
    hidden_states_buf.clear()
    hidden_states_extra_buf.clear()

    if direct_return:
        return output

    # Assemble output: per-chunk (W_S, Sq, B, cs*D) → (B, Sq, heads*D)
    output_chunks = [buf.permute(2, 1, 0, 3) for buf in output_a2a_bufs]
    hidden_states = torch.cat(output_chunks, dim=3).flatten(-2, -1)

    if has_extra and output_extra_a2a_bufs is not None:
        extra_chunks = [buf.permute(2, 1, 0, 3) for buf in output_extra_a2a_bufs]
        hidden_states_extra = torch.cat(extra_chunks, dim=3).flatten(-2, -1)
        hidden_states = hidden_states + hidden_states_extra

    return hidden_states
