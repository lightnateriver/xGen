"""Reusable PHAA local layout boundaries, independent of any model framework."""
import torch

from x_base.operator.npu_fusion import capabilities as caps
from x_base.operator.npu_fusion.quantized import QuantizedRows
from .local_ops import PHAALocalOps


def make_fused_layout_ops(query, *, heads, head_dim, sp_size, splits, padding,
                          layout_kernel=None, merge_kernel=None, attention=None):
    """Build per-call callbacks; core PHAA owns communication and the final join.

    The current kernel implementations have fixed geometry, not fixed sequence
    lengths. Quantized output is permitted only when the caller owns a compatible
    consumer. All callbacks and scratch storage live for exactly one PHAA call.
    """
    if layout_kernel is None and merge_kernel is None:
        return PHAALocalOps(attention=attention)
    if ((sp_size, splits, heads, head_dim) !=
            (caps.PHAA_SP, caps.PHAA_CHUNKS, caps.PHAA_HEADS, caps.PHAA_HEAD_DIM)
            or query.ndim != 3 or query.shape[0] != 1 or query.shape[-1] != heads * head_dim):
        raise ValueError("Local PHAA kernel geometry is unsupported")
    if layout_kernel is not None and padding not in (0, 1):
        raise ValueError("Local PHAA layout supports zero or one padding token")
    B, S, width = query.shape
    chunk_heads = heads // sp_size // splits
    storage = {}

    def prepare_inputs(q, k, v):
        sends = [torch.empty((splits, sp_size, S, B, chunk_heads, head_dim), device=q.device, dtype=q.dtype)
                 for _ in range(3)]
        layout_kernel.pack_qkv(*(x.unflatten(-1, (heads, head_dim)) for x in (q, k, v)), *sends)
        return tuple(list(x.unbind(0)) for x in sends)

    def allocate_outputs(q):
        storage["output"] = torch.empty((splits, sp_size, S, B, chunk_heads * head_dim), device=q.device, dtype=q.dtype)
        return list(storage["output"].unbind(0))

    def restore_output(x, *, chunk_size):
        back = torch.empty((sp_size, S, B, chunk_size * head_dim), device=x.device, dtype=x.dtype)
        layout_kernel.pack_back(x, back, L=sp_size * S - padding, S=S)
        return back

    def merge_outputs(buffers):
        packed = storage["output"] if layout_kernel is not None else torch.stack(buffers)
        if merge_kernel is not None:
            q, scale = merge_kernel.merge_quant(packed, blocks=caps.VECTOR_BLOCKS)
            return QuantizedRows(q, scale, (B, S, width))
        result = torch.empty((B, S, width), device=query.device, dtype=query.dtype)
        layout_kernel.merge_back(packed, result)
        return result

    return PHAALocalOps(
        prepare_inputs=prepare_inputs if layout_kernel is not None else None,
        allocate_outputs=allocate_outputs if layout_kernel is not None else None,
        restore_output=restore_output if layout_kernel is not None else None,
        merge_outputs=merge_outputs, attention=attention,
    )
