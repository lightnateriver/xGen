"""Model-independent fused projection sharding and sequence/TP adapters."""

import torch
from torch import nn

from .layers import ColumnParallelLinear, RowParallelLinear


def shard_fused_tensor(tensor, *, axis, rank, world_size, parts=1):
    """Shard each packed projection independently (Q/K/V, gate/up, etc.)."""
    if any(type(value) is not int for value in (axis, rank, world_size, parts)):
        raise ValueError("Projection partition parameters must be integers")
    if world_size < 1 or parts < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid projection partition")
    if not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError("Projection partition axis is outside the tensor")
    if tensor.shape[axis] % (parts * world_size):
        raise ValueError("Fused projection dimension must divide parts * TP")
    return torch.cat(
        [
            part.chunk(world_size, dim=axis)[rank]
            for part in tensor.detach().chunk(parts, dim=axis)
        ],
        dim=axis,
    ).contiguous()


def shard_linear(original, manager, *, column, parts=1, reduction=None):
    """Create an explicitly bound TP Linear from materialized CPU weights."""
    if not isinstance(original, nn.Linear) or original.weight.device.type != "cpu":
        raise ValueError("TP sharding requires materialized Linear CPU weights")
    if manager.tp_size < 2 or manager.tp_group is None:
        raise ValueError("TP sharding requires an explicit multi-rank group")
    weight = shard_fused_tensor(
        original.weight,
        axis=0 if column else 1,
        rank=manager.tp_rank,
        world_size=manager.tp_size,
        parts=parts,
    )
    bias = original.bias
    if column and bias is not None:
        bias = shard_fused_tensor(
            bias, axis=0, rank=manager.tp_rank, world_size=manager.tp_size, parts=parts
        )
    cls = ColumnParallelLinear if column else RowParallelLinear
    options = {} if column else {"reduce_fn": reduction}
    layer = cls(
        original.in_features,
        original.out_features,
        bias=bias is not None,
        dtype=original.weight.dtype,
        tp_group=manager.tp_group,
        **options,
    )
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.bias = (
        nn.Parameter(bias.detach().clone(), requires_grad=False)
        if bias is not None
        else None
    )
    return layer


class GatherColumnInput(nn.Module):
    def __init__(self, projection, exchange):
        super().__init__()
        self.projection, self.exchange = projection, exchange

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, local):
        return self.projection(self.exchange.gather(local))


class SequenceColumn(nn.Module):
    def __init__(self, projection, exchange, *, parts=1, restore_sequence=False):
        super().__init__()
        self.projection, self.exchange = projection, exchange
        self.parts, self.restore_sequence = parts, restore_sequence

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, local):
        value = self.projection(self.exchange.gather(local))
        return (
            self.exchange.scatter_columns(value, self.parts)
            if self.restore_sequence
            else value
        )


class SequenceRow(nn.Module):
    def __init__(
        self, projection, exchange, *, bias=None, gather_input=False, partitions=2
    ):
        super().__init__()
        self.projection, self.exchange = projection, exchange
        self.gather_input, self.partitions = gather_input, partitions
        self.register_buffer(
            "bias", bias.detach().clone() if bias is not None else None
        )

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, local):
        if self.gather_input:
            local = (
                self.exchange.gather(local)
                .chunk(self.partitions, dim=-1)[self.exchange.rank]
                .contiguous()
            )
        value = self.exchange.reduce_scatter(self.projection(local))
        return value if self.bias is None else value + self.bias
