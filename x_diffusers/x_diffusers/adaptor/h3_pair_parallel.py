"""Optional TP2 QKV/O/FFN projections; retain the existing SP8/PHAA processor."""
import logging
import torch
from torch import nn


logger = logging.getLogger(__name__)


def slice_projection(original, exchange, *, column, parts=1):
    from x_base.operator.h3_fp8 import H3FP8Linear

    if not isinstance(original, (nn.Linear, H3FP8Linear)) or original.weight.device.type != 'cpu':
        raise ValueError('TP2 must slice CPU projections before the resident-device transfer')
    axis = 0 if column else 1
    if original.weight.shape[axis] % (parts*2):
        raise ValueError('TP2 projection dimensions must divide fused parts and two ranks')
    weight = torch.cat([p.chunk(2, dim=axis)[exchange.rank]
                        for p in original.weight.detach().chunk(parts, dim=axis)], dim=axis).contiguous()
    bias = original.bias
    if column and bias is not None:
        bias = torch.cat([p.chunk(2)[exchange.rank] for p in bias.detach().chunk(parts)]).contiguous()
    if isinstance(original, H3FP8Linear):
        # Preserve the original FP8 bytes and global weight scale. No requantization.
        result = H3FP8Linear.from_packed(weight, original.weight_scale, bias if column else None)
        if not column:
            result.activation_group = exchange.group
    else:
        result = nn.Linear(weight.shape[1], weight.shape[0], bias=column and bias is not None,
                           dtype=weight.dtype, device='meta')
        result.weight = nn.Parameter(weight, requires_grad=False)
        if column and bias is not None:
            result.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
    return result


class PairColumn(nn.Module):
    def __init__(self, original, exchange, parts, restore_sp):
        super().__init__()
        self.projection = slice_projection(original, exchange, column=True, parts=parts)
        self.exchange, self.parts, self.restore_sp = exchange, parts, restore_sp

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, local):
        value = self.projection(self.exchange.gather(local))
        return self.exchange.scatter_columns(value, self.parts) if self.restore_sp else value


class PairRow(nn.Module):
    def __init__(self, original, exchange, gather_sp):
        super().__init__()
        self.projection = slice_projection(original, exchange, column=False)
        self.register_buffer('bias', original.bias.detach().clone() if original.bias is not None else None)
        self.exchange, self.gather_sp = exchange, gather_sp

    @property
    def weight(self):
        return self.projection.weight

    def forward(self, local):
        if self.gather_sp:
            local = self.exchange.gather(local).chunk(2, dim=-1)[self.exchange.rank].contiguous()
        value = self.exchange.reduce_scatter(self.projection(local))
        return value if self.bias is None else value+self.bias


def prepare_pair_block(block, exchange):
    attn = block.attn
    if not attn.fused_projections:
        raise ValueError('TP2/SP8 expects the prepared fused QKV projection')
    attn.to_qkv = PairColumn(attn.to_qkv, exchange, parts=3, restore_sp=True)
    attn.to_out[0] = PairRow(attn.to_out[0], exchange, gather_sp=True)
    block.ff.net[0].proj = PairColumn(block.ff.net[0].proj, exchange, parts=2, restore_sp=False)
    block.ff.net[2] = PairRow(block.ff.net[2], exchange, gather_sp=False)


def install_pair_parallel(model):
    from x_base.sequence_parallelism.mesh import ParallelManager
    from x_base.tensor_parallelism.pair_sequence import PairSequenceExchange

    if getattr(model, '_h3_pair_installed', False):
        raise ValueError('H3 TP2 projections have already been installed')
    manager = getattr(model, '_h3_pair_manager', None)
    if manager is None:
        manager = ParallelManager(1, 1, 4, 2, register_tp_global=False)
    exchange = getattr(model, '_h3_pair_exchange', None)
    if exchange is None:
        exchange = PairSequenceExchange(manager.tp_group)
    for block in model.transformer_blocks:
        prepare_pair_block(block, exchange)
    model._h3_pair_manager, model._h3_pair_exchange = manager, exchange
    model._h3_pair_installed = True
    logger.info('Experimental H3 linear TP2, SP8 attention unchanged; BF16 pair transport, ranks=%s',
                exchange.metrics['ranks'])
