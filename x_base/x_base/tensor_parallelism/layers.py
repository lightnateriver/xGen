"""TP 线性层实现。

ColumnParallelLinear: 列并行，按输出维度切分，forward 无通信
RowParallelLinear: 行并行，按输入维度切分，forward 需 all_reduce
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .state import get_tp_group, get_tp_world_size, get_tp_rank


class ColumnParallelLinear(nn.Module):
    """列并行线性层 — 按输出维度 (row of weight) 切分

    用途: q_proj, k_proj, v_proj, gate_proj, up_proj, lm_head
    forward: 无需通信，每张卡只计算自己那部分输出

    GQA 兼容性: 权重按行排列 (out_features, in_features)，均匀行切分天然等价于
    按 head 切分。Q 权重有 num_heads × head_dim 行，K/V 权重有 num_kv_heads ×
    head_dim 行，均匀切分即按 head/kv_head 切分，无需特殊处理。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gather_output: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        tp_size = get_tp_world_size()

        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features {out_features} not divisible by tp_size {tp_size}"
            )
        self.out_features_per_partition = out_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(x.device)
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output and get_tp_world_size() > 1:
            # all_gather along output dim
            world_size = get_tp_world_size()
            gather_list = [torch.empty_like(output) for _ in range(world_size)]
            dist.all_gather(gather_list, output, group=get_tp_group())
            output = torch.cat(gather_list, dim=-1)
        return output

    @torch.no_grad()
    def load_shard(self, full_weight: torch.Tensor, full_bias: torch.Tensor = None):
        """从完整权重中提取本 rank 的分片

        权重按行排列 (out_features, in_features)，均匀行切分天然等价于按 head 切分：
        - Q 权重有 num_heads × head_dim 行 → 均匀切分 = 按 head 切分
        - K/V 权重有 num_kv_heads × head_dim 行 → 均匀切分 = 按 kv_head 切分
        因此 GQA 无需特殊处理类，ColumnParallelLinear 的均匀切分已满足需求。
        """
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        chunk = full_weight.shape[0] // tp_size
        self.weight.data = self.weight.data.to(device=full_weight.device)
        self.weight.data.copy_(
            full_weight[tp_rank * chunk: (tp_rank + 1) * chunk]
        )
        if full_bias is not None and self.bias is not None:
            self.bias.data = self.bias.data.to(device=full_weight.device)
            self.bias.data.copy_(
                full_bias[tp_rank * chunk: (tp_rank + 1) * chunk]
            )


class RowParallelLinear(nn.Module):
    """行并行线性层 — 按输入维度 (col of weight) 切分

    用途: o_proj, down_proj
    forward: 输入已是分片的，输出需要 all_reduce
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 input_is_parallel: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        tp_size = get_tp_world_size()

        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by tp_size {tp_size}"
            )
        self.in_features_per_partition = in_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(x.device)
        if not self.input_is_parallel and get_tp_world_size() > 1:
            tp_rank = get_tp_rank()
            tp_size = get_tp_world_size()
            chunk = x.shape[-1] // tp_size
            x = x[..., tp_rank * chunk: (tp_rank + 1) * chunk]

        # partial sum — 需要后续 all_reduce
        output = F.linear(x, self.weight)
        if get_tp_world_size() > 1:
            dist.all_reduce(output, group=get_tp_group())
        if self.bias is not None:
            output = output + self.bias
        return output

    @torch.no_grad()
    def load_shard(self, full_weight: torch.Tensor, full_bias: torch.Tensor = None):
        """从完整权重中提取本 rank 的分片"""
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        chunk = full_weight.shape[1] // tp_size
        self.weight.data = self.weight.data.to(device=full_weight.device)
        self.weight.data.copy_(
            full_weight[:, tp_rank * chunk: (tp_rank + 1) * chunk]
        )
        if full_bias is not None and self.bias is not None:
            self.bias.data = self.bias.data.to(device=full_weight.device)
            self.bias.data.copy_(full_bias)  # bias 不切分
