# SPDX-License-Identifier: Apache-2.0
"""Store each exact BF16 H3 block modulation on one sequence-parallel rank."""

import torch
from torch import nn
import torch.distributed as dist


class H3OwnedAdaLN(nn.Module):
    def __init__(self, original, index, *, group=None, exact_cache=False):
        super().__init__()
        self.group = group
        self.owner = index % dist.get_world_size(group)
        self.hidden_size = original.hidden_size
        self.out_features = original.linear.out_features
        self.output_dtype = original.linear.weight.dtype
        self.linear = original.linear if dist.get_rank(group) == self.owner else None
        self.exact_cache = exact_cache
        self.register_buffer('_last_input', None, persistent=False)
        self.register_buffer('_last_output', None, persistent=False)
        self.projection_calls = 0
        self.cache_hits = 0

    def forward(self, temb):
        if self.linear is not None:
            if (self.exact_cache and self._last_input is not None
                    and temb.dtype==self._last_input.dtype and temb.device==self._last_input.device
                    and torch.equal(temb, self._last_input)):
                output = self._last_output.clone()
                self.cache_hits += 1
            else:
                output = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype)).contiguous()
                self.projection_calls += 1
                if self.exact_cache:
                    self._last_input = temb.detach().clone()
                    self._last_output = output.detach().clone()
        else:
            output = torch.empty((*temb.shape[:-1], self.out_features), device=temb.device, dtype=self.output_dtype)
        dist.broadcast(output, src=dist.get_global_rank(self.group, self.owner) if self.group is not None else self.owner,
                       group=self.group)
        return output.reshape(-1, 6*self.hidden_size).chunk(6, dim=-1)


def check_shared_timestep(module, inputs, output):
    """Owner projection is exact only when all SP ranks share the timestep."""
    reference = output.detach().clone()
    dist.broadcast(reference, src=0)
    mismatch = torch.tensor(int(not torch.equal(output, reference)), device=output.device, dtype=torch.int32)
    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
    if mismatch.item():
        raise RuntimeError("H3 rank-owner AdaLN requires identical timestep embeddings on all SP ranks")
