"""Runtime compatibility; no vLLM import or compiled-op registration."""
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = tl = None
    HAS_TRITON = False

class SiluAndMul(torch.nn.Module):
    def forward(self, x):
        gate, value = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * value
