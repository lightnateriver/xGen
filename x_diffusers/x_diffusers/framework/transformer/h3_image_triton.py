"""Image SwiGLU fusion with explicit rounding after SiLU, matching the BF16 graph."""
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu(x, out, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // WIDTH, offsets % WIDTH
    value = tl.load(x + row * WIDTH * 2 + col, offsets < N, other=0).to(tl.float32)
    gate = tl.load(x + row * WIDTH * 2 + col + WIDTH, offsets < N, other=0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(out + offsets, value * activated, offsets < N)


def image_swiglu(x):
    if x.device.type != "npu" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Image Triton SwiGLU expects contiguous NPU BF16")
    width = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    _swiglu[(triton.cdiv(out.numel(), 1024),)](x, out, out.numel(), width, 1024)
    return out
