"""Check eager BF16 rounding with strided AdaLN tables."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_indexed_affine_keeps_bf16_intermediates(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("XGEN_DEVICE", "cuda")
    from x_base.operator.h3_indexed_affine import _apply

    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn((2, 37, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    residual = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    tables = torch.randn((7, 6 * x.shape[-1]), device=x.device, dtype=x.dtype, generator=generator)
    shift, scale, gate, *_ = tables.chunk(6, dim=-1)
    indices = torch.randint(7, (x.shape[1],), device=x.device, generator=generator)

    expected = x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    assert torch.equal(_apply(x, scale, shift, indices, modulation=True), expected)
    expected = residual + gate.index_select(0, indices) * x
    assert torch.equal(_apply(x, gate, residual, indices, modulation=False), expected)
