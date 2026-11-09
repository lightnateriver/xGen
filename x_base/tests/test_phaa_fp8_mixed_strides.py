"""Opt-in PHAA packing must preserve the existing byte-level wire protocol."""
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ["mixed", "uniform", "last_stride_two", "single"])
@pytest.mark.parametrize("batch", [1, 2])
def test_mixed_stride_transport_bit_exact(monkeypatch, layout, batch):
    pytest.importorskip("triton")
    from x_base.phaa import fp8_transport as transport

    peers, seq, heads, width = 2, 17, 14, 96
    generator = torch.Generator(device="cuda").manual_seed(24)
    fused = torch.randn((batch, seq, 3 * heads * width), device="cuda",
                        dtype=torch.bfloat16, generator=generator)
    q, k, v = fused.chunk(3, dim=-1)
    if layout != "uniform":
        q, k = q.contiguous(), k.contiguous()
    if layout == "last_stride_two":
        v = v.repeat_interleave(2, dim=-1)[..., ::2]
    inputs = tuple(t.unflatten(-1, (peers, heads // peers, width))
                   [:, :, :, 4:, :].permute(2, 1, 0, 3, 4) for t in (q, k, v))
    if layout == "single":
        inputs = inputs[:1]
    wires = []

    def exchange(output, tensor, **kwargs):
        wires.append(tensor.clone())
        output.copy_(tensor)

    before = transport.mixed_stride_metrics()
    with patch.object(dist, "get_world_size", return_value=peers), \
         patch.object(dist, "all_to_all_single", side_effect=exchange):
        monkeypatch.setenv("XGEN_PHAA_FP8_MIXED_STRIDES", "0")
        old = tuple(torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in inputs)
        transport.fp8_all_to_all_many(old, inputs, width, None)
        assert transport.mixed_stride_metrics() == before
        monkeypatch.setenv("XGEN_PHAA_FP8_MIXED_STRIDES", "1")
        new = tuple(torch.empty_like(t) for t in old)
        transport.fp8_all_to_all_many(new, inputs, width, None)
    assert torch.equal(wires[0], wires[1])
    assert all(torch.equal(a, b) for a, b in zip(old, new))
    expected = 1 if layout == "mixed" else 0
    assert transport.mixed_stride_metrics()['calls'] - before['calls'] == expected
