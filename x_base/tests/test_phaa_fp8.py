"""Exercise PHAA layout and CUDA streams with a simulated SP collective."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_fp8_phaa_uneven_heads_and_padding(monkeypatch, batch_size):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native FP8 conversion required")
    pytest.importorskip("triton")
    monkeypatch.setenv("XGEN_DEVICE", "cuda")
    from x_base.phaa import phaa_attention

    monkeypatch.setattr(phaa_attention, "get_pad", lambda _: 1)
    generator = torch.Generator(device="cuda").manual_seed(42)
    qkv = [torch.randn((batch_size, 32, 14 * 64), dtype=torch.bfloat16, device="cuda",
                       generator=generator) for _ in range(3)]

    def attention(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q, k, v)

    def local_copy(output, tensor, **kwargs):
        output.copy_(tensor)

    options = dict(comm_stream=torch.cuda.Stream(), sp_group=None, sp_size=2,
                   heads=14, phaa_split_num=2, head_dim=64, attention_fn=attention)
    with patch.object(dist, "get_world_size", return_value=2), \
         patch.object(dist, "all_to_all_single", side_effect=local_copy) as exchange:
        dense = phaa_attention.run_phaa_attention(*qkv, **options)
        assert exchange.call_count == 8
        exchange.reset_mock()
        fp8 = phaa_attention.run_phaa_attention(*qkv, **options, fp8_transport=True)
        assert exchange.call_count == 4
        exchange.reset_mock()
        gather_first = phaa_attention.run_phaa_attention(
            *qkv, **options, fp8_transport=True, gather_priority=True,
        )
        assert exchange.call_count == 4
        exchange.reset_mock()
        scatter_group = object()
        dual = phaa_attention.run_phaa_attention(
            *qkv, **options, fp8_transport=True,
            scatter_group=scatter_group, scatter_stream=torch.cuda.Stream(),
        )
        assert [c.kwargs["group"] for c in exchange.call_args_list] == [
            None, None, scatter_group, scatter_group,
        ]

    assert dense.shape == fp8.shape == qkv[0].shape
    assert torch.isfinite(fp8).all()
    assert torch.equal(fp8, gather_first)
    assert torch.equal(fp8, dual)
    assert F.cosine_similarity(dense.float().flatten(), fp8.float().flatten(), dim=0) > 0.99
