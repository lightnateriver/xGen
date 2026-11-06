"""Numerical GPU tests for the opt-in H3 operators (no checkpoint needed)."""
import pytest
import torch

from x_base.operator.h3_fp8 import H3FP8Linear, quantize_per_tensor


def test_zero_quantization_is_finite():
    packed, scale = quantize_per_tensor(torch.zeros(16,32))
    assert scale.item() > 0 and torch.isfinite(packed.float()).all()
    assert torch.equal(packed.float(),torch.zeros(16,32))


def test_projection_rejects_unaligned_dimensions():
    with pytest.raises(ValueError,match='divisible by 16'):
        H3FP8Linear(torch.randn(17,32))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fp8_projection_bias_and_unaligned_rows():
    torch.manual_seed(42)
    weight=torch.randn(256,128,dtype=torch.bfloat16)
    bias=torch.randn(256,dtype=torch.bfloat16)
    layer=H3FP8Linear(weight,bias).cuda()
    x=torch.randn(2,19,128,device='cuda',dtype=torch.bfloat16)
    result=layer(x)
    reference=torch.nn.functional.linear(x,weight.cuda(),bias.cuda())
    assert result.shape==reference.shape and result.dtype==reference.dtype
    assert layer.weight.dtype==torch.float8_e4m3fn
    assert torch.isfinite(result).all()
    relative=(result.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt()
    assert relative.item()<0.08


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_bf16_cudnn_backend_with_phaa_strides():
    from x_base.operator.h3_dense_cuda import H3DenseCUDA
    from torch.nn.attention import sdpa_kernel,SDPBackend

    q,k,v=[torch.randn(257,2,128,device='cuda',dtype=torch.bfloat16).permute(1,0,2).unsqueeze(0) for _ in range(3)]
    result=H3DenseCUDA()(q,k,v)
    with sdpa_kernel(SDPBackend.MATH): reference=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    assert result.shape==reference.shape and torch.isfinite(result).all()
    relative=(result.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt()
    assert relative.item()<0.02


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',(torch.float16,torch.bfloat16))
@pytest.mark.parametrize('shape',((32,128),(997,5376),(2048,14336)))
def test_fused_fp8_packing_matches_native(dtype,shape):
    from x_base.operator.h3_fp8_quant import quantize_activation
    value=torch.randn(shape,device='cuda',dtype=dtype)
    for sample in (value,torch.zeros_like(value)):
        expected,scale=quantize_per_tensor(sample)
        actual,actual_scale=quantize_activation(sample)
        assert torch.equal(actual_scale,scale)
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
