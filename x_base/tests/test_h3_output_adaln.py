"""Verify output AdaLN dispatch, exact BF16 rounding and reduced temporary storage."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
tree=ast.parse((ROOT/'x_diffusers/x_diffusers/framework/transformer/minimax_h3.py').read_text(encoding='utf-8'))
definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_patched_adaln_out_forward')
scope={'torch':torch,'nn':nn}
exec(compile(ast.Module(body=[definition],type_ignores=[]),'<output-adaln>','exec'),scope)
forward=scope[definition.name]


def layer(width,device):
    return SimpleNamespace(norm=nn.LayerNorm(width,elementwise_affine=False).to(device),
                           linear=nn.Linear(16,width*2,bias=False).to(device=device,dtype=torch.bfloat16))


def test_cpu_flag_retains_original_arithmetic():
    module=layer(32,'cpu');x=torch.randn(1,17,32,dtype=torch.bfloat16)
    temb=torch.randn(3,16,dtype=torch.bfloat16);indices=torch.arange(17)%3
    with torch.no_grad():
        expected=forward(module,x,temb,indices)
        module._h3_fused_indexed_affine=True
        assert torch.equal(forward(module,x,temb,indices),expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('cached',(False,True))
@pytest.mark.parametrize('batch,width,seq',((1,5376,2003),(2,512,137)))
def test_cuda_output_adaln_is_bitwise_equal(cached,batch,width,seq):
    torch.manual_seed(42)
    module=layer(width,'cuda');x=torch.randn(batch,seq,width,device='cuda',dtype=torch.bfloat16)
    temb=torch.randn(3,16,device='cuda',dtype=torch.bfloat16);indices=torch.arange(seq,device='cuda')%3
    if cached:
        # Chunk views have nontrivial row strides, as does the real modulation table.
        modulation=torch.randn(3,width*2,device='cuda',dtype=torch.bfloat16)
        module._adaln_cache=SimpleNamespace(has_current=True,lookup_final=lambda:modulation.chunk(2,dim=-1))
    with torch.no_grad():
        expected=forward(module,x,temb,indices)
        module._h3_fused_indexed_affine=True
        actual=forward(module,x,temb,indices)
    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fused_output_avoids_expanded_full_sequence_temporaries():
    module=layer(5376,'cuda');module.norm=nn.Identity()
    x=torch.randn(1,32768,5376,device='cuda',dtype=torch.bfloat16)
    temb=torch.randn(3,16,device='cuda',dtype=torch.bfloat16);indices=torch.arange(32768,device='cuda')%3
    with torch.no_grad():
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        expected=forward(module,x,temb,indices)
        torch.cuda.synchronize();eager_peak=torch.cuda.max_memory_allocated()-base
        reference=expected.cpu();del expected
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        module._h3_fused_indexed_affine=True
        actual=forward(module,x,temb,indices)
        torch.cuda.synchronize();fused_peak=torch.cuda.max_memory_allocated()-base
    assert torch.equal(actual.cpu().view(torch.int16),reference.view(torch.int16))
    assert fused_peak<eager_peak/2,(eager_peak,fused_peak)
    print(dict(eager_extra_bytes=eager_peak,fused_extra_bytes=fused_peak))


def test_installer_is_opt_in():
    from x_diffusers.adaptor.h3_cuda import install_dit_fusions
    model=SimpleNamespace(norm_out=SimpleNamespace())
    install_dit_fusions(model,frozenset())
    assert not hasattr(model.norm_out,'_h3_fused_indexed_affine')
    install_dit_fusions(model,frozenset({'XGEN_H3_FUSED_OUTPUT_ADALN'}))
    assert model.norm_out._h3_fused_indexed_affine
