"""Token-wise chunk parity, tails, failures and no retained output tensors."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.attention import FeedForward

path=Path(__file__).resolve().parents[1]/'x_diffusers/adaptor/h3_cuda_refiner.py'
spec=importlib.util.spec_from_file_location('input_refiner_chunk_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('tokens',[1,7,8,9,16,19])
def test_native_ffn_matches_unchunked_with_tail(tokens):
    torch.manual_seed(42)
    ff=FeedForward(32,inner_dim=64,activation_fn='swiglu',bias=False).double().eval()
    model=SimpleNamespace(token_refiner=SimpleNamespace(refiner_blocks=[SimpleNamespace(ff=ff)]))
    values=torch.randn(2,tokens,32,dtype=torch.float64)
    parameters=[id(p) for p in ff.parameters()]
    with torch.inference_mode():expected=ff(values)
    module.install_ff_chunking(model,8)
    with torch.inference_mode():actual=ff(values)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert [id(p) for p in ff.parameters()]==parameters
    stats=model._h3_refiner_ff_chunking
    assert stats['calls']==stats['completed']==1
    assert stats['chunks_completed']==(tokens+7)//8
    assert stats['max_input_tokens']==tokens
    assert all(not isinstance(value,torch.Tensor) for value in stats.values())
    with pytest.raises(ValueError,match='already installed'):module.install_ff_chunking(model,8)
    with pytest.raises(ValueError,match='inference-only'):ff(values)


def test_original_failure_is_preserved_without_retry():
    error=RuntimeError('injected FFN failure')
    calls=[]
    def original(value):
        calls.append(value.shape[1]);raise error
    ff=SimpleNamespace(forward=original)
    model=SimpleNamespace(token_refiner=SimpleNamespace(refiner_blocks=[SimpleNamespace(ff=ff)]))
    module.install_ff_chunking(model,8)
    with torch.inference_mode(),pytest.raises(RuntimeError) as caught:ff.forward(torch.ones(1,19,4))
    assert caught.value is error and calls==[8]
    assert model._h3_refiner_ff_chunking['completed']==0

