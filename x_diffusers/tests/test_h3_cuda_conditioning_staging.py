"""Device-transition and error contract without loading an H3 checkpoint."""
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'x_diffusers/adaptor/h3_cuda_staging.py'
spec=importlib.util.spec_from_file_location('h3_conditioning_staging_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('fails',[False,True])
def test_stages_existing_weights_and_preserves_original_encoder_failure(fails):
    moves=[]
    model=SimpleNamespace(to=lambda device:moves.append(str(device)))
    error=RuntimeError('injected encoder failure')
    def encode(value,return_dict=True):
        assert moves==['cpu']
        if fails:raise error
        return value+1,return_dict
    encoder=SimpleNamespace(forward=encode)
    pipe=SimpleNamespace(transformer=model,text_encoder=SimpleNamespace(model=encoder))
    module.install_conditioning_staging(pipe,'cuda:0')
    assert inspect.signature(encoder.forward)==inspect.signature(encode)
    with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'memory_allocated',return_value=0):
        if fails:
            with pytest.raises(RuntimeError) as caught:encoder.forward(7)
            assert caught.value is error
            assert moves==['cpu']
            assert pipe._h3_cuda_conditioning_staging['last']['status']=='encoder_failed_dit_on_cpu'
        else:
            assert encoder.forward(7,return_dict=False)==(8,False)
            assert moves==['cpu','cuda:0']
            assert pipe._h3_cuda_conditioning_staging['last']['status']=='completed'
    assert pipe._h3_cuda_conditioning_staging['calls']==1
    with pytest.raises(ValueError,match='already installed'):module.install_conditioning_staging(pipe,'cuda:0')


@pytest.mark.parametrize('execution_device',['cuda:0','cpu'])
def test_deferred_reference_stage_keeps_execution_device_and_restores_once(execution_device):
    moves=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.place='cuda:0'
        def to(self,device):
            self.place=str(device);moves.append(self.place);return self
        def forward(self,value):
            assert self.place=='cuda:0'
            return value+3
    model=Model()
    def encode(value):
        assert model.place=='cpu'
        return value+1
    pipe=SimpleNamespace(transformer=model,text_encoder=SimpleNamespace(model=SimpleNamespace(forward=encode)),
                         _execution_device=execution_device)
    module.install_conditioning_staging(pipe,'cuda:0',defer_until_dit=True)
    with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'memory_allocated',return_value=0):
        if execution_device=='cpu':
            with pytest.raises(RuntimeError,match='pipeline execution device'):pipe.text_encoder.model.forward(7)
            assert moves==['cpu']
            return
        for _ in range(2):
            assert pipe.text_encoder.model.forward(7)==8
            assert model.place=='cpu'  # References execute at this original boundary.
            assert pipe._h3_cuda_conditioning_staging['last']['status']=='awaiting_reference_encoding'
            assert model(9)==12
            assert model(10)==13  # The second DiT step must not stage weights again.
    assert moves==['cpu','cuda:0','cpu','cuda:0']
    assert pipe._h3_cuda_conditioning_staging['calls']==2
    assert pipe._h3_cuda_conditioning_staging['last']['status']=='completed'


def test_deferred_reference_encoder_error_keeps_original_error_and_cpu_weights():
    error=RuntimeError('reference-stage encoder failure')
    moves=[]
    model=torch.nn.Identity()
    model.to=lambda device:moves.append(str(device))
    def encode():raise error
    pipe=SimpleNamespace(transformer=model,text_encoder=SimpleNamespace(model=SimpleNamespace(forward=encode)),
                         _execution_device='cuda:0')
    module.install_conditioning_staging(pipe,'cuda:0',defer_until_dit=True)
    with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'memory_allocated',return_value=0):
        with pytest.raises(RuntimeError) as caught:pipe.text_encoder.model.forward()
    assert caught.value is error and moves==['cpu']
    assert pipe._h3_cuda_conditioning_staging['last']['status']=='encoder_failed_dit_on_cpu'
