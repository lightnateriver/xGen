"""The new recipe is opt-in and validates before reading model weights."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import diffusers.hooks
import inspect
from unittest.mock import Mock
import sys

ROOT=Path(__file__).resolve().parents[1]/'x_diffusers/adaptor'
spec=importlib.util.spec_from_file_location('cuda_precision_config',ROOT/'h3_cuda_precision.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def isolate_chunk_installer():
    with patch.object(module,'__package__','x_diffusers.adaptor'),patch.dict(sys.modules,{
        'x_diffusers.framework.vae.h3_cuda_chunks':SimpleNamespace(install_cpu_decode_chunks=lambda vae:None,
                                                               install_staged_decoder=lambda *args:None),
    }):
        yield


def settings(**values):
    base=dict(h3_cuda_precision='fp8-bf16',model='MiniMax-H3',dtype='bf16',task_type='t2va',
              sp=8,text_encoder_tp=8,atten='none',matmul_quant='none',phaa_num=2,h3_cuda_vae_sp=True)
    return SimpleNamespace(**(base|values))


def test_legacy_validation_does_nothing():
    module.validate(SimpleNamespace(),'npu')


@pytest.mark.parametrize('field,value',[('task_type','ref2va'),('sp',1),('text_encoder_tp',4),('h3_nvfp4_checkpoint','packed'),('h3_refiner',True)])
def test_incompatible_profiles_fail_before_loading(field,value):
    with patch.object(module.dist,'is_initialized',return_value=True),patch.object(module.dist,'get_world_size',return_value=8),patch.dict(module.os.environ,{'XGEN_PHAA_FP8':'0'}):
        with pytest.raises(ValueError): module.validate(settings(**{field:value}),'cuda')


def test_released_profile_requires_sp8():
    with patch.object(module.dist,'is_initialized',return_value=True),patch.object(module.dist,'get_world_size',return_value=4):
        with pytest.raises(ValueError,match='world=SP=text TP=8'):
            module.validate(settings(sp=4,text_encoder_tp=4),'cuda')


@pytest.mark.parametrize('backend,model,task,expected',[
    ('npu','MiniMax-H3','t2va','none'), ('npu','MiniMax-H3','ref2va','none'),
    ('cuda','Wan2.2-T2V-A14B','t2v','none'), ('cuda','MiniMax-H3','ref2va','none'),
    ('cuda','MiniMax-H3','t2va','fp8-bf16'), ('cuda','MiniMax-H3','fl2va','fp8-bf16'),
])
def test_hardware_specific_default(backend,model,task,expected):
    args=settings(h3_cuda_precision='auto',model=model,task_type=task,phaa_num=0)
    with patch.dict(module.os.environ,{},clear=True):
        before=dict(module.os.environ)
        assert module.resolve(args,backend)==expected
        if expected=='none':assert dict(module.os.environ)==before
        else:assert args.phaa_num==2 and args.h3_cuda_vae_sp


@pytest.mark.parametrize('name,value',[('h3_nvfp4_checkpoint','packed'),('h3_refiner',True),('atten','sparge'),('adaln_cache_file','table')])
def test_explicit_legacy_recipes_keep_their_route(name,value):
    args=settings(h3_cuda_precision='auto',**{name:value})
    assert module.resolve(args,'cuda')=='none'


def test_explicit_opt_out_and_npu_rejection():
    assert module.resolve(settings(h3_cuda_precision='none'),'cuda')=='none'
    with pytest.raises(ValueError,match='requires CUDA'):
        module.resolve(settings(),'npu')


def test_resident_http_profile_cannot_silently_change():
    manager=SimpleNamespace(pipe=object(),_h3_cuda_precision_key='fp8-bf16')
    with pytest.raises(ValueError,match='Restart the service'):
        module.validate_manager(manager,settings(h3_cuda_precision='none'),'cuda')


def test_sage_is_explicit_and_keeps_npu_rejected():
    args=settings(h3_cuda_precision='fp8-sage',phaa_num=0)
    assert module.resolve(args,'cuda')=='fp8-sage'
    assert args.phaa_num==2 and args.h3_cuda_vae_sp
    with pytest.raises(ValueError,match='requires CUDA'):
        module.resolve(settings(h3_cuda_precision='fp8-sage'),'npu')


def test_sage_can_reuse_fp8_transport_but_bf16_profile_still_rejects_it():
    with patch.object(module.dist,'is_initialized',return_value=True),patch.object(module.dist,'get_world_size',return_value=8),patch.dict(module.os.environ,{'XGEN_PHAA_FP8':'1'}):
        module.validate(settings(h3_cuda_precision='fp8-sage'),'cuda')
        with pytest.raises(ValueError,match='requires BF16 communication'):
            module.validate(settings(),'cuda')


def test_resident_sage_profile_cannot_silently_change():
    manager=SimpleNamespace(pipe=object(),_h3_cuda_precision_key='fp8-sage')
    with pytest.raises(ValueError,match='Restart the service'):
        module.validate_manager(manager,settings(),'cuda')


def test_precision_argument_default_is_hardware_auto():
    tree=ast.parse((ROOT/'config.py').read_text(encoding='utf-8'))
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--h3_cuda_precision']
    assert len(matches)==1
    assert next(value.value.value for value in matches[0].keywords if value.arg=='default')=='auto'


def test_decode_trim_preserves_signature_and_never_moves_dit():
    def decode(latent,return_dict=True): return latent,return_dict
    audio=Mock(decode=decode)
    audio.modules.return_value=[]
    pipe=SimpleNamespace(vae=SimpleNamespace(decode=decode),audio_vae=audio,transformer=Mock())
    args=settings()
    with patch('diffusers.hooks.apply_group_offloading'),patch.object(module.torch.cuda,'synchronize') as sync,\
            patch.object(module.torch.cuda,'empty_cache') as reclaim,\
            patch.object(module.torch.cuda,'memory_reserved',side_effect=[100,40]),\
            patch.object(module.torch.cuda,'memory_allocated',return_value=10):
        module.prepare_vaes(pipe,args,'cuda')
        assert inspect.signature(pipe.vae.decode)==inspect.signature(decode)
        assert pipe.vae.decode(7,return_dict=False)==(7,False)
    sync.assert_called_once();reclaim.assert_called_once()
    pipe.transformer.to.assert_not_called()
    assert pipe._h3_cuda_decode_trim['reserved_before']==100 and pipe._h3_cuda_decode_trim['reserved_after']==40


@pytest.mark.parametrize('fails',(False,True))
def test_audio_moves_whole_module_and_returns_to_cpu(fails):
    def decode(latent,return_dict=True):
        if fails: raise RuntimeError('injected audio failure')
        return latent,return_dict
    audio=Mock(decode=decode)
    audio.modules.return_value=[]
    pipe=SimpleNamespace(vae=Mock(),audio_vae=audio,transformer=Mock())
    with patch('diffusers.hooks.apply_group_offloading') as offload:
        module.prepare_vaes(pipe,settings(),'cuda:0')
    assert offload.call_count==1 and offload.call_args.args[0] is pipe.vae
    assert inspect.signature(audio.decode)==inspect.signature(decode)
    audio.to.reset_mock()
    if fails:
        with pytest.raises(RuntimeError,match='injected audio failure'): audio.decode(7)
    else:
        assert audio.decode(7,return_dict=False)==(7,False)
    assert [str(call.args[0]) for call in audio.to.call_args_list]==['cuda:0','cpu']
    pipe.transformer.to.assert_not_called()


def test_audio_offload_ignores_unused_meta_weight_cache():
    child=SimpleNamespace(weight_g=module.torch.ones(1),weight_v=module.torch.ones(1),
                          weight=module.torch.empty(1,device='meta'))
    audio=Mock()
    audio.modules.return_value=[child]
    module.offload_audio(audio)
    assert child.weight.device.type=='meta'
    audio.to.assert_called_once()


@pytest.mark.skipif(not module.torch.cuda.is_available(),reason='CUDA required')
def test_weight_norm_audio_decode_can_be_repeated_without_device_or_cache_leak():
    torch=module.torch
    class Audio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=torch.nn.utils.weight_norm(torch.nn.Conv1d(4,8,3,padding=1))
        def decode(self,latent,return_dict=True): return self.conv(latent)
    audio=Audio().eval()
    latent=torch.randn(1,4,16,device='cuda')
    with torch.no_grad():
        reference=audio.cuda().decode(latent).cpu()
        audio.cpu()
        pipe=SimpleNamespace(vae=Mock(),audio_vae=audio)
        with patch('diffusers.hooks.apply_group_offloading'):
            module.prepare_vaes(pipe,settings(),torch.device('cuda',torch.cuda.current_device()))
        for _ in range(2):
            output=audio.decode(latent)
            assert torch.equal(output.cpu(),reference)
            assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in audio.parameters())
            assert audio.conv.weight.device.type=='cpu'
