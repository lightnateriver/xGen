"""Preset expansion, explicit off switches and immutable resident configuration."""
import argparse
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from x_diffusers.adaptor import h3_cuda_options as options
from x_diffusers.adaptor import h3_cuda_precision as precision


def settings(**overrides):
    return SimpleNamespace(**(dict(model='MiniMax-H3',task_type='ref2va',h3_cuda_precision='fp8-sage',
        dtype='bf16',sp=8,text_encoder_tp=8,atten='none',matmul_quant='none',phaa_num=2,
        h3_cuda_vae_sp=True,h3_cuda_preset='sage-fp8-full-parallel',h3_vae_backend='legacy') | overrides))


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os
    with patch.dict(os.environ):
        for key in ('XGEN_PHAA_FP8','XGEN_PHAA_FP8_MIXED_STRIDES'):
            os.environ.pop(key,None)
        yield


def test_preset_and_complete_explicit_off():
    args=settings(); options.resolve_options(args,'cuda')
    assert all(getattr(args,k) for k in options.BOOL_OPTIONS if not k.endswith('offload'))
    assert args.h3_cuda_dit_residency=='resident'
    assert args.h3_cuda_input_refiner=='cudnn-sp-fp8'
    assert args.h3_cuda_audio_decoder_parallel=='long'
    off={key:False for key in options.BOOL_OPTIONS}
    args=settings(**off,h3_cuda_input_refiner_attention='legacy',h3_cuda_input_refiner_ffn='bf16',
                  h3_cuda_audio_decoder_parallel='none')
    options.resolve_options(args,'cuda')
    assert not any(getattr(args,k) for k in options.BOOL_OPTIONS)
    assert args.h3_cuda_input_refiner=='none'
    assert args.h3_cuda_audio_decoder_parallel=='none'


@pytest.mark.parametrize('sp',[False,True])
@pytest.mark.parametrize('attention',['legacy','cudnn'])
@pytest.mark.parametrize('ffn',['bf16','fp8'])
def test_refiner_axes_independent_and_idempotent(sp,attention,ffn):
    args=settings(h3_cuda_input_refiner_sp=sp,h3_cuda_input_refiner_attention=attention,
                  h3_cuda_input_refiner_ffn=ffn)
    precision.resolve(args,'cuda'); before=options.startup_key(args)
    precision.resolve(args,'cuda'); assert options.startup_key(args)==before
    assert (args.h3_cuda_input_refiner_sp,args.h3_cuda_input_refiner_attention,args.h3_cuda_input_refiner_ffn)==(sp,attention,ffn)


@pytest.mark.parametrize('backend,model,task', [('npu','MiniMax-H3','ref2va'),('cuda','Wan2.2-T2V-A14B','t2v'),('npu','Wan2.2-T2V-A14B','t2v')])
def test_old_default_does_not_install_new_features(backend,model,task):
    args=settings(model=model,task_type=task,h3_cuda_preset=None,h3_cuda_precision='auto')
    assert precision.resolve(args,backend)=='none'
    assert args.h3_cuda_input_refiner=='none'
    assert not any(getattr(args,k) for k in options.BOOL_OPTIONS)


def test_parser_preserves_explicit_false_for_both_entries():
    parser=argparse.ArgumentParser(); options.add_arguments(parser,service=True)
    args=parser.parse_args(['--h3_cuda_preset','sage-fp8-full-parallel','--no-h3_cuda_local_output',
                           '--no-h3_cuda_precompute_conditioning','--h3_cuda_input_refiner_ffn','bf16'])
    merged=settings(**vars(args)); options.resolve_options(merged,'cuda')
    assert merged.h3_cuda_local_input and not merged.h3_cuda_local_output
    assert not merged.h3_cuda_precompute_conditioning and merged.h3_cuda_input_refiner_ffn=='bf16'


@pytest.mark.parametrize('values',[
    dict(h3_cuda_dit_residency='resident',h3_cuda_conditioning_offload=True),
    dict(h3_cuda_phaa_fp8=False,h3_cuda_phaa_direct_pack=True),
    dict(h3_parallel_mode='tp_cp_shared',h3_cuda_phaa_direct_pack=True),
    dict(h3_cp_comm='bf16',h3_parallel_mode='tp_cp',h3_cuda_phaa_fp8=True),
    dict(h3_cuda_local_input=1),dict(h3_cuda_input_refiner_sp='false'),
])
def test_invalid_combinations_do_not_mutate_environment(values):
    import os
    before=dict(os.environ)
    with pytest.raises(ValueError): options.resolve_options(settings(**values),'cuda')
    assert dict(os.environ)==before


@pytest.mark.parametrize('residency',['resident','conditioning','reference'])
def test_residency_modes_are_exclusive(residency):
    args=settings(h3_cuda_dit_residency=residency); options.resolve_options(args,'cuda')
    assert args.h3_cuda_conditioning_offload==(residency!='resident')
    assert args.h3_cuda_reference_offload==(residency=='reference')


def test_all_startup_switches_cannot_change_after_loading_but_inputs_can():
    args=settings(h3_cuda_adaln_mode='owner'); manager=SimpleNamespace(pipe=None)
    with patch.object(precision.dist,'is_initialized',return_value=True),patch.object(precision.dist,'get_world_size',return_value=8):
        precision.validate_manager(manager,args,'cuda'); manager.pipe=object()
        per_request=deepcopy(args);per_request.frames=243;per_request.prompt='another prompt';per_request.reference_paths=['other.png']
        precision.validate_manager(manager,per_request,'cuda')
        for key in options.BOOL_OPTIONS:
            requested=deepcopy(args);setattr(requested,key,not getattr(args,key))
            with pytest.raises(ValueError): precision.validate_manager(manager,requested,'cuda')


def test_alias_canonicalized_and_transport_independent():
    args=settings(h3_cuda_preset=None,h3_cuda_linear_tp=2,h3_tp_comm='fp8',h3_cp_comm='bf16')
    options.resolve_options(args,'cuda')
    assert (args.h3_parallel_mode,args.h3_dit_tp,args.h3_cuda_linear_tp)==('tp_sp',2,1)
    assert not args.h3_cuda_phaa_fp8


@pytest.mark.parametrize('mode,expected',[('sp',False),('tp_sp',False),('tp_cp',True),('tp_cp_shared',True)])
def test_direct_adapter_and_resolver_share_legacy_local_io_defaults(mode,expected):
    assert options.local_io_default(mode)==expected
    args=settings(h3_cuda_preset=None,h3_parallel_mode=mode)
    options.resolve_options(args,'cuda')
    assert args.h3_cuda_local_input==args.h3_cuda_local_output==expected


def test_real_offline_parser_bootstraps_preset_without_device_allocation(monkeypatch):
    import os,sys,torch
    from x_diffusers.adaptor.config import parse_args
    from x_base import device
    monkeypatch.setattr(device,'device_type',lambda:'cuda')
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('Parsing allocated a CUDA context'))
    monkeypatch.setattr(sys,'argv',['infer.py','--model','MiniMax-H3','--task_type','ref2va',
        '--sp','8','--text_encoder_tp','8','--h3_cuda_preset','sage-fp8-full-parallel',
        '--no-h3_cuda_local_output','--h3_cuda_input_refiner_ffn','bf16'])
    args=parse_args()
    assert args.h3_cuda_precision=='fp8-sage' and args.h3_cuda_local_input
    from x_diffusers.adaptor.h3_vae_config import resolve_vae_backend
    assert resolve_vae_backend(args)=='native-stream'
    assert not args.h3_cuda_local_output and args.h3_cuda_input_refiner_ffn=='bf16'
    assert os.environ['XGEN_PHAA_FP8']==os.environ['XGEN_PHAA_FP8_MIXED_STRIDES']=='1'


@pytest.mark.parametrize('backend,model',[('cuda','Wan2.2-T2V-A14B'),('npu','Wan2.2-T2V-A14B'),('npu','MiniMax-H3')])
def test_legacy_non_fp8_manager_keeps_original_model_rebuild_contract(backend,model):
    manager=SimpleNamespace(pipe=object())
    with patch.object(precision.dist,'is_initialized',return_value=False):
        for weights in ('weights/original','weights/another'):
            args=settings(model=model,task_type='t2va' if model=='MiniMax-H3' else 't2v',
                h3_cuda_preset=None,h3_cuda_precision='auto',pretrained_model_name_or_path=weights)
            assert precision.resolve(args,backend)=='none'
            precision.validate_manager(manager,args,backend)
            assert not hasattr(manager,'_h3_cuda_options_key')


@pytest.mark.parametrize('backend',['cuda','npu'])
def test_non_h3_unspecified_decoder_preserves_resident_service(backend):
    manager=SimpleNamespace(pipe=object())
    args=settings(model='Wan2.2-T2V-A14B',task_type='t2v',h3_cuda_preset=None,
                  h3_cuda_precision='auto',h3_vae_backend=None,vae_lightning='encoder_and_decoder')
    precision.validate_manager(manager,args,backend)
    assert args.h3_vae_backend=='legacy' and args.vae_lightning=='encoder_and_decoder'
