"""Table-only mode must reject mismatched assets before loading model weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1] / 'x_diffusers/adaptor'
spec = importlib.util.spec_from_file_location('cuda_adaln', ROOT / 'h3_cuda_adaln.py')
module = importlib.util.module_from_spec(spec)
module.__package__ = 'x_diffusers.adaptor'
spec.loader.exec_module(module)


@pytest.fixture
def assets(tmp_path):
    checkpoint = tmp_path / 'stripped'
    checkpoint.mkdir()
    manifest = dict(model_variant='Ref2VA', adaln_weight_sha256='a'*64, removed_keys=[str(i) for i in range(106)], video_shift=12, audio_shift=3)
    (checkpoint/'.xgen_adaln_source.json').write_text(json.dumps(manifest))
    (checkpoint/'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'proj_in.weight':'weights.safetensors'})))
    metadata = dict(num_steps='21',num_eval_steps='20',num_blocks='50',model_variant='Ref2VA',quantization='none',
                    adaln_weight_sha256='a'*64,video_shift='12.0',audio_shift='3.0')
    table = tmp_path/'table.safetensors'
    save_file({'dummy':torch.zeros(1,dtype=torch.bfloat16)},table,metadata=metadata)
    return SimpleNamespace(transformer_path=str(checkpoint), adaln_cache_file=str(table), task_type='ref2va',
                           h3_parallel_mode='sp',num_inference_steps=20), metadata


def test_disabled_path_preserves_original_configuration():
    assert module.validate_options(SimpleNamespace()) is False


@pytest.mark.parametrize('name', ['transformer_path','adaln_cache_file'])
def test_table_and_stripped_checkpoint_are_paired(assets,name):
    args,_=assets
    setattr(args,name,None)
    with pytest.raises(ValueError,match='require both'):module.validate_options(args)


def test_valid_fixed_schedule_table(assets):
    args,_=assets
    assert module.validate_options(args)


@pytest.mark.parametrize('name,value', [('task_type','t2v'),('h3_parallel_mode','tp_cp')])
def test_unvalidated_routes_are_rejected(assets,name,value):
    args,_=assets;setattr(args,name,value)
    with pytest.raises(ValueError,match='native SP'):module.validate_options(args)


@pytest.mark.parametrize('task,variant', [('t2va','FL2VA'),('fl2va','FL2VA'),('ref2va','Ref2VA')])
def test_workflow_requires_its_own_table_and_stripped_weights(assets,task,variant):
    args,metadata=assets;args.task_type=task
    path=Path(args.transformer_path)/'.xgen_adaln_source.json'
    manifest=json.loads(path.read_text());manifest['model_variant']=variant;path.write_text(json.dumps(manifest))
    metadata['model_variant']=variant
    save_file({'dummy':torch.zeros(1,dtype=torch.bfloat16)},args.adaln_cache_file,metadata=metadata)
    assert module.validate_options(args)
    manifest['model_variant']='Ref2VA' if variant=='FL2VA' else 'FL2VA'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='must match'):module.validate_options(args)


@pytest.mark.parametrize('name,value', [('model_variant','FL2VA'),('num_blocks','49'),('quantization','int8'),
                                     ('adaln_weight_sha256','b'*64),('video_shift','10'),('audio_shift','12')])
def test_table_must_match_model_and_scheduler(assets,name,value):
    args,metadata=assets;metadata[name]=value
    save_file({'dummy':torch.zeros(1,dtype=torch.bfloat16)},args.adaln_cache_file,metadata=metadata)
    with pytest.raises(ValueError,match='must match'):module.validate_options(args)


def test_actual_steps_must_match_table(assets):
    args,_=assets;args.num_inference_steps=50
    with pytest.raises(ValueError,match='evaluation count'):module.validate_options(args)


def test_a_full_checkpoint_cannot_masquerade_as_stripped(assets):
    args,_=assets
    path=Path(args.transformer_path)/'diffusion_pytorch_model.safetensors.index.json'
    path.write_text(json.dumps(dict(weight_map={'transformer_blocks.0.adaln_proj.linear.weight':'weights.safetensors'})))
    with pytest.raises(ValueError,match='physically removed'):module.validate_options(args)


def test_residual_runtime_weights_are_rejected():
    model=SimpleNamespace(named_parameters=lambda:iter([('time_embedder.linear_1.weight',torch.zeros(1))]))
    with pytest.raises(ValueError,match='remain'):module.assert_weights_absent(model)


def test_same_table_and_condition_do_not_reload(assets):
    args,_=assets
    cache=SimpleNamespace(_source_file=str(Path(args.adaln_cache_file).resolve()),_conditioning_type='video')
    model=SimpleNamespace(_h3_cuda_adaln_checkpoint=str(Path(args.transformer_path).resolve()),_adaln_cache=cache)
    pipe=SimpleNamespace(transformer=model,scheduler=SimpleNamespace(config=SimpleNamespace(shift=12)),
                         audio_scheduler=SimpleNamespace(config=SimpleNamespace(shift=3)))
    with patch('x_base.adaln_cache.install_adaln_cache') as install:
        module.select_table(pipe,args,'video')
        install.assert_not_called()
        module.select_table(pipe,args,'both')
        install.assert_called_once_with(pipe,args.adaln_cache_file,conditioning_type='both')


def test_fl2va_first_and_last_frames_share_one_visual_constant(assets):
    args,metadata=assets;args.task_type='fl2va';metadata['model_variant']='FL2VA'
    path=Path(args.transformer_path)/'.xgen_adaln_source.json'
    manifest=json.loads(path.read_text());manifest['model_variant']='FL2VA';path.write_text(json.dumps(manifest))
    save_file({'dummy':torch.zeros(1,dtype=torch.bfloat16)},args.adaln_cache_file,metadata=metadata)
    cache=SimpleNamespace(_source_file=str(Path(args.adaln_cache_file).resolve()),_conditioning_type='none')
    pipe=SimpleNamespace(transformer=SimpleNamespace(_h3_cuda_adaln_checkpoint=str(Path(args.transformer_path).resolve()),_adaln_cache=cache),
        scheduler=SimpleNamespace(config=SimpleNamespace(shift=12)),audio_scheduler=SimpleNamespace(config=SimpleNamespace(shift=3)))
    def installed(*values,conditioning_type):cache._conditioning_type=conditioning_type
    with patch('x_base.adaln_cache.install_adaln_cache',side_effect=installed) as install:
        module.select_table(pipe,args,'video')  # First-frame request, after a T2VA request.
        module.select_table(pipe,args,'video')  # First/last frames share the same visual timestep.
        install.assert_called_once_with(pipe,args.adaln_cache_file,conditioning_type='video')


@pytest.mark.parametrize('condition', ['none','video'])
@pytest.mark.parametrize('conditioned', [False,True])
def test_cache_keeps_legacy_merge_and_complete_condition_batches(tmp_path,condition,conditioned):
    from x_base.adaln_cache import AdalnScheduleCache
    ts=torch.tensor([.1,.2]);const=torch.tensor([.999,1.])
    raw=torch.arange(72,dtype=torch.bfloat16).reshape(2,36)
    final=torch.arange(8,dtype=torch.bfloat16).reshape(2,4)
    const_raw=raw+100;const_final=final+100
    full=torch.cat((ts,const[:1])) if condition=='video' else ts
    expected=torch.cat((raw,const_raw[:1])) if condition=='video' else raw
    expected_final=torch.cat((final,const_final[:1])) if condition=='video' else final
    metadata=dict(num_steps='2',num_eval_steps='1',num_blocks='1',quantization='none')
    if conditioned:
        # Intentionally differ from a split batch. Loader must not reconstruct
        # these complete, jointly evaluated rows from separate constants.
        expected=expected+10;expected_final=expected_final+10
        tensors={f'{condition}_step0_timesteps':full,f'{condition}_step0_block0':expected,
                 f'{condition}_step0_final':expected_final}
        metadata['conditioning_tables']=condition
    else:
        tensors=dict(step0_timesteps=ts,step0_block0=raw,step0_final=final,
                     const_timesteps=const,const_block0=const_raw,const_final=const_final)
    path=tmp_path/'cache.safetensors';save_file(tensors,path,metadata=metadata)
    with patch('x_base.adaln_cache.cache.device_type',return_value='cpu'):
        cache=AdalnScheduleCache(SimpleNamespace(transformer_blocks=[None]),str(path),condition)
    cache.maybe_install(full)
    assert torch.equal(torch.cat(cache.lookup_block(0),dim=-1),expected.reshape(-1,12))
    assert torch.equal(torch.cat(cache.lookup_final(),dim=-1),expected_final)
    if conditioned:
        with patch('x_base.adaln_cache.cache.device_type',return_value='cpu'),pytest.raises(ValueError,match='does not contain'):
            AdalnScheduleCache(SimpleNamespace(transformer_blocks=[None]),str(path),'both')
