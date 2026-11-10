"""CUDA defaults, explicit rollback, and resident per-step table selection."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from x_diffusers.adaptor.h3_cuda_options import select_default_recipe, startup_key
from x_diffusers.adaptor.h3_cuda_adaln import resolve_assets, validate_options


def settings(**overrides):
    return SimpleNamespace(**(dict(model='MiniMax-H3', task_type='t2va', sp=8,
        text_encoder_tp=8, h3_cuda_precision='auto', pretrained_model_name_or_path='weights/h3',
        num_inference_steps=20) | overrides))


@pytest.mark.parametrize('task,variant', [('t2va','fl2va'), ('fl2va','fl2va'), ('ref2va','ref2va')])
def test_default_paths_and_schedule_switch_keep_model_startup_key(task,variant):
    args=settings(task_type=task)
    select_default_recipe(args,'cuda')
    assert args.h3_cuda_preset=='sage-fp8-full-parallel'
    resolve_assets(args,'cuda','fp8-sage')
    assert args.h3_cuda_adaln_mode=='table'
    assert args.adaln_cache_file.replace('\\','/').endswith(f'{variant}/tables/adaln_{variant}_bf16_20step.safetensors')
    second=deepcopy(args);second.num_inference_steps=50
    resolve_assets(second,'cuda','fp8-sage')
    assert second.transformer_path==args.transformer_path
    assert second.adaln_cache_file.endswith('50step.safetensors')
    assert startup_key(second)==startup_key(args)


@pytest.mark.parametrize('values', [dict(h3_cuda_preset='none'), dict(h3_cuda_precision='none'),
    dict(h3_cuda_precision='fp8-bf16'), dict(h3_cuda_precision='fp8-sage'), dict(sp=4),
    dict(text_encoder_tp=1), dict(h3_refiner=True), dict(h3_nvfp4_checkpoint='packed'),
    dict(atten='sparge'), dict(matmul_quant='int8'), dict(transformer_path='explicit'),
    dict(h3_features='full-parallel'), dict(h3_parallel_mode='tp_cp')])
def test_explicit_recipes_are_not_overridden(values):
    args=settings(**values)
    before=vars(args).copy()
    select_default_recipe(args,'cuda')
    assert vars(args)==before


@pytest.mark.parametrize('backend,model', [('npu','MiniMax-H3'), ('cuda','Wan2.2-T2V-A14B')])
def test_other_backend_or_model_unchanged(backend,model):
    args=settings(model=model);before=vars(args).copy()
    select_default_recipe(args,backend)
    assert vars(args)==before


def test_missing_default_assets_fail_with_explicit_rollback(tmp_path):
    args=settings(pretrained_model_name_or_path=str(tmp_path))
    select_default_recipe(args,'cuda');resolve_assets(args,'cuda','fp8-sage')
    with pytest.raises(ValueError,match='Missing prepared CUDA AdaLN assets'):
        validate_options(args)
    args.h3_cuda_adaln_mode='owner'
    resolve_assets(args,'cuda','fp8-sage')
    assert not args.transformer_path and not args.adaln_cache_file
    assert validate_options(args) is False


def test_explicit_pair_is_fixed_and_cannot_mix_directory():
    args=settings(h3_cuda_adaln_mode='table',transformer_path='weights/stripped',adaln_cache_file='tables/20.safetensors')
    resolve_assets(args,'cuda','fp8-sage')
    second=deepcopy(args);second.adaln_cache_file='tables/50.safetensors'
    assert startup_key(second)!=startup_key(args)
    args.h3_cuda_adaln_dir='cache'
    with pytest.raises(ValueError,match='not both'):
        resolve_assets(args,'cuda','fp8-sage')
