"""Portable option contracts without accelerator allocation."""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'x_diffusers/adaptor'
spec = importlib.util.spec_from_file_location('portable_features', ROOT/'h3_features.py')
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def test_cli_old_and_new_names_use_one_destination_and_explicit_false():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h3_cuda_local_input', action=argparse.BooleanOptionalAction)
    features.add_aliases(parser)
    assert parser.parse_args(['--h3_local_input']).h3_cuda_local_input is True
    assert parser.parse_args(['--no-h3_local_input']).h3_cuda_local_input is False
    assert parser.parse_args(['--h3_cuda_local_input', '--no-h3_local_input']).h3_cuda_local_input is False


def test_json_alias_conflicts_are_not_silently_ignored():
    assert features.normalize_mapping({'h3_local_output': False}) == {'h3_cuda_local_output': False}
    with pytest.raises(ValueError, match='Conflicting'):
        features.normalize_mapping({'h3_local_output': False, 'h3_cuda_local_output': True})


@pytest.mark.parametrize('backend', ['cuda', 'npu'])
def test_shared_preset_preserves_precision_and_explicit_overrides(backend):
    args = SimpleNamespace(h3_features='full-parallel', model='MiniMax-H3', task_type='ref2va',
        dtype='bf16', sp=8, h3_cuda_local_input=False, h3_cuda_precision='none', matmul_quant='int8')
    features.resolve_defaults(args)
    features.validate(args, backend)
    assert not args.h3_cuda_local_input
    assert args.h3_cuda_precompute_conditioning and args.h3_cuda_vision_sp
    assert args.h3_cuda_input_refiner_attention == 'legacy'
    assert args.h3_cuda_input_refiner_ffn == 'bf16'
    assert args.h3_cuda_precision == 'none' and args.matmul_quant == 'int8'


@pytest.mark.parametrize('field,value', [('input_refiner_attention', 'cudnn'), ('input_refiner_ffn', 'fp8')])
def test_npu_rejects_cuda_kernels_even_when_scheduling_is_portable(field, value):
    args = SimpleNamespace(h3_features='full-parallel', model='MiniMax-H3', task_type='ref2va', dtype='bf16', sp=8)
    features.resolve_defaults(args)
    setattr(args, 'h3_cuda_'+field, value)
    with pytest.raises(ValueError, match='CUDA-only'):
        features.validate(args, 'npu')


def test_unspecified_old_request_does_not_enable_portable_features():
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va')
    features.resolve_defaults(args)
    assert not features.enabled(args)


@pytest.mark.parametrize('mode,tp,sp', [('tp_cp', 2, 4), ('tp_cp_shared', 8, 8), ('tp_sp', 2, 8)])
def test_npu_parallel_layout_uses_native_bf16_and_keeps_int8_operator_separate(mode, tp, sp):
    from x_diffusers.adaptor.h3_parallel_config import H3ParallelConfig
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', h3_parallel_mode=mode,
                           h3_dit_tp=tp, sp=sp, matmul_quant='none')
    config = H3ParallelConfig.from_args(args, 'npu', 8)
    assert (config.tp, config.cp, config.cp_comm) == (tp, sp, 'bf16')
    args.matmul_quant = 'int8'
    with pytest.raises(ValueError, match='operator adapter'):
        H3ParallelConfig.from_args(args, 'npu', 8)


def test_native_npu_shared_options_resolve_without_cuda_precision():
    from x_diffusers.adaptor.h3_cuda_precision import resolve
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', sp=8,
                           h3_features='full-parallel', h3_vae_backend='lightning')
    assert resolve(args, 'npu') == 'none'
    assert args.h3_cuda_input_refiner == 'custom'
    assert args.h3_cuda_input_refiner_sp
    assert not args.h3_cuda_phaa_fp8


@pytest.mark.parametrize('name,task,chunk', [('base','t2va',0),('ref2va','ref2va',2048)])
def test_published_npu_json_passes_real_option_resolution(name,task,chunk):
    import json
    from x_diffusers.adaptor.h3_cuda_precision import resolve
    path=ROOT.parents[2]/'example/diffusers/configs'/('h3_npu_'+name+'.example.json')
    args=SimpleNamespace(**json.loads(path.read_text(encoding='utf-8')))
    assert resolve(args,'npu')=='none'
    assert args.task_type==task and args.h3_cuda_refiner_ff_chunk_size==chunk
    assert args.h3_vae_backend=='native-stream' and args.matmul_quant=='int8'


def test_existing_native_service_cannot_enable_local_input_without_rebuilding():
    from x_diffusers.adaptor.h3_cuda_precision import validate_manager
    manager = SimpleNamespace(pipe=None)
    base = dict(model='MiniMax-H3', task_type='ref2va', sp=4,
                h3_cuda_precision='none', h3_vae_backend='legacy')
    validate_manager(manager, SimpleNamespace(**base), 'npu')
    manager.pipe = SimpleNamespace()
    with pytest.raises(ValueError, match='Restart the service'):
        validate_manager(manager, SimpleNamespace(**base, h3_cuda_local_input=True), 'npu')


def test_refiner_coordinator_rejects_competing_standalone_offload():
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', sp=8,
        h3_refiner=True, h3_features='full-parallel')
    features.resolve_defaults(args)
    features.validate(args, 'npu')
    args.h3_cuda_reference_offload = True
    with pytest.raises(ValueError, match='coordinator owns residency'):
        features.validate(args, 'npu')


@pytest.mark.parametrize('field,value', [('h3_cuda_input_refiner_attention', 'cudnn'),
                                       ('h3_cuda_input_refiner_ffn', 'fp8')])
def test_npu_kernel_guard_cannot_be_bypassed_by_disabling_refiner_sp(field, value):
    from x_diffusers.adaptor.h3_cuda_precision import resolve
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', sp=4,
        h3_cuda_precision='none', h3_cuda_input_refiner_sp=False, **{field: value})
    with pytest.raises(ValueError, match='CUDA-only'):
        resolve(args, 'npu')


@pytest.mark.parametrize('residency,expected', [(None, False), ('inherit', False), ('resident', True)])
def test_explicit_resident_is_not_lost_in_legacy_default_resolution(residency, expected):
    from x_diffusers.adaptor.h3_cuda_precision import resolve
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', sp=4,
        h3_cuda_precision='none', h3_cuda_dit_residency=residency)
    resolve(args, 'npu')
    assert features.enabled(args) == expected
    resolve(args, 'npu')
    assert features.enabled(args) == expected


@pytest.mark.parametrize('field,value', [('sp', 8), ('text_encoder_tp', 8), ('phaa_num', 4),
    ('matmul_quant', 'none'), ('atten', 'none'), ('transformer_path', 'other-weights'),
    ('adaln_cache_file', 'other-cache')])
def test_native_resident_layout_and_operator_settings_cannot_silently_rebuild(field, value):
    from x_diffusers.adaptor.h3_cuda_precision import validate_manager
    manager = SimpleNamespace(pipe=None)
    initial = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', sp=4, text_encoder_tp=4,
        dtype='bf16', h3_features='full-parallel', h3_cuda_precision='none', h3_vae_backend='legacy',
        atten='laser', matmul_quant='int8', vae_matmul_quant='int8', phaa_num=2,
        transformer_path='weights', adaln_cache_file='cache')
    validate_manager(manager, initial, 'npu')
    manager.pipe = SimpleNamespace()
    requested = deepcopy(initial)
    requested.prompt, requested.frames = 'a new request', 243
    validate_manager(manager, requested, 'npu')
    setattr(requested, field, value)
    with pytest.raises(ValueError, match='Restart the service'):
        validate_manager(manager, requested, 'npu')


@pytest.mark.parametrize('coordinator_owned', [False, True])
def test_native_decoder_is_transferred_before_encoder_root_hook(coordinator_owned):
    from x_diffusers.adaptor.h3_features import configure_components
    pipe = SimpleNamespace(vae=Mock(), transformer=SimpleNamespace())
    observed = []
    def install(*a, **kw):
        observed.append((pipe.vae.decoder.to.called, pipe.vae.post_quant_conv.to.called, kw['stage_weights']))
    with patch('x_diffusers.adaptor.h3_condition_parallel.install_condition_parallel', side_effect=install):
        configure_components(pipe, SimpleNamespace(h3_refiner=coordinator_owned,
            h3_cuda_vae_encoder_parallel=True), 'cpu')
    assert observed == [(not coordinator_owned, not coordinator_owned, not coordinator_owned)]
