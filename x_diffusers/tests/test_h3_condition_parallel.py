"""Opt-in boundaries and uneven temporal transport without model downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'x_diffusers/adaptor'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'x_diffusers.adaptor'
    spec.loader.exec_module(module)
    return module


parallel = load('h3_condition_parallel')
precision = load('h3_cuda_precision')


@pytest.mark.parametrize('name', precision.CONDITION_PARALLEL)
def test_switches_are_cuda_ref2va_sp8_only(name):
    base = dict(model='MiniMax-H3', task_type='ref2va', sp=8, h3_cuda_precision='fp8-sage',
                h3_parallel_mode='sp')
    base[name] = 'long' if name.endswith('decoder_parallel') else True
    assert precision.resolve(SimpleNamespace(**base), 'cuda') == 'fp8-sage'
    for changes, backend in ((dict(task_type='t2va'), 'cuda'), (dict(sp=4), 'cuda'),
                             (dict(h3_cuda_precision='none'), 'cuda'), ({}, 'npu')):
        with pytest.raises(ValueError):
            precision.resolve(SimpleNamespace(**(base | changes)), backend)
    with pytest.raises(ValueError, match='Restart the service to change H3 condition parallelism'):
        precision.validate_manager(SimpleNamespace(pipe=object()), SimpleNamespace(**base), 'cuda')


@pytest.mark.parametrize('total,unit', [(7, 1), (8, 1), (17, 1), (400, 1),
                                     (405, 800), (600, 800), (603, 800)])
def test_uneven_gather_preserves_samples_and_batch(total, unit):
    expected = torch.arange(2 * 3 * total * unit).reshape(2, 3, total * unit).float()
    sizes = [(total*(i+1)//8-total*i//8)*unit for i in range(8)]
    chunks = list(expected.split(sizes, dim=-1))

    def gather(received, payload, group):
        assert payload.shape[-1] == max(sizes)
        for destination, chunk in zip(received, chunks):
            destination.zero_()
            destination[..., :chunk.shape[-1]].copy_(chunk)

    with patch.object(parallel.dist, 'get_world_size', return_value=8), \
         patch.object(parallel.dist, 'all_gather', side_effect=gather):
        for chunk in chunks:
            result = parallel.gather_time(chunk, total, object(), unit=unit)
            assert torch.equal(result, expected)


def test_numeric_policy_restores_after_error():
    before = torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError):
        with parallel.strict_audio_convolution():
            assert not torch.backends.cudnn.allow_tf32
            raise RuntimeError('injected')
    assert torch.backends.cudnn.allow_tf32 == before


def test_disabled_profile_preserves_npu_defaults():
    args = SimpleNamespace(model='MiniMax-H3', task_type='ref2va', h3_cuda_precision='auto')
    assert precision.resolve(args, 'npu') == 'none'
    assert precision.condition_parallel_key(args) == (False, False, False, 'none')


def test_manual_encoder_stage_remains_visible_to_native_pipeline_offload_detection():
    from diffusers.hooks.group_offloading import _get_group_onload_device, _is_group_offload_enabled

    vae = torch.nn.Module()
    vae.encoder = torch.nn.Linear(2, 2)
    vae.decoder = torch.nn.Linear(2, 2)
    vae.decoder.register_buffer('frequencies', torch.arange(8).float())
    original = vae.decoder.frequencies
    assert not _is_group_offload_enabled(vae)
    weights = SimpleNamespace(stream=None, offload_leader=vae.encoder, onload_leader=vae.encoder)
    parallel.register_encoder_stage(vae, weights, 'cuda:0')
    assert _is_group_offload_enabled(vae)
    assert _get_group_onload_device(vae) == torch.device('cuda:0')
    assert vae.decoder.frequencies is original and original.device.type == 'cpu'
