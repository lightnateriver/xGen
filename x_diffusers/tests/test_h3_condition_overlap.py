"""CPU contracts: defaults, device-neutral scheduling, bytes, isolation and errors."""
import argparse
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ADAPTOR = Path(__file__).resolve().parents[1] / 'x_diffusers/adaptor'
def load(name):
    spec = importlib.util.spec_from_file_location(name, ADAPTOR/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

overlap = load('h3_condition_overlap')
features = load('h3_features')

def args(**changes):
    values = dict(model='MiniMax-H3', task_type='ref2va', sp=1,
                  h3_condition_overlap=True, h3_condition_overlap_timeout=2.)
    values.update(changes)
    return SimpleNamespace(**values)


def test_default_off_and_explicit_rollback():
    parser = argparse.ArgumentParser()
    features.add_aliases(parser)
    assert overlap.option_key(parser.parse_args([])) == (False,600.)
    assert parser.parse_args(['--h3_condition_overlap']).h3_condition_overlap is True
    assert parser.parse_args(['--h3_condition_overlap', '--no-h3_condition_overlap']).h3_condition_overlap is False
    with patch.object(overlap, 'ConditionOverlap', side_effect=AssertionError('must not initialize')):
        with overlap.request_context(object(), parser.parse_args([])):
            pass
    settings = args(h3_condition_overlap=False, h3_features='full-parallel')
    features.resolve_defaults(settings)
    assert not settings.h3_condition_overlap


@pytest.mark.parametrize('backend', ['cuda', 'npu'])
def test_same_validation_for_both_backends(backend):
    overlap.validate(args(), backend)


@pytest.mark.parametrize('changes,backend', [({'model':'Wan2.2'},'cuda'), ({'task_type':'t2va'},'npu'),
    ({'model':'MiniMax-H3-Image'},'npu'), ({'h3_refiner':True},'cuda'), ({'h3_parallel_mode':'tp_cp'},'npu'),
    ({'h3_condition_overlap_timeout':0},'cuda'), ({'h3_condition_overlap_timeout':float('nan')},'npu'), ({},'cpu')])
def test_invalid_profiles_rejected(changes, backend):
    with pytest.raises(ValueError):
        overlap.validate(args(**changes), backend)


def test_disabled_does_not_validate_unrelated_models():
    overlap.validate(args(model='Wan2.2', h3_condition_overlap=False), 'cpu')


def test_debug_barriers_rejected_only_when_enabled(monkeypatch):
    monkeypatch.setenv('DEBUG_TIME', '1')
    with pytest.raises(ValueError, match='DEBUG_TIME'):
        overlap.validate(args(), 'cuda')
    overlap.validate(args(h3_condition_overlap=False), 'npu')


def test_live_option_changes_require_restart():
    manager = SimpleNamespace(pipe=None, init_args=None)
    overlap.validate_manager(manager, args(), 'npu')
    manager.pipe = object()
    with pytest.raises(ValueError, match='Restart'):
        overlap.validate_manager(manager, args(h3_condition_overlap=False), 'npu')
    with pytest.raises(ValueError, match='Restart'):
        overlap.validate_manager(manager, args(h3_condition_overlap_timeout=3), 'npu')
    old = SimpleNamespace(pipe=object(), init_args=args(h3_condition_overlap=False))
    with pytest.raises(ValueError, match='Restart'):
        overlap.validate_manager(old, args(), 'cuda')


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_cpu_transport_preserves_bytes_and_does_not_cast(dtype):
    owner = SimpleNamespace(vae_group=object(), metrics=dict(vae_cpu_gathers=0, vae_payload_bytes=0))
    transport = overlap._CPUTransport(owner)
    payload = torch.arange(17).to(dtype)
    received = [torch.empty_like(payload), torch.empty_like(payload)]
    def gather(buffers, host, group):
        assert group is owner.vae_group and host.device.type=='cpu' and host.dtype==dtype
        buffers[0].copy_(host);buffers[1].copy_(host+1)
    with patch('torch.distributed.all_gather', side_effect=gather):
        transport.gather(received, payload)
    assert torch.equal(received[0].view(torch.uint8), payload.view(torch.uint8))
    assert torch.equal(received[1].view(torch.uint8), (payload+1).view(torch.uint8))
    assert owner.metrics==dict(vae_cpu_gathers=1, vae_payload_bytes=17*payload.element_size())


class State:
    def __init__(self, values, kwargs_mapping=None):
        self.values = values
        self.kwargs_mapping = kwargs_mapping or {}
    def get(self, name):
        return self.values.get(name)


class Text:
    def __call__(self, pipe, state):
        assert pipe.reference_started.wait(1), 'Reference was not overlapped'
        if pipe.fail_text:
            raise ValueError('primary text failure')
        assert 'condition_latents' not in state.values
        if pipe.audio:
            assert not pipe.audio_called
        state.values['prompt_embeds'] = 'text output'
        return pipe, state


class Reference:
    def __call__(self, pipe, state):
        pipe.reference_started.set()
        if pipe.audio:
            pipe.audio_vae.encode()
        if pipe.fail_reference:
            raise ValueError('reference failure')
        state.values['condition_latents'] = [torch.ones(2)]
        state.values['audio_condition_latents'] = []
        return pipe, state
    def get_block_state(self, state):
        return SimpleNamespace()
    def set_block_state(self, state, block):
        state.values.update(condition_latents=block.condition_latents,
                            audio_condition_latents=block.audio_condition_latents)


class Stream:
    def wait_stream(self, stream):
        pass
    def synchronize(self):
        pass


class Pipe:
    def __init__(self, audio=False, fail_text=False, fail_reference=False):
        self._blocks = SimpleNamespace(sub_blocks={'nested': SimpleNamespace(sub_blocks={'text':Text(), 'reference':Reference()})})
        self.vae = SimpleNamespace()
        self.audio_called = False
        def encode():
            self.audio_called = True
        self.audio_vae = SimpleNamespace(encode=encode)
        self.reference_started = threading.Event()
        self.audio = audio
        self.fail_text = fail_text
        self.fail_reference = fail_reference


def owner(monkeypatch, backend='cuda', **kwargs):
    encoders = ModuleType('diffusers.modular_pipelines.minimax_h3.encoders')
    encoders.MiniMaxH3Ref2VATextEncoderStep = Text
    encoders.MiniMaxH3Ref2VAReferenceEncoderStep = Reference
    monkeypatch.setitem(sys.modules, encoders.__name__, encoders)
    device = ModuleType('x_base.device');device.device_type=lambda:backend
    monkeypatch.setitem(sys.modules, 'x_base.device', device)
    accelerator = SimpleNamespace(current_device=lambda:0, Stream=lambda **kwargs:Stream(),
        device=lambda device:nullcontext(), stream=lambda stream:nullcontext(), current_stream=lambda device:Stream())
    pipe = Pipe(**kwargs)
    with patch.object(torch, 'get_device_module', return_value=accelerator), \
         patch.object(torch, 'device', return_value=SimpleNamespace(type=backend)), \
         patch('torch.distributed.is_initialized', return_value=False):
        controller = overlap.ConditionOverlap(pipe, args())
    return pipe, controller


@pytest.mark.parametrize('backend', ['cuda','npu'])
def test_isolated_native_blocks_join_and_repeated_requests(monkeypatch, backend):
    original = Text.__call__
    pipe, controller = owner(monkeypatch, backend)
    blocks=pipe._blocks.sub_blocks['nested'].sub_blocks
    original_blocks=dict(blocks)
    original_audio=pipe.audio_vae.encode
    for _ in range(2):
        state=State(dict(normalized_references=[SimpleNamespace(has_audio=False)]))
        with controller.request():
            blocks['text'](pipe,state)
            assert 'condition_latents' not in state.values
            blocks['reference'](pipe,state)
            assert state.get('prompt_embeds')=='text output'
            assert torch.equal(state.get('condition_latents')[0],torch.ones(2))
        assert controller.future is controller.pool is None
        assert not hasattr(pipe.vae,'_h3_overlap_transport')
        assert blocks==original_blocks and pipe.audio_vae.encode is original_audio
        assert pipe._h3_condition_overlap_metrics['submitted']==pipe._h3_condition_overlap_metrics['joined']==1
    assert Text.__call__ is original, 'Never patch global Diffusers classes'


def test_audio_waits_until_text_completion(monkeypatch):
    pipe,controller=owner(monkeypatch,audio=True)
    blocks=pipe._blocks.sub_blocks['nested'].sub_blocks
    with controller.request():
        state=State(dict(normalized_references=[SimpleNamespace(has_audio=True)]))
        blocks['text'](pipe,state);blocks['reference'](pipe,state)
    assert pipe.audio_called


@pytest.mark.parametrize('where',['text','reference'])
def test_error_drains_thread_preserves_primary_and_prevents_reuse(monkeypatch,where):
    pipe,controller=owner(monkeypatch,fail_text=where=='text',fail_reference=where=='reference')
    blocks=pipe._blocks.sub_blocks['nested'].sub_blocks
    with pytest.raises(ValueError,match='primary text failure' if where=='text' else 'reference failure'):
        with controller.request():
            state=State(dict(normalized_references=[SimpleNamespace(has_audio=False)]))
            blocks['text'](pipe,state);blocks['reference'](pipe,state)
    assert controller.future is controller.pool is None and not controller.active
    assert not hasattr(pipe.vae,'_h3_overlap_transport')
    with pytest.raises(RuntimeError,match='restart'):
        with controller.request():
            pass


def test_concurrent_request_rejected_without_disturbing_owner(monkeypatch):
    pipe,controller=owner(monkeypatch)
    with controller.request():
        with pytest.raises(RuntimeError,match='concurrent'):
            with controller.request():
                pass
        assert controller.active


def test_no_reference_preserves_serial_delegation(monkeypatch):
    pipe,controller=owner(monkeypatch)
    callback=Mock(return_value=('pipe','state'))
    with controller.request():
        assert controller.text(callback,pipe,State({}))==('pipe','state')
        assert controller.future is None and controller.metrics['submitted']==0
    callback.assert_called_once()


def test_timed_out_join_does_not_wait_a_second_timeout(monkeypatch):
    from concurrent.futures import Future, TimeoutError
    pipe,controller=owner(monkeypatch)
    pending=Future()
    pending.result=Mock(side_effect=AssertionError('must not wait twice'))
    with pytest.raises(TimeoutError):
        with controller.request():
            controller.future=pending
            raise TimeoutError('injected join timeout')
    pending.result.assert_not_called()
    assert controller.failed and controller.future is controller.pool is None
