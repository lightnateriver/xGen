"""Request reuse, changing conditions and successful/failed request cleanup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

path=Path(__file__).resolve().parents[1]/'x_diffusers/adaptor/h3_conditioning.py'
spec=importlib.util.spec_from_file_location('h3_conditioning_unit',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def model():
    return SimpleNamespace(context_embedder=torch.nn.Linear(4,8).eval(),
                           token_refiner=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.SiLU()).eval())


@torch.no_grad()
def test_twenty_forwards_compute_once_and_match_original():
    net=model();x=torch.randn(1,7,4)
    expected=net.token_refiner(net.context_embedder(x))
    with module.request_conditioning(net):
        outputs=[module.project_conditioning(net,x) for _ in range(20)]
        assert all(y is outputs[0] and torch.equal(y,expected) for y in outputs)
    assert net._h3_conditioning_metrics['computations']==1
    assert net._h3_conditioning_metrics['reuses']==19
    assert net._h3_conditioning_metrics['cleared']
    assert module._CURRENT.get() is None


@torch.no_grad()
def test_positive_negative_and_inplace_changes_do_not_reuse_stale_values():
    net=model();positive=torch.randn(1,7,4);negative=torch.randn(1,7,4)
    with module.request_conditioning(net):
        first=module.project_conditioning(net,positive)
        other=module.project_conditioning(net,negative)
        assert module.project_conditioning(net,positive) is first
        negative.add_(1)
        changed=module.project_conditioning(net,negative)
        assert changed is not other
        assert torch.equal(changed,net.token_refiner(net.context_embedder(negative)))
    assert net._h3_conditioning_metrics['computations']==3


@torch.inference_mode()
def test_inference_mode_inplace_change_is_detected_without_version_counter():
    net=model();x=torch.randn(1,7,4)
    with module.request_conditioning(net):
        a=module.project_conditioning(net,x)
        assert module.project_conditioning(net,x) is a
        x.add_(1)
        b=module.project_conditioning(net,x)
        assert b is not a and torch.equal(b,net.token_refiner(net.context_embedder(x)))


@torch.no_grad()
@pytest.mark.parametrize('failed',(False,True))
def test_request_end_and_error_release_tensors_and_next_request_recomputes(failed):
    net=model();x=torch.randn(1,7,4)
    try:
        with module.request_conditioning(net):
            y=module.project_conditioning(net,x);ref=weakref.ref(y);del y
            if failed:raise RuntimeError('injected')
    except RuntimeError:
        assert failed
    assert ref() is None and module._CURRENT.get() is None
    with module.request_conditioning(net):module.project_conditioning(net,x)
    assert net._h3_conditioning_metrics['computations']==1
    assert net._h3_conditioning_metrics['reuses']==0


@torch.no_grad()
def test_release_before_vae_keeps_only_scalar_metrics():
    net=model();x=torch.randn(1,7,4)
    with module.request_conditioning(net):
        y=module.project_conditioning(net,x);ref=weakref.ref(y);del y
        module.release_conditioning(net)
        assert ref() is None
    assert net._h3_conditioning_metrics['released_before_vae']
    assert all(isinstance(v,(int,bool)) for v in net._h3_conditioning_metrics.values())


def test_requires_managed_scope_and_no_grad():
    net=model();x=torch.randn(1,7,4)
    with pytest.raises(RuntimeError,match='managed request scope'):module.project_conditioning(net,x)
    with module.request_conditioning(net):
        with pytest.raises(ValueError,match='inference-only'):module.project_conditioning(net,x)
