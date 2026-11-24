"""CUDA-only, default-off Torch Compiler contract for the native H3 VAE."""

from types import SimpleNamespace

import pytest
import torch

from x_diffusers.adaptor import h3_vae_compile


def args(**overrides):
    values = dict(h3_vae_torch_compile=False, model="MiniMax-H3", h3_vae_backend="native-stream")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compiler_is_default_off_and_validation_is_noop():
    h3_vae_compile.validate_options(args(), "npu")


@pytest.mark.parametrize("backend", ["npu", "cpu"])
def test_compiler_rejects_non_cuda(backend):
    with pytest.raises(ValueError, match="CUDA-only"):
        h3_vae_compile.validate_options(args(h3_vae_torch_compile=True), backend)


def test_compiler_requires_native_stream():
    with pytest.raises(ValueError, match="native-stream"):
        h3_vae_compile.validate_options(args(h3_vae_torch_compile=True, h3_vae_backend="legacy"), "cuda")


def test_compiler_wraps_only_decoder_forward(monkeypatch):
    calls = {}

    def fake_compile(fn, **kwargs):
        calls.update(kwargs)
        return lambda *a, **kw: fn(*a, **kw)

    monkeypatch.setattr(torch, "compile", fake_compile)
    decoder = torch.nn.Linear(4, 4)
    original = decoder.forward
    h3_vae_compile.compile_decoder_tile_subgraph(decoder)
    assert calls == {"dynamic": True, "fullgraph": False, "mode": "max-autotune-no-cudagraphs"}
    assert decoder._xgen_torch_compile_enabled is True
    assert decoder._xgen_torch_compile_original_forward is original
    h3_vae_compile.compile_decoder_tile_subgraph(decoder)
    assert calls == {"dynamic": True, "fullgraph": False, "mode": "max-autotune-no-cudagraphs"}
