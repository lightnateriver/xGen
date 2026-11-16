"""Audio precision regressions without accelerator hardware or model downloads."""
import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


ADAPTOR = Path(__file__).resolve().parents[1] / 'x_diffusers/adaptor'
spec = importlib.util.spec_from_file_location('h3_audio_precision', ADAPTOR / 'h3_audio_precision.py')
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


class Audio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(4, 4, 3, padding=1))
        self.register_buffer('scale', torch.tensor(0.75))

    def encode(self, sample, return_dict=True):
        assert sample.dtype == torch.float32
        assert not torch.is_autocast_enabled('cpu')
        result = self.conv(sample) * self.scale
        return SimpleNamespace(sample=result) if return_dict else (result,)

    def decode(self, latents, return_dict=True):
        return self.encode(latents, return_dict=return_dict)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('method,keyword', [('encode', 'sample'), ('decode', 'latents')])
def test_audio_runs_fp32_inside_autocast_and_preserves_interface(dtype, method, keyword):
    audio = Audio()
    weights = {name: value.clone() for name, value in audio.state_dict().items()}
    signature = inspect.signature(getattr(audio, method))
    precision.prepare_audio_vae(audio)
    assert not audio.training
    assert all(value.dtype == torch.float32 for value in audio.state_dict().values())
    assert all(torch.equal(value, weights[name]) for name, value in audio.state_dict().items())
    assert inspect.signature(getattr(audio, method)) == signature

    sample = torch.randn(2, 4, 16).to(dtype)
    expected = getattr(audio, method)(sample.float(), return_dict=False)[0]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = getattr(audio, method)(**{keyword: sample}, return_dict=False)[0]
        assert output.dtype == torch.float32
        assert torch.equal(output, expected)
        assert torch.is_autocast_enabled('cpu')
        assert getattr(audio, method)(sample).sample.dtype == torch.float32


def test_audio_restores_autocast_on_decode_failure():
    audio = Audio()
    def fail(latents, return_dict=True):
        assert latents.dtype == torch.float32
        assert not torch.is_autocast_enabled('cpu')
        raise RuntimeError('injected decoder failure')
    audio.decode = fail
    precision.prepare_audio_vae(audio)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match='injected decoder failure'):
            audio.decode(torch.zeros(2, 4, 8, dtype=torch.bfloat16))
        assert torch.is_autocast_enabled('cpu')


@pytest.mark.parametrize('parallel_encode', [False, True])
def test_native_audio_vae_matches_fp32_reference_under_autocast(parallel_encode):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

    audio = AutoencoderKLMiniMaxH3Audio(encoder_dim=2, latent_dim=32, latent_channels=8,
                                      decoder_dim=128, num_attention_heads=8).eval()
    sample = torch.randn(2, 1, 1600)
    latent = torch.randn(2, 8, 2)
    with torch.no_grad():
        encoded = audio.encode(sample).latent_dist.mode()
        decoded = audio.decode(latent, return_dict=False)[0]
        precision.prepare_audio_vae(audio)
        if parallel_encode:
            parallel_spec = importlib.util.spec_from_file_location('audio_parallel_test', ADAPTOR / 'h3_condition_parallel.py')
            parallel = importlib.util.module_from_spec(parallel_spec)
            parallel.__package__ = 'audio_parallel_test'
            parallel_spec.loader.exec_module(parallel)
            # One temporal partition exercises the actual native encoder/projection
            # path without requiring a process group or changing the waveform.
            with patch.dict('sys.modules', {
                'audio_parallel_test.h3_audio_receptive': SimpleNamespace(
                    audio_halos=lambda audio: {'encoder_used': 0, 'decoder_used': 0}),
                'audio_parallel_test.h3_cuda_precision': SimpleNamespace(offload_audio=lambda audio: None),
            }):
                parallel.install_audio_parallel(audio, lambda: None, 'cpu', True, 'none', stage_weights=False)
            parallel.dist = SimpleNamespace(get_world_size=lambda group: 1)
            parallel.partition = lambda total, halo, group: (0, total, 0, total)
            parallel.gather_time = lambda local, total, group: local
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual_encoded = audio.encode(sample).latent_dist.mode()
            actual_decoded = audio.decode(latent, return_dict=False)[0]
    assert actual_encoded.dtype == actual_decoded.dtype == torch.float32
    assert torch.equal(actual_encoded, encoded)
    assert torch.equal(actual_decoded, decoded)
    assert torch.isfinite(actual_decoded).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('custom,packed', [(False, False), (True, False), (False, True)])
def test_all_component_loaders_request_fp32_audio(dtype, custom, packed):
    path = ADAPTOR / 'load_pipe.py'
    function = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'load_minimax_h3_pipe')
    dtypes = next(n for n in function.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'component_dtypes')
    loading = next(n for n in function.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'packed_checkpoint')
    code = compile(ast.Module(body=[dtypes, loading], type_ignores=[]), str(path), 'exec')
    pipe = Mock()
    scope = dict(__package__='audio_loader_test', torch=torch, dtype=dtype, pipe=pipe,
                 args=SimpleNamespace(pretrained_model_name_or_path='weights/h3'),
                 task_type='fl2va', transformer_path='weights/custom' if custom else None,
                 packed_checkpoint='weights/packed' if packed else None, component_options={},
                 logger=Mock(), MiniMaxH3Transformer3DModel=Mock())
    with patch.dict('sys.modules', {'audio_loader_test.h3_nvfp4': SimpleNamespace(load_packed_h3=Mock())}):
        exec(code, scope)
    assert pipe.load_components.call_args.kwargs['dtype'] == {'default': dtype, 'audio_vae': torch.float32}
