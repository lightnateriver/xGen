"""Startup-only overlap settings at the real HTTP argument boundary."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
import video_server


@pytest.mark.parametrize('enabled', [False,True])
def test_request_cannot_override_condition_overlap_startup(enabled,tmp_path):
    config=Path(video_server.__file__).resolve().parents[2]/'example/diffusers/configs/h3_cuda5090_ref2va.example.json'
    reference=tmp_path/'reference.png'
    Image.new('RGB',(512,512),(80,90,100)).save(reference)
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    service.default_config.update(h3_condition_overlap=enabled,h3_condition_overlap_timeout=55.)
    args=service._build_args(dict(prompt='startup contract',task_type='ref2va',duration=5,
        ref_path=[str(reference)],h3_condition_overlap=not enabled,h3_condition_overlap_timeout=1.),save_path='unused.mp4')
    assert args.h3_condition_overlap is enabled and args.h3_condition_overlap_timeout==55.


@pytest.mark.parametrize('backend',['cuda','npu'])
def test_cached_http_pipeline_requires_restart_to_enable(backend):
    manager=video_server.VideoInferenceManager()
    manager.pipe=object()
    manager.init_args=SimpleNamespace(h3_condition_overlap=False)
    args=SimpleNamespace(model='MiniMax-H3',task_type='ref2va',h3_condition_overlap=True,
                         h3_condition_overlap_timeout=600.)
    with patch.object(video_server,'device_type',return_value=backend):
        with pytest.raises(ValueError,match='Restart'):
            manager.initialize_pipe(args)


def test_omitted_cli_preserves_startup_json_and_explicit_false_overrides_it():
    import argparse
    from x_diffusers.adaptor.h3_features import add_aliases
    parser=argparse.ArgumentParser()
    add_aliases(parser)
    service=SimpleNamespace(default_config=dict(h3_condition_overlap=True,h3_condition_overlap_timeout=900.))
    video_server._apply_mmh3_config(service,parser.parse_args([]),0)
    assert service.default_config==dict(h3_condition_overlap=True,h3_condition_overlap_timeout=900.)
    video_server._apply_mmh3_config(service,parser.parse_args(['--no-h3_condition_overlap','--h3_condition_overlap_timeout','120']),0)
    assert service.default_config==dict(h3_condition_overlap=False,h3_condition_overlap_timeout=120.)
