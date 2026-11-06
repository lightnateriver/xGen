"""CPU checks for the CUDA default at the existing HTTP boundary."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import video_server


def test_npu_service_defaults_are_unchanged():
    source=Path(video_server.__file__).read_text(encoding='utf-8')
    tree=ast.parse(source)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_MINIMAX_H3_CONFIG' for t in n.targets))
    override=next(n for n in tree.body if isinstance(n,ast.If) and any(isinstance(x,ast.Attribute) and x.attr=='update' and isinstance(x.value,ast.Name) and x.value.id=='DEFAULT_MINIMAX_H3_CONFIG' for x in ast.walk(n)))
    original=ast.literal_eval(assignment.value)
    for backend in ('npu','cuda'):
        namespace={'device_type':lambda:backend}
        exec(compile(ast.Module(body=[assignment,override],type_ignores=[]),'<service-defaults>','exec'),namespace)
        result=namespace['DEFAULT_MINIMAX_H3_CONFIG']
        if backend=='npu':assert result==original
        else:assert result['phaa_num']==2 and result['matmul_quant']=='none' and result['h3_cuda_precision']=='auto'


@pytest.mark.parametrize('profile,preset',[('none','fast'),('fp8-bf16','veryfast'),('fp8-sage','veryfast')])
def test_service_encoding_uses_selected_profile_only(profile,preset):
    service=object.__new__(video_server.VideoGenerationService)
    process=Mock(returncode=0)
    args=SimpleNamespace(h3_cuda_precision=profile,save_fps=24)
    with patch.object(video_server.subprocess,'Popen',return_value=process) as launch:
        service._save_mmh3_video_ffmpeg([np.zeros((2,2,3),dtype=np.uint8)],None,'unused.mp4',args)
    command=launch.call_args.args[0]
    assert command[command.index('-preset')+1]==preset
    assert command[command.index('-crf')+1]=='10'


@pytest.mark.parametrize('profile',('fp8-bf16','fp8-sage'))
def test_http_cached_worker_rejects_precision_change_before_loading(profile):
    manager=video_server.VideoInferenceManager()
    manager.pipe=object()
    manager._h3_cuda_precision_key=profile
    args=SimpleNamespace(h3_cuda_precision='none')
    with patch.object(video_server,'device_type',return_value='cuda'):
        with pytest.raises(ValueError,match='Restart the service'):
            manager.initialize_pipe(args)


@pytest.mark.parametrize('config_name,profile',[
    ('h3_cuda_base.example.json','fp8-bf16'),('h3_cuda_sage.example.json','fp8-sage'),
])
def test_http_uses_startup_precision_and_ignores_request_override(config_name,profile):
    config=Path(video_server.__file__).resolve().parents[2]/'example/diffusers/configs'/config_name
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    args=service._build_args(dict(prompt='route contract',task_type='t2va',duration=5,
                                 h3_cuda_precision='none'),save_path='unused.mp4')
    assert args.h3_cuda_precision==profile and args.sp==args.text_encoder_tp==8
