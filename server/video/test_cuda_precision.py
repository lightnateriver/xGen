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


def test_http_parallel_layout_is_startup_only():
    config=Path(video_server.__file__).resolve().parents[2]/'example/diffusers/configs/h3_cuda_sage.example.json'
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    service.default_config.update(h3_parallel_mode='tp_cp',h3_dit_tp=2,h3_tp_comm='bf16',h3_cp_comm='fp8',sp=4)
    args=service._build_args(dict(prompt='layout contract',task_type='t2va',duration=5,
                                 h3_parallel_mode='sp',h3_dit_tp=8,h3_tp_comm='fp8',h3_cp_comm='bf16'),save_path='unused.mp4')
    assert (args.h3_parallel_mode,args.h3_dit_tp,args.h3_tp_comm,args.h3_cp_comm,args.sp)==('tp_cp',2,'bf16','fp8',4)


def test_http_preset_false_survives_serialization_and_request_cannot_enable_it():
    import os
    from x_diffusers.adaptor.h3_cuda_options import resolve_options
    config=Path(video_server.__file__).resolve().parents[2]/'example/diffusers/configs/h3_cuda_sage.example.json'
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    service.default_config.update(task_type='ref2va',h3_cuda_preset='sage-fp8-full-parallel',
        h3_cuda_local_input=False,h3_cuda_precompute_conditioning=False)
    before=dict(os.environ)
    args=service._build_args(dict(prompt='startup override',task_type='ref2va',duration=5,
        h3_cuda_local_input=True,h3_cuda_precompute_conditioning=True),save_path='unused.mp4')
    assert dict(os.environ)==before, 'Request parsing changed live communication environment'
    resolve_options(args,'cuda')
    assert not args.h3_cuda_local_input and not args.h3_cuda_precompute_conditioning
    assert args.h3_cuda_local_output and args.h3_cuda_dit_residency=='resident'


def test_public_cuda5090_ref2va_config_builds_independent_requests(tmp_path):
    from x_diffusers.adaptor.h3_cuda_options import resolve_options
    from PIL import Image
    refs=[str(tmp_path/f'ref{i}.png') for i in range(3)]
    for path in refs:Image.new('RGB',(512,512),(60,100,140)).save(path)
    config=Path(video_server.__file__).resolve().parents[2]/'example/diffusers/configs/h3_cuda5090_ref2va.example.json'
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    first=service._build_args(dict(prompt='First independent request',task_type='ref2va',duration=5,
        ref_path=refs[:1]),save_path='first.mp4')
    second=service._build_args(dict(prompt='Second independent request',task_type='ref2va',duration=15,
        ref_path=refs[1:],h3_cuda_preset='none'),save_path='second.mp4')
    resolve_options(first,'cuda');resolve_options(second,'cuda')
    assert first.h3_cuda_preset==second.h3_cuda_preset=='sage-fp8-full-parallel'
    assert first.h3_cuda_dit_residency==second.h3_cuda_dit_residency=='resident'
    assert first.prompt!=second.prompt and first.frames!=second.frames
    assert first.reference_paths==refs[:1]
    assert second.reference_paths==refs[1:]
    assert first.num_inference_steps==second.num_inference_steps==20


def test_cuda5090_real_http_routes_queue_valid_requests_and_reject_boundaries(tmp_path):
    """Real Flask routes/queue/config, without pretending to run GPU inference."""
    from PIL import Image
    config=Path(video_server.__file__).resolve().parents[2]/'example/diffusers/configs/h3_cuda5090_ref2va.example.json'
    reference=tmp_path/'reference.png'
    Image.new('RGB',(512,512),(80,110,130)).save(reference)
    with patch.object(video_server.VideoGenerationService,'_setup_signal_handlers'):
        service=video_server.VideoGenerationService(str(config))
    service.is_initialized=True
    client=service.app.test_client()
    assert client.get('/health').status_code==200
    for duration in (5,15):
        response=client.post('/generate',json=dict(task_type='ref2va',prompt=f'Request {duration}',
            duration=duration,resolution='768p',num_inference_steps=1,ref_path=[str(reference)],
            h3_cuda_preset='none'))
        assert response.status_code==200,response.get_json()
        task_id=response.get_json()['task_id']
        assert client.get('/status/'+task_id).status_code==200
        queued=service.task_queue.get_nowait()
        args=service._build_args(queued['data'],save_path=queued['save_path'])
        assert args.h3_cuda_preset=='sage-fp8-full-parallel'
        assert args.num_inference_steps==1
    count=len(service.tasks)
    for invalid in (dict(duration=4),dict(duration=16),dict(duration=5.5),dict(task_type='t2va')):
        body=dict(task_type='ref2va',prompt='Invalid boundary',duration=5,resolution='768p',
            num_inference_steps=1,ref_path=[str(reference)])
        response=client.post('/generate',json=body|invalid)
        assert response.status_code==400,response.get_json()
        assert len(service.tasks)==count and service.task_queue.empty()


@pytest.mark.parametrize('enabled,failed',[(False,False),(True,False),(True,True)])
def test_http_inference_manages_condition_scope_and_next_request_recomputes(enabled,failed):
    """Exercise the real HTTP inference method with a small CPU condition model."""
    from contextlib import nullcontext
    import torch
    from x_diffusers.framework.transformer.h3_conditioning import (
        _CURRENT,project_conditioning,release_conditioning)
    model=SimpleNamespace(context_embedder=torch.nn.Linear(4,8),token_refiner=torch.nn.Linear(8,8))
    class Pipeline:
        transformer=model
        source=torch.randn(1,3,4)
        fail=failed
        def __call__(self,**kwargs):
            for _ in range(3):
                if enabled:
                    output=project_conditioning(model,self.source)
                else:
                    assert _CURRENT.get() is None
                    output=model.token_refiner(model.context_embedder(self.source))
            if self.fail:raise RuntimeError('injected pipeline failure')
            release_conditioning(model)
            return dict(videos=output)
    manager=video_server.VideoInferenceManager();manager.pipe=Pipeline()
    manager._mmh3_setup_super_resolution=Mock(return_value=(False,32,32))
    manager._mmh3_build_infer_params=Mock(return_value={'num_frames':5})
    manager._mmh3_load_references=Mock(return_value=[])
    manager._mmh3_setup_attention=Mock()
    manager._mmh3_patch_denoiser=Mock(return_value=(False,None))
    manager._mmh3_switch_adaln_cache=Mock()
    manager._mmh3_cleanup=Mock()
    args=SimpleNamespace(h3_refiner=False,h3_cuda_precompute_conditioning=enabled,
        task_type='ref2va',num_inference_steps=3,frames=5,width=32,height=32,enable_audio=True)
    with patch.object(video_server,'base_temporal_padding',return_value=nullcontext()),torch.no_grad():
        if failed:
            with pytest.raises(RuntimeError,match='injected pipeline failure'):manager._infer_minimax_h3(args)
        else:
            first=manager._infer_minimax_h3(args)
            assert torch.equal(first['videos'],model.token_refiner(model.context_embedder(manager.pipe.source)))
        assert _CURRENT.get() is None
        if enabled:
            assert model._h3_conditioning_metrics['computations']==1
            assert model._h3_conditioning_metrics['reuses']==2
            assert model._h3_conditioning_metrics['cleared']
        manager.pipe.fail=False;manager.pipe.source=torch.randn(1,3,4)
        second=manager._infer_minimax_h3(args)
        assert torch.equal(second['videos'],model.token_refiner(model.context_embedder(manager.pipe.source)))
        if enabled:
            assert model._h3_conditioning_metrics['computations']==1
            assert model._h3_conditioning_metrics['reuses']==2
            assert model._h3_conditioning_metrics['released_before_vae']
    assert manager._mmh3_cleanup.call_count==2
