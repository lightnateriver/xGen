"""Default isolation, chunk ownership/error propagation and staged GPU decode."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

VAE=Path(__file__).resolve().parents[1]/'x_diffusers/framework/vae'
sys.path.insert(0,str(VAE))
from x_diffusers.framework.vae import omni_decoder
from omni_h3.chunked_video import ChunkedVideoMP4Session, quantize_chunk
from omni_h3.media_utils import ChunkedMP4Encoder


def options(**kw):
    return SimpleNamespace(**dict(dict(model='MiniMax-H3',h3_vae_backend='legacy',h3_vae_model_path=None,
                                      h3_vae_trust_remote_code=False,vae_lightning='none',sp=8),**kw))


def test_vendored_source_hashes_match_lf_canonical_files():
    import hashlib
    root=VAE/'omni_h3'
    manifest=json.loads((root/'UPSTREAM.json').read_text(encoding='utf-8'))
    for name,entry in manifest['files'].items():
        data=(root/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(data).hexdigest()==entry['adapted_sha256'],name


def test_legacy_does_not_construct_a_backend(monkeypatch):
    def fail(*a,**kw):raise AssertionError('legacy loaded new checkpoint')
    monkeypatch.setattr(omni_decoder,'OmniH3Decoder',fail)
    pipe=SimpleNamespace()
    omni_decoder.validate_options(options(),'npu')
    omni_decoder.configure(pipe,options(),torch.device('cpu'))
    assert not hasattr(pipe,'_h3_vae_decode_backend')


@pytest.mark.parametrize('device_type', ['cuda','npu'])
def test_lightning_selects_original_decoder_without_native_checkpoint(monkeypatch,device_type):
    def fail(*a,**kw):raise AssertionError('Lightning constructed native decoder')
    monkeypatch.setattr(omni_decoder,'OmniH3Decoder',fail)
    args=options(h3_vae_backend='lightning')
    omni_decoder.validate_options(args,device_type)
    pipe=SimpleNamespace()
    omni_decoder.configure(pipe,args,torch.device('cpu'))
    assert args.vae_lightning=='decoder' and pipe._h3_vae_backend_name=='lightning'
    assert not hasattr(pipe,'_h3_vae_decode_backend')


def test_native_stream_npu_options_do_not_construct_cuda_objects(monkeypatch):
    def fail(*a,**kw):raise AssertionError('validation allocated an accelerator object')
    monkeypatch.setattr(torch.cuda,'Stream',fail)
    args=options(h3_vae_backend='native-stream',h3_vae_model_path='weights/video_vae',h3_vae_trust_remote_code=True)
    omni_decoder.validate_options(args,'npu')
    assert args.vae_lightning=='none'


@pytest.mark.parametrize('device_type',['cuda','npu'])
@pytest.mark.parametrize('selected',['legacy','lightning','native-stream'])
def test_all_three_backends_validate_on_both_devices(device_type,selected):
    extra=dict(h3_vae_model_path='weights/video_vae',h3_vae_trust_remote_code=True) if selected=='native-stream' else {}
    omni_decoder.validate_options(options(h3_vae_backend=selected,**extra),device_type)


@pytest.mark.parametrize('backend',['cuda','npu'])
@pytest.mark.parametrize('profile',['none','fp8-bf16','fp8-sage'])
def test_native_backend_reuses_fp16_linear_storage_on_both_devices(monkeypatch,backend,profile):
    captured={}
    def construct(path,device,**kw):
        captured.update(kw);return SimpleNamespace(path=path)
    monkeypatch.setattr(omni_decoder,'OmniH3Decoder',construct)
    args=options(h3_vae_backend='native-stream',h3_vae_model_path='weights/video_vae',
                 h3_vae_trust_remote_code=True,h3_cuda_precision=profile)
    omni_decoder.configure(SimpleNamespace(),args,backend+':0')
    assert captured['fp16_linears'] is True


def test_fp16_storage_preserves_non_linear_parameters():
    from x_diffusers.framework.vae.h3_cuda_chunks import prepare_decoder_linears
    decoder=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.LayerNorm(8),torch.nn.Conv1d(8,8,1))
    assert prepare_decoder_linears(decoder)==1
    assert decoder[0].weight.dtype==decoder[0].bias.dtype==torch.float16
    assert decoder[1].weight.dtype==decoder[2].weight.dtype==torch.float32


@pytest.mark.parametrize('old,new', [('legacy','lightning'),('lightning','legacy')])
def test_resident_service_requires_restart_to_change_lightning(old,new):
    from x_diffusers.adaptor.h3_cuda_precision import validate_manager
    args=options(h3_vae_backend=new,h3_cuda_precision='none')
    manager=SimpleNamespace(pipe=SimpleNamespace(_h3_vae_backend_name=old))
    with pytest.raises(ValueError,match='Restart the service to change H3 VAE'):
        validate_manager(manager,args,'cuda')


@pytest.mark.parametrize('kw,backend',[
    ({'h3_vae_backend':'native-stream'},'cuda'),
    ({'h3_vae_backend':'native-stream','h3_vae_model_path':'weights/video_vae','h3_vae_trust_remote_code':True},'cpu'),
    ({'h3_vae_backend':'native-stream','h3_vae_model_path':'weights/video_vae','h3_vae_trust_remote_code':True,'vae_lightning':'decoder'},'cuda'),
    ({'h3_vae_model_path':'weights/video_vae'},'cuda'),
    ({'h3_vae_backend':'native-stream','h3_vae_model_path':'weights/video_vae','h3_vae_trust_remote_code':True,'vae_matmul_quant':'int8'},'npu'),
])
def test_reject_invalid_combinations(kw,backend):
    with pytest.raises(ValueError):omni_decoder.validate_options(options(**kw),backend)


def test_cpu_chunk_encoder_and_reader_release():
    consumed=[]
    encoder=ChunkedMP4Encoder(width=32,height=32,fps=24,video_codec_options={'preset':'fast','crf':'10'})
    chunk=np.zeros((4,32,32,3),dtype=np.uint8);chunk[...,0]=200
    encoder.push(chunk,on_consumed=lambda:consumed.append(True))
    encoded=encoder.finish()
    assert consumed==[True] and encoder.frame_count==4 and len(encoded)>100
    encoder.abort()


def test_consumer_failure_releases_queued_reader():
    consumed=[]
    encoder=ChunkedMP4Encoder(width=32,height=32,fps=24)
    with pytest.raises(ValueError):
        encoder.push(np.zeros((1,31,32,3),dtype=np.uint8),on_consumed=lambda:consumed.append(True))
    encoder.abort()
    assert consumed==[True]


def test_quantization_matches_native_rounding():
    pixels=torch.tensor([0.,.5,1.]).view(1,3,1,1,1)
    assert quantize_chunk(pixels,(0.,1.)).flatten().tolist()==[0,128,255]


def test_decoder_linear_storage_reuses_existing_fp16_policy_only_for_linears():
    from x_diffusers.framework.vae.h3_cuda_chunks import prepare_decoder_linears
    decoder=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.LayerNorm(4),torch.nn.Linear(4,4))
    assert prepare_decoder_linears(decoder)==2
    assert decoder[0].weight.dtype==decoder[2].weight.dtype==torch.float16
    assert decoder[1].weight.dtype==torch.float32


def test_configure_reuses_installed_backend(monkeypatch):
    existing=SimpleNamespace(path='weights/video_vae')
    pipe=SimpleNamespace(_h3_vae_decode_backend=existing)
    def fail(*a,**kw):raise AssertionError('reloaded resident backend')
    monkeypatch.setattr(omni_decoder,'OmniH3Decoder',fail)
    omni_decoder.configure(pipe,options(h3_vae_backend='native-stream',h3_vae_model_path=existing.path),'cuda')
    assert pipe._h3_vae_decode_backend is existing
    with pytest.raises(ValueError,match='Restart'):
        omni_decoder.configure(pipe,options(h3_vae_backend='native-stream',h3_vae_model_path='different'),'cuda')


def test_latent_output_bypasses_device_staging():
    backend=object.__new__(omni_decoder.OmniH3Decoder)
    values=SimpleNamespace(output_type='latent',latents=torch.zeros(1,24,4,16,16))
    block=SimpleNamespace(get_block_state=lambda state:values,set_block_state=lambda state,value:None)
    backend.decode_video(block,SimpleNamespace(),{})
    assert values.videos is values.latents


def test_audio_context_cleared_after_video_failure():
    backend=object.__new__(omni_decoder.OmniH3Decoder)
    audio=torch.zeros(1,2,100)
    state={'audio':audio,'sampling_rate':32000}
    def fail(pipe,state):
        assert backend.audio is audio and backend.sample_rate==32000
        raise RuntimeError('video failure')
    block=SimpleNamespace(sub_blocks={'audio':lambda pipe,state:(pipe,state),'video':fail})
    with pytest.raises(RuntimeError,match='video failure'):backend.decode_step(block,SimpleNamespace(),state)
    assert backend.audio is backend.sample_rate is None


@pytest.mark.parametrize('failure',['prepare','onload','stream','decode','device','offload'])
def test_master_restoration_attempted_after_each_staging_failure(monkeypatch,failure):
    from contextlib import nullcontext
    from x_base.utils.infer_info import infer_info
    calls=[]
    def stage(name):
        calls.append(name)
        if name==failure:raise RuntimeError(name+' injected')
        return torch.zeros(1,3,1,4,4)
    backend=object.__new__(omni_decoder.OmniH3Decoder)
    backend.device=torch.device('cpu')
    backend.group=SimpleNamespace(onload_=lambda:stage('onload'),offload_=lambda:stage('offload'),
                                  stream=SimpleNamespace(synchronize=lambda:stage('stream')))
    backend.device_module=SimpleNamespace(synchronize=lambda device:stage('device'))
    backend.vae=SimpleNamespace(decode_dtype=torch.float16,decode_latent=lambda z:stage('decode'))
    values=SimpleNamespace(output_type='np',latents=torch.zeros(1))
    block=SimpleNamespace(get_block_state=lambda s:values,set_block_state=lambda s,v:None)
    pipe=SimpleNamespace(video_processor=SimpleNamespace(postprocess_video=lambda x,**kw:x),
                         _h3_prepare_video_decode=lambda:stage('prepare'))
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    with pytest.raises(RuntimeError,match=failure+' injected'):backend.decode_video(block,pipe,{})
    assert calls.count('offload')==1
    assert calls.count('device')==int(failure in ('device','offload'))
    assert calls[0]=='prepare'
    assert not backend.metrics['success'] and not infer_info._route_b_saved


@pytest.mark.parametrize('decode_fails',[False,True])
def test_encoder_abort_cannot_skip_restoration_or_hide_original_error(monkeypatch,tmp_path,decode_fails):
    from contextlib import nullcontext
    from x_base.utils.infer_info import infer_info
    from x_diffusers.framework.vae.omni_h3 import chunked_video
    calls=[]
    def abort():
        calls.append('abort');raise RuntimeError('abort injected')
    def decode(*a,**kw):
        if decode_fails:raise RuntimeError('decode injected')
        return [b'video']
    session=SimpleNamespace(start=lambda **kw:None,abort=abort,transfer_stats=lambda:None)
    monkeypatch.setattr(chunked_video,'ChunkedVideoMP4Session',lambda **kw:session)
    monkeypatch.setattr(chunked_video,'decode_to_mp4',decode)
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    backend=object.__new__(omni_decoder.OmniH3Decoder)
    backend.device=torch.device('cpu');backend.audio=backend.sample_rate=None
    backend.vae=SimpleNamespace(decode_dtype=torch.float16,chunk_value_range=(0.,1.))
    backend.group=SimpleNamespace(onload_=lambda:None,offload_=lambda:calls.append('offload'),
                                  stream=SimpleNamespace(synchronize=lambda:None))
    backend.device_module=SimpleNamespace(synchronize=lambda device:calls.append('device'))
    infer_info.save_path=str(tmp_path/'video.mp4')
    infer_info.height=infer_info.save_height=32;infer_info.width=infer_info.save_width=32;infer_info.fps=24
    values=SimpleNamespace(output_type='pil',latents=torch.zeros(1))
    block=SimpleNamespace(get_block_state=lambda s:values,set_block_state=lambda s,v:None)
    with pytest.raises(RuntimeError,match='publication failed' if decode_fails else 'abort injected') as exc:
        backend.decode_video(block,SimpleNamespace(),{})
    if decode_fails:assert str(exc.value.__cause__)=='decode injected'
    assert calls==(['abort','offload'] if decode_fails else ['abort','device','offload'])
    assert not backend.metrics['success'] and not infer_info._route_b_saved


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA native checkpoint smoke')
def test_native_backend_reuses_cpu_master_and_restores_after_error(tmp_path):
    import os
    checkpoint=os.environ.get('XGEN_TEST_NATIVE_VAE')
    if not checkpoint:pytest.skip('set XGEN_TEST_NATIVE_VAE for model smoke')
    from x_base.utils.infer_info import infer_info
    backend=omni_decoder.OmniH3Decoder(checkpoint,'cuda:0',trust_remote_code=True)
    identity=[(id(p),p.data_ptr()) for p in backend.vae.model.parameters()]
    infer_info.height=infer_info.save_height=256
    infer_info.width=infer_info.save_width=256
    infer_info.fps=24
    values=SimpleNamespace(output_type='pil',latents=torch.zeros(1,24,4,16,16,device='cuda'))
    block=SimpleNamespace(get_block_state=lambda state:values,set_block_state=lambda state,block_state:None)
    pipe=SimpleNamespace()
    for i in range(2):
        infer_info.save_path=str(tmp_path/f'video-{i}.mp4')
        backend.decode_video(block,pipe,{})
        assert Path(infer_info.save_path).stat().st_size>100 and infer_info._route_b_saved
        assert all(p.device.type=='cpu' for p in backend.vae.model.parameters())
        assert identity==[(id(p),p.data_ptr()) for p in backend.vae.model.parameters()]
    values.latents=values.latents.expand(2,-1,-1,-1,-1)
    with pytest.raises(ValueError,match='one video'):backend.decode_video(block,pipe,{})
    assert not infer_info._route_b_saved
    assert identity==[(id(p),p.data_ptr()) for p in backend.vae.model.parameters()]
