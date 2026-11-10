"""Shard-layout and boundary tests complement the real distributed block test."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from x_diffusers.adaptor import h3_parallel
from x_diffusers.adaptor.h3_parallel_config import H3ParallelConfig, validate_resident_args
from x_diffusers.adaptor import h3_cuda_precision
from x_diffusers.framework.transformer.h3_local_io import LocalSpan, prepare_local


def test_shared_layout_accepts_full_condition_profile_without_changing_sp_defaults():
    args=SimpleNamespace(model='MiniMax-H3',dtype='bf16',task_type='ref2va',
        h3_cuda_precision='fp8-sage',h3_parallel_mode='tp_cp_shared',h3_dit_tp=8,sp=8,
        text_encoder_tp=8,phaa_num=2,atten='none',matmul_quant='none',
        h3_tp_comm='fp8',h3_cp_comm='fp8',h3_cuda_input_refiner='cudnn-sp-fp8',
        h3_cuda_precompute_conditioning=True,h3_cuda_vision_sp=True,
        h3_cuda_vae_encoder_parallel=True,h3_cuda_audio_encoder_parallel=True,
        h3_cuda_audio_decoder_parallel='long')
    assert h3_cuda_precision.resolve(args,'cuda')=='fp8-sage'
    with patch.object(h3_cuda_precision.dist,'is_initialized',return_value=True), \
         patch.object(h3_cuda_precision.dist,'get_world_size',return_value=8):
        h3_cuda_precision.validate(args,'cuda')
    original=H3ParallelConfig.from_args(SimpleNamespace(),'npu',8)
    assert original==H3ParallelConfig()


def test_shared_layout_does_not_claim_unused_phaa_transport():
    from x_diffusers.adaptor.h3_cuda import configure_parallelism
    attention=SimpleNamespace()
    model=SimpleNamespace(_h3_parallel_config=H3ParallelConfig('tp_cp_shared',8,8,'fp8','fp8'),
                          transformer_blocks=[SimpleNamespace(attn=attention)])
    with patch.dict('os.environ',{'XGEN_H3_CUDA_VAE_TILES':'0'}):
        configure_parallelism(SimpleNamespace(transformer=model),SimpleNamespace(h3_cuda_vae_sp=False))
    assert not hasattr(attention,'_h3_phaa_fp8')


@pytest.mark.parametrize('column,parts',[(True,3),(True,2),(False,1)])
@pytest.mark.parametrize('fp8',[False,True])
def test_partition_mapping_bias_and_original_scale(column,parts,fp8):
    group=object()
    original=torch.nn.Linear(64,192,bias=True,dtype=torch.bfloat16)
    with torch.no_grad():
        original.weight.copy_(torch.linspace(-2,3,192*64).reshape(192,64))
        original.bias.copy_(torch.arange(192))
    axis=0 if column else 1
    scale=(original.weight.float().abs().amax()/448).clamp_min(torch.finfo(torch.float32).tiny)
    slices=[]
    for rank in (0,1):
        manager=SimpleNamespace(tp_group=group,tp_size=2,tp_rank=rank)
        with patch('torch.distributed.get_world_size',return_value=2):
            layer=h3_parallel.shard_projection(original,manager,column=column,parts=parts,fp8=fp8)
        expected=torch.cat([p.chunk(2,axis)[rank] for p in original.weight.chunk(parts,axis)],axis).contiguous()
        if fp8:
            assert torch.equal(layer.weight_scale,scale)
            assert torch.equal(layer.weight.view(torch.uint8),((expected.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)).view(torch.uint8))
            assert layer.in_features==expected.shape[1] and layer.out_features==expected.shape[0]
        else:
            assert torch.equal(layer.weight,expected)
        expected_bias=(torch.cat([p.chunk(2)[rank] for p in original.bias.chunk(parts)]) if column else original.bias)
        assert torch.equal(layer.bias,expected_bias)
        assert layer.tp_group is group
        slices.append(layer.weight)
    assert original.weight.shape==(192,64)  # source is never mutated


@pytest.mark.parametrize('total,peers',[(1,4),(17,4),(22,4),(19,2)])
def test_local_span_has_no_duplicate_or_missing_tokens(total,peers):
    tensor=torch.arange(total)
    recovered=[]
    for rank in range(peers):
        span=LocalSpan.for_rank(total,peers,rank)
        local=span.slice_padded(tensor,value=-1)
        assert len(local)==span.length
        assert torch.equal(local[span.valid:],torch.full_like(local[span.valid:],-1))
        recovered.append(local[:span.valid])
        source,target=span.indices(tensor.flip(0))
        assert torch.equal(tensor.flip(0).index_select(0,source)-span.start,target)
        assert bool(((target>=0)&(target<span.valid)).all())
    assert torch.equal(torch.cat(recovered),tensor)


@pytest.mark.parametrize('fails',[False,True])
def test_reference_audio_staging_always_returns_cpu(fails):
    audio=Mock()
    audio.encode=Mock(side_effect=RuntimeError('encode failed')) if fails else Mock(return_value='encoded')
    pipe=SimpleNamespace(audio_vae=audio)
    with patch.object(h3_cuda_precision,'offload_audio') as offload:
        h3_parallel.stage_reference_audio(pipe,'cuda:0')
        if fails:
            with pytest.raises(RuntimeError,match='encode failed'): audio.encode('input')
        else:
            assert audio.encode('input')=='encoded'
        audio.to.assert_called_once_with('cuda:0')
        offload.assert_called_once_with(audio)


def test_resident_service_rejects_transport_changes():
    manager=SimpleNamespace(pipe=object(),_h3_parallel_key=H3ParallelConfig('tp_cp',2,4,'bf16','bf16'))
    args=SimpleNamespace(h3_parallel_mode='tp_cp',h3_dit_tp=2,sp=4,h3_tp_comm='fp8',h3_cp_comm='bf16',
                         h3_cuda_precision='fp8-sage',model='MiniMax-H3',task_type='ref2va',h3_vae_backend='legacy')
    with patch.object(h3_cuda_precision.dist,'is_initialized',return_value=True),patch.object(h3_cuda_precision.dist,'get_world_size',return_value=8):
        with pytest.raises(ValueError,match='Restart the service'):
            h3_cuda_precision.validate_manager(manager,args,'cuda')


def test_invalid_packed_index_is_not_silently_filtered():
    model=SimpleNamespace(parallel_manager=SimpleNamespace(sp_size=1,sp_rank=0))
    with pytest.raises(RuntimeError,match='outside the sequence'):
        prepare_local(model,torch.zeros(1,1,16),torch.zeros(1,0,8),torch.zeros(1,0,64),
                      torch.zeros(1,3,dtype=torch.int64),torch.empty(0,dtype=torch.int64),
                      torch.tensor([1]),torch.empty(0,dtype=torch.int64))


@torch.no_grad()
@pytest.mark.parametrize('peers,rank',[(1,0),(8,0),(8,1),(8,7),(4,3)])
def test_local_io_reuses_only_fixed_conditions_across_steps(peers,rank):
    from x_diffusers.adaptor.h3_conditioning import request_conditioning

    model=SimpleNamespace(parallel_manager=SimpleNamespace(sp_size=peers,sp_rank=rank,sp_group=object()),
        context_embedder=torch.nn.Linear(4,8),token_refiner=torch.nn.Linear(8,8),
        proj_in=torch.nn.Linear(3,8),audio_proj_in=torch.nn.Linear(2,8),rope=lambda p:(p,p))
    calls=[]
    model.token_refiner.register_forward_hook(lambda *args:calls.append(1))
    text=torch.randn(1,2,4);video=torch.randn(1,2,3);audio=torch.randn(1,1,2)
    positions=torch.zeros(5,3,dtype=torch.int64)
    indices=(torch.tensor([0,3]),torch.tensor([1,4]),torch.tensor([2]))
    def run(step):
        return prepare_local(model,video+step,audio-step,text,positions,*indices)[0]
    with patch('x_diffusers.framework.transformer.h3_local_io.set_pad'):
        expected=[run(step) for step in range(20)]
        assert len(calls)==20
        calls.clear();model._h3_precompute_conditioning=True
        with request_conditioning(model):
            actual=[run(step) for step in range(20)]
    assert len(calls)==1
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert model._h3_conditioning_metrics['computations']==1
    assert model._h3_conditioning_metrics['reuses']==19
    assert model._h3_conditioning_metrics['cleared']


def test_resident_manager_reuses_fl2va_but_reloads_ref2va():
    from x_diffusers import InferenceManager
    manager=InferenceManager()
    values=dict(h3_parallel_mode='tp_cp',task_type='t2va',model='MiniMax-H3',
                pretrained_model_name_or_path='models/h3',fuse_lora=False)
    manager.init_args=SimpleNamespace(**values)
    assert manager.check_abilities_args_consistent(SimpleNamespace(**(values|dict(task_type='fl2va'))))
    assert not manager.check_abilities_args_consistent(SimpleNamespace(**(values|dict(task_type='ref2va'))))
    assert not manager.check_abilities_args_consistent(SimpleNamespace(**(values|dict(pretrained_model_name_or_path='models/other'))))


@pytest.mark.parametrize('change',[dict(h3_dit_tp=1),dict(h3_tp_comm='fp8'),dict(h3_cp_comm='bf16'),
                                 dict(task_type='ref2va'),dict(h3_parallel_mode='sp'),dict(h3_cuda_precision='fp8-bf16')])
def test_resident_layout_changes_fail_before_reloading(change):
    values=dict(h3_parallel_mode='tp_cp',h3_dit_tp=2,sp=4,h3_tp_comm='bf16',h3_cp_comm='fp8',
                h3_cuda_precision='fp8-sage',task_type='t2va')
    initial=SimpleNamespace(**values)
    validate_resident_args(initial,SimpleNamespace(**(values|dict(task_type='fl2va',frames=362))))
    with pytest.raises(ValueError,match='Restart the manager'):
        validate_resident_args(initial,SimpleNamespace(**(values|change)))
