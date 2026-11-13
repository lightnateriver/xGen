"""The NPU VC/fusions INT8 full-parallel Stream launch recipe for H3 Base."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from x_diffusers.adaptor.h3_cuda_precision import resolve
from x_diffusers.adaptor.h3_npu_options import H3_FUSION_FEATURES, validate as validate_fusions
from x_diffusers.framework.vae.omni_decoder import validate_options

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('task', ['t2va','fl2va','ref2va'])
def test_default_npu_recipe_keeps_portable_features_and_native_stream(task):
    name='h3_npu_ref2va.example.json' if task=='ref2va' else 'h3_npu_base.example.json'
    config=json.loads((ROOT/'example/diffusers/configs'/name).read_text(encoding='utf-8'))
    args=SimpleNamespace(**(config|dict(task_type=task)))
    assert resolve(args,'npu')=='none'
    validate_options(args,'npu')
    assert (args.atten,args.matmul_quant,args.vae_matmul_quant)==('vc','int8','none')
    assert args.h3_opts=='all' and validate_fusions(args,'npu')==H3_FUSION_FEATURES
    assert (args.sp,args.text_encoder_tp,args.phaa_num,args.num_inference_steps)==(8,8,2,20)
    for name in ('local_input','local_output','precompute_conditioning','input_refiner_sp',
                 'vision_sp','vae_encoder_parallel','audio_encoder_parallel'):
        assert getattr(args,'h3_cuda_'+name) is True
    assert args.h3_cuda_audio_decoder_parallel=='long'
    assert args.h3_cuda_dit_residency=='resident'
    assert (args.h3_cuda_input_refiner_attention,args.h3_cuda_input_refiner_ffn)==('legacy','bf16')
    assert args.h3_cuda_refiner_ff_chunk_size==(2048 if task=='ref2va' else 0)
    assert args.h3_vae_backend=='native-stream'
    assert not args.h3_cuda_phaa_fp8 and not args.h3_cuda_phaa_direct_pack


@pytest.mark.parametrize('task,full,precision,refiner,expected', [
    ('t2va',True,'none',False,{'workflow':'fl2va'}),
    ('fl2va',True,'none',False,{'workflow':'fl2va'}),
    ('t2va',False,'fp8-sage',False,{'workflow':'fl2va'}),
    ('t2va',False,'none',False,{}),
    ('ref2va',True,'none',False,{}),
    ('fl2va',True,'none',True,{}),
])
def test_workflow_component_filter_is_shared_without_changing_legacy_or_refiner(task,full,precision,refiner,expected):
    tree=ast.parse((ROOT/'x_diffusers/x_diffusers/adaptor/load_pipe.py').read_text(encoding='utf-8'))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='component_options' for t in n.targets))
    scope=dict(task_type=task,args=SimpleNamespace(h3_features='full-parallel' if full else None,
        h3_cuda_precision=precision,h3_refiner=refiner))
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<component-options>','exec'),scope)
    assert scope['component_options']==expected
