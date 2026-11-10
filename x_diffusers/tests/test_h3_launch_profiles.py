"""Validate public launch recipes without loading models or allocating devices."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
pytestmark=pytest.mark.skipif(os.name=='nt',reason='Linux deployment launcher contract')


@pytest.fixture
def launch_env(tmp_path):
    prompt=tmp_path/'prompt.txt';prompt.write_text('public launcher contract',encoding='utf-8')
    stub=tmp_path/'torchrun'
    stub.write_text('''#!/usr/bin/env python3
import json,os,sys
keys=['XGEN_DEVICE','XGEN_PHAA_FP8','XGEN_H3_FUSED_OUTPUT_ADALN','PYTORCH_CUDA_ALLOC_CONF','NCCL_MAX_CTAS']
print(json.dumps(dict(args=sys.argv[1:],env={k:os.environ.get(k) for k in keys})))
''')
    stub.chmod(0o755)
    env={key:value for key,value in os.environ.items() if not key.startswith(('H3_','XGEN_','PYTORCH_'))}
    env.update(PATH=str(tmp_path)+':'+env['PATH'],PROMPT_FILE=str(prompt),OUTPUT_PATH=str(tmp_path/'out.mp4'))
    return env


@pytest.mark.parametrize('backend,task,precision',[
    ('cuda','t2va',None),('cuda','i2va',None),('cuda','r2va',None),('cuda','t2va','none'),
    ('npu','t2va',None),('npu','i2va',None),('npu','r2va',None),
    ('cuda','t2va','fp8-sage'),('cuda','i2va','fp8-sage'),
])
def test_launcher_profile_contract(launch_env,backend,task,precision,monkeypatch):
    env=launch_env
    if precision:env['H3_CUDA_PRECISION']=precision
    # Old transport settings are cleared; the Sage preset must restore its own.
    env['XGEN_PHAA_FP8']='1'
    command=['bash','example/diffusers/scripts/run_h3.sh',backend,task,'5','768p']
    result=json.loads(subprocess.check_output(command,cwd=ROOT,env=env,text=True))
    args,actual=result['args'],result['env']
    if backend=='cuda':
        assert args[args.index('--h3_cuda_precision')+1]==(precision or 'fp8-sage')
    else:assert '--h3_cuda_precision' not in args
    if backend=='cuda' and precision in (None,'fp8-sage'):
        assert actual['XGEN_PHAA_FP8']==actual['XGEN_H3_FUSED_OUTPUT_ADALN']=='1'
        assert actual['PYTORCH_CUDA_ALLOC_CONF']=='expandable_segments:True'
        assert actual['NCCL_MAX_CTAS']=='2'
    elif backend=='cuda':
        assert actual['XGEN_PHAA_FP8'] is None
        assert actual['XGEN_H3_FUSED_OUTPUT_ADALN'] is None
        assert actual['PYTORCH_CUDA_ALLOC_CONF'] is None
    assert args[args.index('--num_inference_steps')+1]==('20' if backend=='npu' else '50')
    if backend=='npu':
        assert args[args.index('--h3_vae_backend')+1]=='native-stream'
        assert args[args.index('--vae_matmul_quant')+1]=='none'
        assert args[args.index('--matmul_quant')+1]=='int8'
        assert args[args.index('--h3_features')+1]=='full-parallel'
        assert args[args.index('--h3_refiner_ff_chunk_size')+1]==('2048' if task=='r2va' else '0')
        assert args[args.index('--adaln_cache_file')+1].endswith('-21step.safetensors')
        # Validate the emitted recipe, including workflow-specific guards.
        import sys
        from x_diffusers.adaptor.config import parse_args
        from x_diffusers.adaptor.h3_cuda_precision import resolve
        monkeypatch.setattr(sys,'argv',['infer.py',*args[args.index('--model'):]])
        parsed=parse_args()
        assert resolve(parsed,'npu')=='none'


def test_npu_explicit_steps_and_backend_rollback(launch_env):
    launch_env['H3_BASE_STEPS']='50'
    command=['bash','example/diffusers/scripts/run_h3.sh','npu','r2va','5','768p','ref.png',
             '--','--h3_vae_backend','lightning','--vae_matmul_quant','int8',
             '--h3_features','none','--h3_refiner_ff_chunk_size','0']
    args=json.loads(subprocess.check_output(command,cwd=ROOT,env=launch_env,text=True))['args']
    assert args[args.index('--num_inference_steps')+1]=='50'
    assert args[args.index('--adaln_cache_file')+1].endswith('-51step.safetensors')
    assert args[-8:]==command[-8:]


@pytest.mark.parametrize('backend,task,resolution',[
    ('npu','t2va','768p'),('cuda','i2va','2k'),
])
def test_sage_rejects_unsupported_launcher_routes(launch_env,backend,task,resolution):
    launch_env['H3_CUDA_PRECISION']='fp8-sage'
    result=subprocess.run(['bash','example/diffusers/scripts/run_h3.sh',backend,task,'5',resolution],
                          cwd=ROOT,env=launch_env,text=True,capture_output=True)
    assert result.returncode==2 and 'fp8-sage' in result.stderr


def test_sage_rejects_conflicting_allocator(launch_env):
    launch_env.update(H3_CUDA_PRECISION='fp8-sage',PYTORCH_ALLOC_CONF='expandable_segments:False')
    result=subprocess.run(['bash','example/diffusers/scripts/run_h3.sh','cuda','t2va','5','768p'],
                          cwd=ROOT,env=launch_env,text=True,capture_output=True)
    assert result.returncode==2 and 'PYTORCH_ALLOC_CONF' in result.stderr


def test_cuda5090_r2va_preset_and_explicit_overrides(launch_env):
    command=['bash','example/diffusers/scripts/run_h3.sh','cuda-5090','r2va','5','768p','ref.png',
             '--','--no-h3_cuda_local_input','--h3_cuda_input_refiner_ffn','bf16']
    result=json.loads(subprocess.check_output(command,cwd=ROOT,env=launch_env,text=True))
    args=result['args']
    assert args[args.index('--h3_cuda_preset')+1]=='sage-fp8-full-parallel'
    assert args[-3:]==['--no-h3_cuda_local_input','--h3_cuda_input_refiner_ffn','bf16']
    launch_env['H3_CUDA_PRESET']='none'
    result=json.loads(subprocess.check_output(command,cwd=ROOT,env=launch_env,text=True))
    assert result['args'][result['args'].index('--h3_cuda_preset')+1]=='none'


@pytest.mark.parametrize('task',['t2va','i2va'])
def test_cuda5090_all_workflows_select_new_default(launch_env,task):
    result=json.loads(subprocess.check_output(['bash','example/diffusers/scripts/run_h3.sh',
        'cuda-5090',task,'5','768p'],cwd=ROOT,env=launch_env,text=True))
    args=result['args']
    assert args[args.index('--h3_cuda_preset')+1]=='sage-fp8-full-parallel'
    assert args[args.index('--h3_cuda_adaln_mode')+1]=='auto'
    assert args[args.index('--h3_cuda_precision')+1]=='fp8-sage'


@pytest.mark.parametrize('backend,task,resolution',[
    ('npu','r2va','768p'),('cuda','i2va','2k')])
def test_full_parallel_wrong_route_fails_before_launch(launch_env,backend,task,resolution):
    launch_env['H3_CUDA_PRESET']='sage-fp8-full-parallel'
    result=subprocess.run(['bash','example/diffusers/scripts/run_h3.sh',backend,task,'5',resolution],
        cwd=ROOT,env=launch_env,text=True,capture_output=True)
    assert result.returncode==2 and 'full-parallel' in result.stderr


def test_table_options_are_independent_of_full_preset(launch_env):
    launch_env.update(H3_CUDA_PRECISION='fp8-bf16', H3_CUDA_ADALN_MODE='table',
                      H3_CUDA_ADALN_DIR='model-cache/custom')
    result=json.loads(subprocess.check_output(['bash','example/diffusers/scripts/run_h3.sh',
        'cuda','t2va','5','768p'],cwd=ROOT,env=launch_env,text=True))
    args=result['args']
    assert args[args.index('--h3_cuda_preset')+1]=='none'
    assert args[args.index('--h3_cuda_adaln_mode')+1]=='table'
    assert args[args.index('--h3_cuda_adaln_dir')+1]=='model-cache/custom'
