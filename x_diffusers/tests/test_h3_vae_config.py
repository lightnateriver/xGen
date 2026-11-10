"""One public H3 selector, original internal Lightning implementation."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h3_vae_config',ROOT/'x_diffusers/adaptor/h3_vae_config.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('selected,internal', [('legacy','none'),('lightning','decoder'),('native-stream','none')])
def test_public_choice_reuses_existing_switch_and_is_idempotent(selected,internal):
    args=SimpleNamespace(model='MiniMax-H3',sp=8,h3_vae_backend=selected,vae_lightning='none')
    assert module.resolve_vae_backend(args)==selected
    assert args.vae_lightning==internal
    assert module.resolve_vae_backend(args)==selected
    assert args.vae_lightning==internal


def test_h3_old_enable_flag_is_rejected():
    args=SimpleNamespace(model='MiniMax-H3',sp=8,vae_lightning='decoder')
    with pytest.raises(ValueError,match='--h3_vae_backend lightning'):
        module.resolve_vae_backend(args)


@pytest.mark.parametrize('selected,expected', [(None,'native-stream'),('legacy','legacy'),
                                               ('lightning','lightning'),('native-stream','native-stream')])
def test_full_parallel_default_and_explicit_override(selected,expected):
    args=SimpleNamespace(model='MiniMax-H3',sp=8,h3_cuda_preset='sage-fp8-full-parallel',
                         h3_vae_backend=selected,vae_lightning='none')
    assert module.resolve_vae_backend(args)==expected
    assert module.resolve_vae_backend(args)==expected


def test_unspecified_backend_preserves_legacy_without_preset():
    args=SimpleNamespace(model='MiniMax-H3',sp=8,h3_vae_backend=None,vae_lightning='none')
    assert module.resolve_vae_backend(args)=='legacy'


def test_non_h3_keeps_original_lightning_flag():
    args=SimpleNamespace(model='Wan2.2-T2V-A14B',vae_lightning='encoder_and_decoder')
    assert module.resolve_vae_backend(args)=='legacy'
    assert args.vae_lightning=='encoder_and_decoder'
    assert not hasattr(args,'_h3_vae_selection')


def test_lightning_sp1_does_not_silently_fall_back():
    args=SimpleNamespace(model='MiniMax-H3',sp=1,h3_vae_backend='lightning',vae_lightning='none')
    with pytest.raises(ValueError,match='--sp greater than 1'):module.resolve_vae_backend(args)


def test_server_keeps_npu_lightning_and_cuda_legacy_defaults():
    tree=ast.parse((ROOT.parent/'server/video/video_server.py').read_text(encoding='utf-8'))
    default=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_MINIMAX_H3_CONFIG' for t in n.targets))
    values=ast.literal_eval(default.value)
    assert values['h3_vae_backend']=='lightning' and values['vae_lightning']=='none'
    updates=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and isinstance(n.func.value,ast.Name) and n.func.value.id=='DEFAULT_MINIMAX_H3_CONFIG' and n.func.attr=='update']
    cuda=ast.literal_eval(updates[0].args[0])
    assert cuda['h3_vae_backend'] is None and cuda['vae_lightning']=='none'
    args=SimpleNamespace(model='MiniMax-H3',sp=8,**{k:cuda[k] for k in ('h3_vae_backend','vae_lightning')})
    assert module.resolve_vae_backend(args)=='legacy'


def test_service_applies_native_options_and_explicit_quant_none():
    import logging
    tree=ast.parse((ROOT.parent/'server/video/video_server.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_mmh3_config')
    scope={'VideoGenerationService':object,'logger':logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<service-config>','exec'),scope)
    service=SimpleNamespace(default_config={'h3_vae_backend':'lightning','vae_matmul_quant':'int8'})
    apply=scope['_apply_mmh3_config']
    apply(service,SimpleNamespace(),0)
    assert service.default_config=={'h3_vae_backend':'lightning','vae_matmul_quant':'int8'}
    apply(service,SimpleNamespace(h3_vae_backend='native-stream',h3_vae_model_path='weights/native/video_vae',
                                  h3_vae_trust_remote_code=True,vae_matmul_quant='none'),0)
    assert service.default_config=={'h3_vae_backend':'native-stream','vae_matmul_quant':'none',
        'h3_vae_model_path':'weights/native/video_vae','h3_vae_trust_remote_code':True}


@pytest.mark.parametrize('hardware,task',[('cuda','t2va'),('cuda','i2va'),('cuda-5090','r2va')])
def test_recommended_launchers_emit_native_backend_without_running_devices(tmp_path,hardware,task):
    import json,os,shutil,subprocess,sys
    if not shutil.which('bash') or os.name=='nt':pytest.skip('POSIX launch wrapper test')
    fake=tmp_path/'torchrun'
    fake.write_text('#!'+sys.executable+'\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    prompt=tmp_path/'prompt.txt';prompt.write_text('A person walks.')
    env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],PROMPT_FILE=str(prompt),
             OUTPUT_PATH=str(tmp_path/'video.mp4'),H3_VAE_MODEL_PATH='weights/native/video_vae',
             H3_VAE_TRUST_REMOTE_CODE='1')
    for name in ('H3_CUDA_PRECISION','H3_CUDA_PRESET','H3_VAE_BACKEND','H3_FAST'):
        env.pop(name,None)
    launcher=ROOT.parent/'example/diffusers/scripts/run_h3.sh'
    args=json.loads(subprocess.check_output(['bash',str(launcher),hardware,task,'5','768p'],env=env,text=True))
    selected=[args[i+1] for i,x in enumerate(args[:-1]) if x=='--h3_vae_backend']
    assert selected[-1]=='native-stream'
    assert args[args.index('--h3_vae_model_path')+1]=='weights/native/video_vae'
    assert '--h3_vae_trust_remote_code' in args
