import argparse
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_size', ROOT/'x_diffusers/adaptor/h3_references.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Pipe:
    def __init__(self, edge=2048):
        self.config = dict(reference_image_short_edge=edge, canvas_short_edge=768, canvas_max_pixels=768*1344)
        self.updates = []

    def update_components(self, **kwargs):
        self.config.update(kwargs)
        self.updates.append(kwargs)


def test_unset_preserves_checkpoint_and_other_workflows():
    for task in ('t2va', 'fl2va', 'ref2va'):
        for edge in (2048, 1536):
            pipe = Pipe(edge=edge)
            module.configure_reference_image_size(pipe, SimpleNamespace(task_type=task))
            assert not pipe.updates and pipe.config['reference_image_short_edge'] == edge


@pytest.mark.parametrize('edge', [32, 768, 1024, 1536, 2048])
def test_explicit_size_only_updates_shared_reference_input(edge):
    pipe = Pipe()
    module.configure_reference_image_size(pipe, SimpleNamespace(task_type='ref2va', h3_reference_image_short_edge=edge))
    assert pipe.updates == [dict(reference_image_short_edge=edge)]
    assert pipe.config['canvas_short_edge'] == 768
    assert pipe.config['canvas_max_pixels'] == 768*1344


@pytest.mark.parametrize('edge', [False, True, 0, -32, 31, 769, 2080, 768.0, '768', float('nan')])
def test_invalid_sizes_fail_before_update(edge):
    pipe = Pipe()
    with pytest.raises(ValueError, match='integer multiple'):
        module.configure_reference_image_size(pipe, SimpleNamespace(task_type='ref2va', h3_reference_image_short_edge=edge))
    assert not pipe.updates


@pytest.mark.parametrize('task,refiner', [('t2va',False),('fl2va',False),('ref2va',True)])
def test_unvalidated_workflows_reject_explicit_override(task, refiner):
    pipe = Pipe()
    with pytest.raises(ValueError, match='Base only'):
        module.configure_reference_image_size(pipe, SimpleNamespace(task_type=task,h3_refiner=refiner,h3_reference_image_short_edge=768))
    assert not pipe.updates


def test_missing_upstream_config_does_not_silently_ignore_option():
    pipe = Pipe()
    del pipe.config['reference_image_short_edge']
    with pytest.raises(ValueError, match='does not expose'):
        module.configure_reference_image_size(pipe, SimpleNamespace(task_type='ref2va',h3_reference_image_short_edge=768))
    assert not pipe.updates


def test_cli_parses_optional_integer_without_initializing_hardware():
    tree=ast.parse((ROOT/'x_diffusers/adaptor/config.py').read_text(encoding='utf-8'))
    call=next(node for node in ast.walk(tree) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
              and node.value.args and isinstance(node.value.args[0],ast.Constant)
              and node.value.args[0].value=='--h3_reference_image_short_edge')
    parser=argparse.ArgumentParser()
    exec(compile(ast.Module(body=[call],type_ignores=[]),'<argument-registration>','exec'),{'group':parser})
    assert parser.parse_args([]).h3_reference_image_short_edge is None
    assert parser.parse_args(['--h3_reference_image_short_edge','768']).h3_reference_image_short_edge == 768


def test_server_startup_override_is_optional():
    tree=ast.parse((ROOT.parent/'server/video/video_server.py').read_text(encoding='utf-8'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_apply_mmh3_config')
    loop=next(node for node in function.body if isinstance(node,ast.For) and isinstance(node.iter,ast.Tuple)
              and any(isinstance(value,ast.Constant) and value.value=='h3_reference_image_short_edge' for value in node.iter.elts))
    service=SimpleNamespace(default_config={'h3_vae_backend':'native-stream'})
    code=compile(ast.Module(body=[loop],type_ignores=[]),'<service-startup>','exec')
    exec(code,{'service':service,'args':SimpleNamespace()})
    assert service.default_config == {'h3_vae_backend':'native-stream'}
    exec(code,{'service':service,'args':SimpleNamespace(h3_reference_image_short_edge=768)})
    assert service.default_config == {'h3_vae_backend':'native-stream','h3_reference_image_short_edge':768}
    exec(code,{'service':service,'args':SimpleNamespace(h3_reference_image_short_edge=2048)})
    assert service.default_config == {'h3_vae_backend':'native-stream','h3_reference_image_short_edge':2048}


@pytest.mark.parametrize('settings, error', [
    ({'h3_reference_image_resize_mode': 'mtach'}, 'Choose one of diffusers, max, match'),
    ({'h3_reference_image_resize_mode': 'match', 'h3_reference_image_short_edge': 768}, 'Set either'),
    ({'h3_reference_image_resize_mode': 'match', 'h3_refiner': True}, 'requires Ref2VA Base'),
    ({'h3_reference_image_resize_mode': 'match'}, None),
    ({'h3_reference_image_short_edge': 768}, None),
])
def test_server_validates_reference_resize_at_startup(settings, error, capsys):
    tree = ast.parse((ROOT.parent / 'server/video/video_server.py').read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    check = next(node for node in main.body if isinstance(node, ast.Try) and any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == 'reference_image_resize_mode' for call in ast.walk(node)))
    startup = compile(ast.Module(body=[check], type_ignores=[]), '<server-startup>', 'exec')
    namespace = dict(argparse=argparse, parser=argparse.ArgumentParser(),
                     service=SimpleNamespace(default_config=dict(task_type='ref2va', **settings)),
                     reference_image_resize_mode=module.reference_image_resize_mode)
    if error is None:
        exec(startup, namespace)
    else:
        with pytest.raises(SystemExit) as exited:
            exec(startup, namespace)
        assert exited.value.code == 2
        assert error in capsys.readouterr().err


@pytest.mark.parametrize('task, startup, request_data, expected_mode', [
    ('ref2va', {}, {}, 'max'),
    ('ref2va', {'h3_reference_image_resize_mode': None}, {}, 'max'),
    ('ref2va', {'h3_reference_image_resize_mode': 'diffusers'}, {}, 'diffusers'),
    ('ref2va', {'h3_reference_image_resize_mode': 'match'}, {}, 'match'),
    ('ref2va', {'h3_reference_image_short_edge': 768}, {}, None),
    ('ref2va', {'h3_refiner': True}, {}, None),
    ('fl2va', {}, {}, None),
    ('fl2va', {}, {'task_type': 'ref2va'}, 'max'),
    ('ref2va', {}, {'h3_reference_image_resize_mode': 'diffusers'}, 'max'),
])
def test_server_ref2va_resize_uses_deployment_settings(task, startup, request_data, expected_mode):
    tree = ast.parse((ROOT.parent / 'server/video/video_server.py').read_text(encoding='utf-8'))
    service_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == 'VideoGenerationService')
    function = next(node for node in service_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_build_args')
    namespace = dict(Any=Any, Dict=Dict, Optional=Optional,
                     MMH3_RESOLUTION_MAP={'768p': {'16:9': (1344, 768)}})
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<build_args>', 'exec'), namespace)

    class ConfigReady(Exception):
        pass

    def capture_config(config):
        raise ConfigReady(config)

    deployment = dict(model='MiniMax-H3', task_type=task, **startup)
    service = SimpleNamespace(default_config=deployment.copy(), maas=False,
                              _update_basic_config=lambda *args: None,
                              _update_path_config=lambda *args: None,
                              _config_to_arg_list=capture_config)
    with pytest.raises(ConfigReady) as captured:
        namespace['_build_args'](service, request_data, require_prompt=False)
    config, = captured.value.args
    assert config.get('h3_reference_image_resize_mode') == expected_mode
    assert config.get('h3_reference_image_short_edge') == startup.get('h3_reference_image_short_edge')
    assert service.default_config == deployment


@pytest.mark.parametrize('mode, expected', [
    ('diffusers', ((3648, 2048), (3648, 2048), (2720, 2048))),
    ('max', ((1920, 1088), (640, 352), (2720, 2048))),
    ('match', ((1344, 768), (640, 352), (1184, 864))),
])
def test_resize_modes_distinguish_small_medium_and_large_sources(mode, expected):
    sources = [(1920, 1080), (640, 360), (4000, 3000)]
    assert tuple(module.reference_image_size(*size, 1344, 768, 32, mode) for size in sources) == expected


def test_resize_mode_rejects_conflicting_short_edge_and_other_workflows():
    for overrides, message in (
        ({'h3_reference_image_short_edge': 768}, 'Set either'),
        ({'h3_refiner': True}, 'requires Ref2VA Base'),
        ({'task_type': 'fl2va'}, 'requires Ref2VA Base'),
        ({'h3_reference_image_resize_mode': 'unknown'}, 'Invalid'),
    ):
        args = dict(task_type='ref2va', h3_reference_image_resize_mode='match')
        args.update(overrides)
        with pytest.raises(ValueError, match=message):
            module.reference_image_resize_mode(SimpleNamespace(**args))
