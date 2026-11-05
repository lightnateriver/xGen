"""Dependency-light regression tests. Run directly with unittest; no NPU required."""
import argparse
import ast
import copy
from contextlib import nullcontext
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import re
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
ADAPTOR = ROOT/'x_diffusers/x_diffusers/adaptor'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = ModuleType('_h3_test'); package.__path__ = [str(ADAPTOR)]
sys.modules['_h3_test'] = package
contract = load('_h3_test.h3_refiner_config', ADAPTOR/'h3_refiner_config.py')
service_config = load('_h3_test.h3_refiner_service', ADAPTOR/'h3_refiner_service.py')
references = sys.modules['_h3_test.h3_references']
config = load('_h3_test.config', ADAPTOR/'config.py')


def args(**overrides):
    with patch.object(sys, 'argv', ['test']):
        result = config.parse_args()
    for key, value in overrides.items():
        setattr(result,key,value)
    return result


class PortableBaseline(ast.NodeTransformer):
    """Normalize v0.1 example asset locations; do not alter inference logic."""
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            parts = node.value.split('/')
            if parts[:3] == ['', 'path', 'to']:
                tail = parts[3:]
                if tail and tail[0] == 'models': tail[0] = 'weights'
                if tail and tail[0] == 'user-home': tail[0] = 'runtime'
                node.value = './' + '/'.join(tail)
        return node


BASELINE_COMMIT = 'ef1a3210ff7eea8c4d23227719352fb78e8e53cf'


def baseline(path):
    raw = subprocess.check_output(['git','show',BASELINE_COMMIT+':'+path], cwd=ROOT).decode()
    return ast.unparse(PortableBaseline().visit(ast.parse(raw)))


class WithoutDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Module = visit_FunctionDef


def logic_tree(text):
    return WithoutDocstrings().visit(ast.parse(text))


class ContractTests(unittest.TestCase):
    def test_refiner_draft_loads_reference_audio_weights_and_restores_them(self):
        source = ROOT/'x_diffusers/x_diffusers/framework/pipeline/h3_regenerate/pipeline.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'H3RefinerPipeline')
        draft = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_draft')
        draft.decorator_list = []
        torch = Mock()
        namespace = dict(torch=torch, schedule_points=lambda steps: steps + 1)
        exec(compile(ast.Module(body=[draft], type_ignores=[]), str(source), 'exec'), namespace)
        modules = {
            'diffusers.utils': SimpleNamespace(load_image=Mock()),
            'x_diffusers.adaptor.h3_refiner_config': SimpleNamespace(native_padding=lambda *args: nullcontext()),
            'x_diffusers.adaptor.h3_references': SimpleNamespace(
                load_references=Mock(), reference_conditioning=Mock(), select_adaln=Mock()),
        }
        for conditioning in ('both', 'video'):
            base = Mock(side_effect=RuntimeError('encode sentinel'))
            original = base._blocks
            base.blocks = SimpleNamespace(sub_blocks={'decode': object()})
            holder = Mock()
            lifecycle = Mock()
            lifecycle.attach_mock(holder, 'audio')
            lifecycle.attach_mock(base, 'base')
            runner = SimpleNamespace(base=base, base_audio=holder, base_weights=Mock(), base_vae=Mock(),
                                     metrics=SimpleNamespace(span=lambda *args: nullcontext()))
            request = SimpleNamespace(task_type='ref2va', prompt='test', num_inference_steps=20,
                                      seed=42, reference_paths=['reference'])
            modules['x_diffusers.adaptor.h3_references'].reference_conditioning.return_value = conditioning
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, 'encode sentinel'):
                namespace['_draft'](runner, request, {'generation_frames': 124})
            self.assertIs(base._blocks, original)
            expected = ['audio.onload', 'base', 'audio.offload'] if conditioning == 'both' else ['base']
            self.assertEqual([call[0] for call in lifecycle.mock_calls], expected)

    def test_partial_initialization_failure_cannot_silently_reload(self):
        package_name='_refiner_init_test'
        root=ModuleType(package_name);root.__path__=[]
        adaptor=ModuleType(package_name+'.adaptor');adaptor.__path__=[str(ADAPTOR)]
        loader=ModuleType(package_name+'.adaptor.load_pipe')
        loader.load_pipe=Mock(return_value=object())
        loader.update_lora=Mock();loader.update_pipe=Mock();loader.pipe_to_device=lambda pipe,args:pipe
        pipeline=ModuleType(package_name+'.framework.pipeline.h3_regenerate.pipeline')
        pipeline.H3RefinerPipeline=Mock(side_effect=RuntimeError('constructor sentinel'))
        torch_stub=ModuleType('torch');torch_stub.__path__=[]
        distributed=ModuleType('torch.distributed');distributed.is_initialized=lambda:True;distributed.get_world_size=lambda:8
        torch_stub.distributed=distributed
        av_writer=ModuleType('x_base.vae_parallelism.save_av_stream')
        av_writer.FFMPEG_PATH='ffmpeg'
        modules={package_name:root,package_name+'.adaptor':adaptor,
                 av_writer.__name__:av_writer,
                 package_name+'.adaptor.h3_refiner_config':contract,
                 package_name+'.adaptor.load_pipe':loader,
                 package_name+'.framework.pipeline.h3_regenerate.pipeline':pipeline,
                 'torch':torch_stub,'torch.distributed':distributed}
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,modules):
            adapter=load(package_name+'.adaptor.h3_refiner',ADAPTOR/'h3_refiner.py')
            value=args(h3_refiner=True,model='MiniMax-H3',task_type='t2va',sp=8,dtype='bf16',atten='laser',
                       num_inference_steps=50,width=2560,height=1440,save_fps=24,phaa_num=2,
                       matmul_quant='int8',vae_matmul_quant='none',vae_lightning='none',
                       inf_vram_blocks_num=0,h3_refiner_weights=folder,h3_refiner_prepared=folder)
            manager=SimpleNamespace(pipe=None)
            with patch.object(adapter,'which',return_value='ffmpeg'):
                with self.assertRaisesRegex(RuntimeError,'constructor sentinel'):adapter.prepare(manager,value)
            self.assertTrue(manager._h3_refiner_failed)
            with self.assertRaisesRegex(RuntimeError,'restart'):adapter.prepare(manager,value)
            self.assertEqual(loader.load_pipe.call_count,1)

    def test_default_disabled_without_assets_or_optional_imports(self):
        before = set(sys.modules)
        value = args()
        self.assertFalse(value.h3_refiner)
        self.assertIsNone(contract.validate_settings(value))
        self.assertFalse(any(name.startswith(('torch','diffusers','transformers')) for name in set(sys.modules)-before))

    def test_service_cli_does_not_erase_json_asset_paths(self):
        startup=dict(h3_refiner_weights='/weights',h3_refiner_prepared='/prepared',h3_refiner_seconds=15)
        value=args(h3_refiner=True)
        service_config.apply_startup_settings(startup,value,['--h3_refiner'])
        self.assertEqual(startup['h3_refiner_weights'],'/weights')
        self.assertEqual(startup['h3_refiner_seconds'],15)
        value.h3_refiner_seconds=10
        service_config.apply_startup_settings(startup,value,['--h3_refiner_seconds=10'])
        self.assertEqual(startup['h3_refiner_seconds'],10)

    def test_existing_cli_defaults_unchanged(self):
        original = baseline('x_diffusers/x_diffusers/adaptor/config.py')
        namespace = {'__name__': '_h3_test.baseline_config', '__package__': '_h3_test'}
        exec(compile(original,'baseline-config','exec'),namespace)
        with patch.object(sys, 'argv', ['test']):
            old = vars(namespace['parse_args']())
        now = vars(args())
        self.assertEqual(old, now)
        self.assertIsNone(now['h3_nvfp4_checkpoint'])
        for flag in ('h3_nvfp4_resident','h3_nvfp4_compile','h3_cuda_vae_sp'):
            self.assertFalse(now[flag])

    def test_exact_native_padding_and_output_counts(self):
        for seconds, generation, latent in [(5,124,37),(10,243,72),(15,362,107)]:
            spec = contract.profile(seconds)
            self.assertEqual((spec['output_frames'],spec['generation_frames'],spec['latent_frames']),
                             (seconds*24,generation,latent))
        for value in (True,5.0,'5',0,6,16,None):
            with self.assertRaises(ValueError): contract.profile(value)

    def test_padding_is_instance_scoped_and_restored_on_error(self):
        class Pipe:
            fps=24
            @property
            def max_duration(self): return 15
        first, other = Pipe(), Pipe()
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with contract.native_padding(first,contract.profile(15)):
                self.assertGreater(first.max_duration,15)
                self.assertEqual(other.max_duration,15)
                raise RuntimeError('sentinel')
        self.assertIs(type(first),Pipe)
        self.assertEqual(first.max_duration,15)

    def test_shape_changes_reuse_assets_but_workflow_changes_do_not(self):
        first = args(); second = copy.copy(first)
        second.h3_refiner_seconds=15; second.frames=362; second.seed=100; second.prompt='other'
        self.assertEqual(contract.model_key(first),contract.model_key(second))
        second.task_type='ref2va'
        self.assertNotEqual(contract.model_key(first),contract.model_key(second))

    def test_conflicts_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            valid=args(h3_refiner=True,model='MiniMax-H3',task_type='t2va',sp=8,dtype='bf16',atten='laser',
                       num_inference_steps=50,width=2560,height=1440,save_fps=24,phaa_num=2,
                       matmul_quant='int8',vae_matmul_quant='none',vae_lightning='none',
                       inf_vram_blocks_num=0,h3_refiner_weights=folder,h3_refiner_prepared=folder)
            self.assertEqual(contract.validate_settings(valid)['output_frames'],120)
            for key,value in [('sp',1),('num_inference_steps',51),('width',1344),('adopt_sr',True),
                              ('cache_dit',True),('vae_lightning','decoder'),('inf_vram_blocks_num',1),
                              ('vae_matmul_quant','int8'),('task_type','i2v'),('h3_refiner_weights','missing')]:
                bad=copy.copy(valid); setattr(bad,key,value)
                with self.subTest(key=key),self.assertRaises(ValueError): contract.validate_settings(bad)

    def test_reference_counts_decode_and_seed_types(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for index in range(10):
                path=Path(folder)/f'{index}.png';Image.new('RGB',(16,16)).save(path);paths.append(str(path))
            value=args(task_type='ref2va',prompt='test',seed=42)
            for count in (1, 5, 9):
                value.reference_paths = paths[:count]
                contract.validate_inputs(value)
            for images in ([], paths, [paths[0]]*5):
                value.reference_paths=images
                with self.assertRaises(ValueError): contract.validate_inputs(value)
            value.reference_paths=paths[:5];value.seed=True
            with self.assertRaises(ValueError):contract.validate_inputs(value)
            value.seed=42;value.reference_paths=paths[:1];Path(paths[0]).write_text('corrupt')
            with self.assertRaises(Exception):contract.validate_inputs(value)

    def test_reference_media_count_duration_and_conditioning(self):
        from decimal import Decimal
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = [root / f"image-{i}.png" for i in range(10)]
            videos = [root / f"video-{i}.mp4" for i in range(4)]
            audios = [root / f"audio-{i}.wav" for i in range(4)]
            for path in images:
                Image.new("RGB", (16, 16)).save(path)
            for path in videos + audios:
                path.touch()
            durations = dict(zip(videos + audios, [2, 5, 8, 6] * 2))

            def probe(path, kind):
                return Decimal(str(durations[path])), kind == "audio" or path == videos[1]

            def check(paths):
                return references.validate_references([str(path) for path in paths])

            with patch.object(references, "probe_clip", side_effect=probe):
                self.assertEqual(check([videos[0]]), "video")
                self.assertEqual(check([videos[1]]), "both")
                self.assertEqual(check(images[:6] + videos[:3] + audios[:3]), "both")
                for paths in (images, videos, images[:1] + audios, images[:9] + videos[:2] + audios[:2],
                              audios[:1], [videos[1], videos[2], videos[3]],
                              images[:1] + audios[1:]):
                    with self.subTest(paths=paths), self.assertRaises(ValueError):
                        check(paths)
                for seconds in (1.99, 15.01):
                    durations[videos[0]] = seconds
                    with self.assertRaisesRegex(ValueError, "Each video"):
                        check([videos[0]])
                durations[videos[0]] = 15
                self.assertEqual(check([videos[0]]), "video")

                pipe = SimpleNamespace(transformer=SimpleNamespace(
                    _adaln_cache=SimpleNamespace(_conditioning_type="video"),
                ))
                request = args(reference_paths=[str(videos[1])], adaln_cache_file="ref2va-21step.safetensors")
                installer = Mock()
                with patch.dict(sys.modules, {"x_base.adaln_cache": SimpleNamespace(install_adaln_cache=installer)}):
                    references.select_reference_adaln(pipe, request)
                installer.assert_called_once_with(pipe, "ref2va-21step.safetensors", conditioning_type="both")

    def test_service_profiles_and_immutable_startup_settings(self):
        fake=SimpleNamespace(default_config=dict(task_type='t2va',num_inference_steps=50))
        fake._validate_prompt=lambda data,errors:data.get('prompt','test')
        fake._validate_mmh3_frame_fields=lambda data,task,errors:('','')
        fake._validate_mmh3_ref_path=lambda data,task,errors:[]
        for duration in (5,10,15):
            ok,result=service_config.validate_request(fake,dict(prompt='test',duration=duration),[],[])
            self.assertTrue(ok,result)
            self.assertEqual(result['data']['resolution'],'2k')
        for data in [dict(h3_refiner=False),dict(duration=6),dict(resolution='1080p'),dict(task_type='ref2va'),
                     dict(num_inference_steps=21),dict(seed=True),dict(enable_audio='true'),dict(adopt_sr=True)]:
            ok,_=service_config.validate_request(fake,data,[],[]);self.assertFalse(ok,data)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_non_h3_paths_match_previous_release(self):
        # H3 steps and request cleanup intentionally change. Compare the actual
        # Wan branches, and the untouched CLI/VAE, against a fixed release.
        for path in ('x_diffusers/x_diffusers/adaptor/config.py',
                     'x_diffusers/x_diffusers/framework/vae/minimax_h3.py'):
            with self.subTest(path=path):
                old=ast.dump(logic_tree(baseline(path)),include_attributes=False)
                new=ast.dump(logic_tree((ROOT/path).read_text(encoding='utf-8')),include_attributes=False)
                self.assertEqual(old,new,'Default-path AST changed: '+path)
        for path, name in (('x_diffusers/x_diffusers/adaptor/infer_tools.py', 'InferenceManager'),
                           ('server/video/video_server.py', 'VideoInferenceManager')):
            methods = []
            for source in (baseline(path), (ROOT/path).read_text(encoding='utf-8')):
                cls = next(node for node in ast.parse(source).body
                           if isinstance(node, ast.ClassDef) and node.name == name)
                method = next(node for node in cls.body
                              if isinstance(node, ast.FunctionDef) and node.name == 'infer_single')
                method.body = [node for node in method.body if not (
                    isinstance(node, ast.If) and 'minimax-h3' in ast.unparse(node.test).lower())]
                methods.append(ast.dump(method, include_attributes=False))
            self.assertEqual(*methods, msg=path)
        path = 'x_diffusers/x_diffusers/adaptor/load_pipe.py'
        loaders = []
        for source in (baseline(path), (ROOT/path).read_text(encoding='utf-8')):
            tree = logic_tree(source)
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.FunctionDef) and node.name == '_apply_adaln_cache')]
            loaders.append(ast.dump(tree, include_attributes=False))
        self.assertEqual(*loaders, msg='Non-AdaLN loading paths changed')

    def test_no_independent_service_or_scheduler(self):
        folder=ROOT/'x_diffusers/x_diffusers/framework/pipeline/h3_regenerate'
        self.assertFalse((folder/'serve.py').exists())
        self.assertFalse((folder/'resident.py').exists())
        self.assertFalse((folder/'export.py').exists())
        source=(folder/'pipeline.py').read_text(encoding='utf-8')
        self.assertNotIn('InferenceManager(',source)
        self.assertNotIn('while True',source)
        self.assertNotIn('from_pretrained =',source)


def service_classes(source):
    """Execute real service methods without importing cloud storage or NPU packages."""
    tree=ast.parse(source)
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('VideoInferenceManager','VideoGenerationService')]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+classes,type_ignores=[])
    namespace=dict(sys=sys,os=os,re=re,math=math,Path=Path,logger=Mock(),parse_args=config.parse_args,
                   jsonify=lambda value:value,_resolve_input_file=lambda value:value,
                   MAAS_QUEUE_CAPACITY=10,MAAS_QUEUE_TIMEOUT_S=600,
                   H3_BASE_MAX_DURATION_S=contract.H3_BASE_MAX_DURATION_S,
                   MMH3_RESOLUTION_MAP={'768p':{'16:9':(1344,768)},'1080p':{'16:9':(1920,1080)}})
    exec(compile(ast.fix_missing_locations(module),'service-methods','exec'),namespace)
    return namespace


class ServiceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        path='server/video/video_server.py'
        self.old=service_classes(baseline(path))
        self.new=service_classes((ROOT/path).read_text(encoding='utf-8'))

    def instance(self,namespace,enabled):
        obj=namespace['VideoGenerationService'].__new__(namespace['VideoGenerationService'])
        obj.maas_fl2va_input=False
        obj.default_config=dict(model='MiniMax-H3',task_type='t2va',num_inference_steps=50,
                                h3_refiner=enabled,vae_lightning='none',sp=8)
        return obj

    def test_service_build_args_preserves_startup_steps_and_other_fields(self):
        left=self.instance(self.old,False);right=self.instance(self.new,False)
        for data in [dict(prompt='test'),dict(prompt='test',duration=10,resolution='1080p'),
                     dict(prompt='test',duration=5,seed=100)]:
            a=vars(left._build_args(data,'./test.mp4'));b=vars(right._build_args(data,'./test.mp4'))
            a.pop('num_inference_steps')
            self.assertEqual(b.pop('num_inference_steps'),50)
            self.assertEqual(a,b)

    def test_service_enabled_build_args_keeps_startup_model_and_changes_duration(self):
        service=self.instance(self.new,True)
        modules={'x_diffusers.adaptor.h3_refiner_service':service_config}
        with patch.dict(sys.modules,modules):
            for seconds in (5,10,15):
                value=service._build_args(dict(prompt='test',duration=seconds,resolution='2k'),'./test.mp4')
                self.assertTrue(value.h3_refiner)
                self.assertEqual((value.width,value.height,value.num_inference_steps),(2560,1440,50))
                self.assertEqual(value.frames,contract.profile(seconds)['generation_frames'])
                self.assertFalse(value.adopt_sr)

    def test_params_route_unchanged_by_default_and_has_opt_in_bounds(self):
        class App:
            def __init__(self):self.routes={}
            def route(self,path,**kwargs):
                def register(fn):self.routes[path]=fn;return fn
                return register
        services=[self.instance(self.old,False),self.instance(self.new,False),self.instance(self.new,True)]
        with patch.dict(sys.modules,{'x_diffusers.adaptor.h3_refiner_service':service_config}):
            for service in services:
                service.app=App();service._setup_config_routes()
            old=services[0].app.routes['/params']();new=services[1].app.routes['/params']()
            old['parameters'].pop('num_inference_steps')
            self.assertEqual(new['parameters'].pop('num_inference_steps')['default'],50)
            self.assertEqual(old,new)
            result=services[2].app.routes['/params']()
            self.assertTrue(result['experimental'])
            self.assertEqual(result['parameters']['duration']['allowed_values'],[5,10,15])

    def test_online_backend_cannot_be_silently_changed_in_place(self):
        cls=self.new['VideoInferenceManager']
        manager=cls.__new__(cls);manager._h3_refiner_key=('loaded',)
        with self.assertRaisesRegex(ValueError,'Restart'):
            manager.initialize_pipe(SimpleNamespace(h3_refiner=False))


if __name__=='__main__':
    unittest.main(verbosity=2)
