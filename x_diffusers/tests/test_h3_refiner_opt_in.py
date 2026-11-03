"""Dependency-light regression tests. Run directly with unittest; no NPU required."""
import argparse
import ast
import copy
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


def baseline(path):
    raw = subprocess.check_output(['git','show','origin/v0.1:'+path], cwd=ROOT).decode()
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
                       num_inference_steps=51,width=2560,height=1440,save_fps=24,phaa_num=2,
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
        namespace = {}; exec(compile(original,'v0.1-config','exec'),namespace)
        with patch.object(sys, 'argv', ['test']):
            old = vars(namespace['parse_args']())
        now = vars(args())
        self.assertEqual(old, {key:now[key] for key in old})
        self.assertEqual(set(now)-set(old),{'h3_turbo8','h3_attention','h3_veda_predictor','h3_veda_sparsity','h3_refiner','h3_refiner_weights','h3_refiner_prepared',
                                         'h3_refiner_seconds','h3_refiner_loading'})

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
                       num_inference_steps=51,width=2560,height=1440,save_fps=24,phaa_num=2,
                       matmul_quant='int8',vae_matmul_quant='none',vae_lightning='none',
                       inf_vram_blocks_num=0,h3_refiner_weights=folder,h3_refiner_prepared=folder)
            self.assertEqual(contract.validate_settings(valid)['output_frames'],120)
            for key,value in [('sp',1),('num_inference_steps',50),('width',1344),('adopt_sr',True),
                              ('cache_dit',True),('vae_lightning','decoder'),('inf_vram_blocks_num',1),
                              ('vae_matmul_quant','int8'),('task_type','i2v'),('h3_refiner_weights','missing')]:
                bad=copy.copy(valid); setattr(bad,key,value)
                with self.subTest(key=key),self.assertRaises(ValueError): contract.validate_settings(bad)

    def test_reference_counts_decode_and_seed_types(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for index in range(5):
                path=Path(folder)/f'{index}.png';Image.new('RGB',(16,16)).save(path);paths.append(str(path))
            value=args(task_type='ref2va',reference_paths=paths,prompt='test',seed=42)
            contract.validate_inputs(value)
            for images in (paths[:4],paths+[paths[0]],[paths[0]]*5):
                value.reference_paths=images
                with self.assertRaises(ValueError): contract.validate_inputs(value)
            value.reference_paths=paths;value.seed=True
            with self.assertRaises(ValueError):contract.validate_inputs(value)
            value.seed=42;Path(paths[0]).write_text('corrupt')
            with self.assertRaises(Exception):contract.validate_inputs(value)

    def test_service_profiles_and_immutable_startup_settings(self):
        fake=SimpleNamespace(default_config=dict(task_type='t2va',num_inference_steps=51))
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


class LegacyBranchNormalizer(ast.NodeTransformer):
    """Specialize only feature gates to their default values; do not erase old logic."""
    def visit_ImportFrom(self,node):
        if node.module and node.module.endswith(('h3_refiner_config','h3_refiner_service','h3_attention_config')):
            return None
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('add_arguments','add_h3_attention_arguments'):
            return None
        if isinstance(node.value, ast.Call) and any(
                isinstance(arg, ast.Constant) and arg.value == '--h3_turbo8' for arg in node.value.args):
            return None
        return self.generic_visit(node)

    def visit_List(self, node):
        node.elts = [elt for elt in node.elts
                     if not isinstance(elt, ast.Constant) or elt.value != 'h3_turbo8']
        return self.generic_visit(node)

    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('add_arguments','add_h3_attention_arguments'):
            return None
        return self.generic_visit(node)

    def visit_If(self,node):
        test=ast.unparse(node.test)
        if 'h3_attention' in test or "getattr(args, 'h3_turbo8', False)" in test:
            return [self.visit(n) for n in node.orelse]
        if ('h3_refiner' in test and not isinstance(node.test,ast.BoolOp)) or "'failed', False" in test:
            return [self.visit(n) for n in node.orelse]
        if test=='frame_callback is None':
            return [self.visit(n) for n in node.body]
        return self.generic_visit(node)

    def visit_BoolOp(self,node):
        if isinstance(node.op,ast.And):
            node.values=[v for v in node.values if 'h3_refiner' not in ast.unparse(v)]
            if len(node.values)==1:return self.visit(node.values[0])
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        if node.name=='_decode_chunks_streaming':
            node.args.args=[a for a in node.args.args if a.arg!='frame_callback'];node.args.defaults=[]
        return self.generic_visit(node)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_disabled_original_code_paths_match_v01_ast(self):
        paths=['x_diffusers/x_diffusers/adaptor/config.py','x_diffusers/x_diffusers/adaptor/infer_tools.py',
               'x_diffusers/x_diffusers/adaptor/load_pipe.py','server/video/video_server.py',
               'x_diffusers/x_diffusers/framework/vae/minimax_h3.py']
        for path in paths:
            with self.subTest(path=path):
                original=baseline(path)
                old=ast.dump(logic_tree(original),include_attributes=False)
                new=ast.dump(LegacyBranchNormalizer().visit(logic_tree((ROOT/path).read_text(encoding='utf-8'))),include_attributes=False)
                self.assertEqual(old,new,'Default-path AST changed: '+path)

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
        obj.default_config=dict(model='MiniMax-H3',task_type='t2va',num_inference_steps=51,
                                h3_refiner=enabled,vae_lightning='none',sp=8)
        return obj

    def test_existing_service_build_args_is_identical_when_disabled(self):
        left=self.instance(self.old,False);right=self.instance(self.new,False)
        for data in [dict(prompt='test'),dict(prompt='test',duration=10,resolution='1080p'),
                     dict(prompt='test',duration=5,seed=100)]:
            a=vars(left._build_args(data,'./test.mp4'));b=vars(right._build_args(data,'./test.mp4'))
            self.assertEqual(a,b)

    def test_service_enabled_build_args_keeps_startup_model_and_changes_duration(self):
        service=self.instance(self.new,True)
        modules={'x_diffusers.adaptor.h3_refiner_service':service_config}
        with patch.dict(sys.modules,modules):
            for seconds in (5,10,15):
                value=service._build_args(dict(prompt='test',duration=seconds,resolution='2k'),'./test.mp4')
                self.assertTrue(value.h3_refiner)
                self.assertEqual((value.width,value.height,value.num_inference_steps),(2560,1440,51))
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
            self.assertEqual(services[0].app.routes['/params'](),services[1].app.routes['/params']())
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
