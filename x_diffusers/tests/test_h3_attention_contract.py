"""CPU-only public contract and default-path compatibility tests."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'x_diffusers/x_diffusers/adaptor/h3_attention_config.py'
spec=importlib.util.spec_from_file_location('_attention_contract',path)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
package_name='_h3_attention_contract_package'
from types import ModuleType
package=ModuleType(package_name);package.__path__=[str(path.parent)]
sys.modules[package_name]=package
service_spec=importlib.util.spec_from_file_location(package_name+'.h3_attention_service',path.parent/'h3_attention_service.py')
service=importlib.util.module_from_spec(service_spec);service_spec.loader.exec_module(service)


def args(**changes):
    values=dict(h3_attention='spark',h3_veda_predictor=None,model='MiniMax-H3',task_type='t2va',
                dtype='bf16',atten='laser',sp=8,num_inference_steps=51,frames=124,width=1344,height=768,
                save_fps=24,h3_refiner=False,keyframe_image='',keyframe_last_image='',reference_paths=[],
                h3_turbo_sla_lora=None,lora_path_list=None,cache_dit=False)
    values.update(changes)
    return SimpleNamespace(**values)


class ContractTests(unittest.TestCase):
    def setUp(self):
        environment=patch.dict(os.environ,TASK_QUEUE_ENABLE='1')
        environment.start();self.addCleanup(environment.stop)

    def test_spark_rejects_incompatible_graph_environment_before_loading(self):
        with patch.dict(os.environ,TASK_QUEUE_ENABLE='2'):
            with self.assertRaisesRegex(ValueError,'TASK_QUEUE_ENABLE'):
                contract.validate(args())
            contract.validate(SimpleNamespace(h3_attention='legacy'))

    def test_default_does_not_import_optional_or_torch(self):
        before=set(sys.modules)
        parser=contract.add_arguments(argparse.ArgumentParser())
        value=parser.parse_args([])
        self.assertEqual(value.h3_attention,'legacy')
        self.assertIsNone(contract.validate(value))
        self.assertFalse(any(s.startswith(('torch','h3_ascend','triton')) for s in set(sys.modules)-before))

    def test_original_args_without_new_fields_still_disabled(self):
        self.assertIsNone(contract.validate(SimpleNamespace()))
        self.assertEqual(contract.key(SimpleNamespace()),('legacy',None))

    def test_50_steps_both_output_paths(self):
        contract.validate(args())
        contract.validate(args(h3_refiner=True,h3_refiner_seconds=5,width=2560,height=1440))

    def test_reject_wrong_shapes_models_and_schedule(self):
        for changes in [dict(model='Wan2.2-T2V-A14B'),dict(task_type='ref2va'),dict(task_type='fl2va'),
                        dict(sp=4),dict(dtype='fp16'),dict(atten='none'),dict(num_inference_steps=21),
                        dict(frames=243),dict(width=2560,height=1440),dict(cache_dit=True),
                        dict(h3_refiner=True,h3_refiner_seconds=10,width=2560,height=1440),
                        dict(keyframe_image='image.png'),dict(reference_paths=['r.png']),
                        dict(lora_path_list=['lora.safetensors']),dict(h3_attention='unknown')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):contract.validate(args(**changes))

    def test_predictor_required_only_for_veda(self):
        with self.assertRaises(ValueError):contract.validate(args(h3_attention='veda'))
        with tempfile.NamedTemporaryFile() as file:
            contract.validate(args(h3_attention='veda',h3_veda_predictor=file.name))

    def test_service_preserves_startup_predictor_and_rejects_request_changes(self):
        config=dict(h3_attention='veda',h3_veda_predictor='weights/predictor.safetensors')
        service.apply_startup_settings(config,args(h3_attention='legacy'),[])
        self.assertEqual(config['h3_attention'],'veda')
        for data in [dict(h3_attention='spark'),dict(h3_veda_predictor='elsewhere'),dict(duration=10),
                     dict(duration=True),dict(task_type='ref2va'),dict(resolution='2k'),dict(num_inference_steps=21),
                     dict(first_frame='image.png'),dict(attention_sparse=True)]:
            errors=[];service.validate_request(config,data,errors)
            self.assertTrue(errors,data)

    def test_service_uses_native_124_frames_and_50_actual_steps(self):
        value=args(num_inference_steps=40,frames=120)
        config=dict(h3_attention='spark',num_inference_steps=51)
        service.configure_args(config,dict(prompt='test'),value)
        self.assertEqual(value.frames,124)
        self.assertEqual(value.num_inference_steps,51)
        spec=service.parameter_spec(config)
        self.assertEqual(spec['parameters']['duration']['allowed_values'],[5])

    def test_v03_does_not_modify_refiner_loop_or_legacy_codecs(self):
        for name in ['x_diffusers/x_diffusers/framework/pipeline/h3_regenerate/pipeline.py',
                     'x_diffusers/x_diffusers/framework/vae/minimax_h3.py',
                     'x_diffusers/x_diffusers/framework/vae/h3_refiner_output.py',
                     'x_base/x_base/vae_parallelism/save_av_stream.py']:
            old=subprocess.check_output(['git','show','v0.2:'+name],cwd=ROOT)
            self.assertEqual(old,(ROOT/name).read_bytes(),name)

    def test_vendored_veda_accepts_shared_bundle_without_vllm_dependency(self):
        source=(ROOT/'x_base/x_base/operator/h3_attention/veda_vendor.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        self.assertIn('shared_bundle',[a.arg for a in init.args.args])
        self.assertNotIn('vllm',source)


if __name__=='__main__':unittest.main(verbosity=2)
