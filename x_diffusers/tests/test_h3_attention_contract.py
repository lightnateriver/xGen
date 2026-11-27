"""CPU-only public contract and default-path compatibility tests."""
from importlib import import_module
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'x_diffusers/x_diffusers/adaptor/validation.py'
spec=importlib.util.spec_from_file_location('_attention_contract',path)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
runtime = import_module("x_diffusers.adaptor.h3_runtime")
from x_diffusers.adaptor.load_pipe import configure_h3_attention



def args(**changes):
    values=dict(h3_attention='spark',h3_veda_predictor=None,model='MiniMax-H3',task_type='t2va',
                dtype='bf16',atten='laser',sp=8,num_inference_steps=50,frames=124,width=1344,height=768,
                save_fps=24,h3_refiner=False,keyframe_image='',keyframe_last_image='',reference_paths=[],
                h3_turbo_sla_lora=None,lora_path_list=None,cache_dit=False)
    values.update(changes)
    return SimpleNamespace(**values)



from x_diffusers.adaptor import config as _parameter_config

class ContractTests(unittest.TestCase):
    def setUp(self):
        environment=patch.dict(os.environ,TASK_QUEUE_ENABLE='1')
        environment.start();self.addCleanup(environment.stop)

    def test_spark_rejects_incompatible_graph_environment_before_loading(self):
        with patch.dict(os.environ,TASK_QUEUE_ENABLE='2'):
            with self.assertRaisesRegex(ValueError,'TASK_QUEUE_ENABLE'):
                contract.validate_h3_attention(args())
            contract.validate_h3_attention(SimpleNamespace(h3_attention='legacy'))

    def test_default_does_not_import_optional_or_torch(self):
        before=set(sys.modules)
        parser=_parameter_config.add_h3_attention_arguments(argparse.ArgumentParser())
        value=parser.parse_args([])
        self.assertEqual(value.h3_attention,'legacy')
        self.assertIsNone(contract.validate_h3_attention(value))
        self.assertFalse(any(s.startswith(('torch','h3_ascend','triton')) for s in set(sys.modules)-before))

    def test_veda_sparsity_is_an_offline_argument(self):
        offline = _parameter_config.add_h3_attention_arguments(argparse.ArgumentParser())
        self.assertEqual(offline.parse_args(['--h3_veda_sparsity', '0.7']).h3_veda_sparsity, 0.7)
        service_parser = _parameter_config.add_h3_attention_arguments(argparse.ArgumentParser(), offline=False)
        _, unknown = service_parser.parse_known_args(['--h3_veda_sparsity', '0.7'])
        self.assertEqual(unknown, ['--h3_veda_sparsity', '0.7'])

    def test_original_args_without_new_fields_still_disabled(self):
        self.assertIsNone(contract.validate_h3_attention(SimpleNamespace()))
        self.assertEqual(contract.h3_attention_key(SimpleNamespace()),('legacy',None,False,0.9))

    def test_50_steps_both_output_paths(self):
        contract.validate_h3_attention(args())
        contract.validate_h3_attention(args(h3_refiner=True,h3_refiner_seconds=5,width=2560,height=1440))

    def test_turbo8_veda_requires_explicit_profile_and_eight_evaluations(self):
        with tempfile.NamedTemporaryFile() as file:
            value = args(h3_attention='veda', h3_veda_predictor=file.name,
                         h3_turbo8=True, num_inference_steps=8)
            contract.validate_h3_attention(value)
            value.h3_turbo8 = False
            with self.assertRaises(ValueError):
                contract.validate_h3_attention(value)

    def test_veda_i2va_10s_and_sparsity(self):
        with tempfile.NamedTemporaryFile() as file:
            value = args(h3_attention='veda', h3_veda_predictor=file.name, task_type='fl2va',
                         frames=243, num_inference_steps=20, keyframe_image='first.png',
                         h3_veda_sparsity=0.7)
            contract.validate_h3_attention(value)
            value.h3_veda_sparsity = 0.8
            with self.assertRaisesRegex(ValueError, 'sparsity'):
                contract.validate_h3_attention(value)
            value.h3_veda_sparsity = 0.9
            value.keyframe_image = ''
            with self.assertRaisesRegex(ValueError, 'first frame'):
                contract.validate_h3_attention(value)

    def test_reject_wrong_shapes_models_and_schedule(self):
        for changes in [dict(model='Wan2.2-T2V-A14B'),dict(task_type='ref2va'),dict(task_type='fl2va'),
                        dict(sp=4),dict(dtype='fp16'),dict(atten='none'),dict(num_inference_steps=21),
                        dict(frames=243),dict(width=2560,height=1440),dict(cache_dit=True),
                        dict(h3_refiner=True,h3_refiner_seconds=10,width=2560,height=1440),
                        dict(keyframe_image='image.png'),dict(reference_paths=['r.png']),
                        dict(lora_path_list=['lora.safetensors']),dict(h3_attention='unknown'),
                        dict(num_inference_steps=50.0),dict(sp=8.0),dict(frames=124.0),
                        dict(width=1344.0),dict(height=768.0),dict(save_fps=24.0),
                        dict(h3_refiner=True,h3_refiner_seconds=5.0,width=2560,height=1440)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):contract.validate_h3_attention(args(**changes))

    def test_predictor_required_only_for_veda(self):
        with self.assertRaises(ValueError):contract.validate_h3_attention(args(h3_attention='veda'))
        with tempfile.NamedTemporaryFile() as file:
            contract.validate_h3_attention(args(h3_attention='veda',h3_veda_predictor=file.name))

    def test_service_preserves_startup_predictor_and_rejects_request_changes(self):
        config=dict(h3_attention='veda',h3_veda_predictor='weights/predictor.safetensors')
        contract.apply_h3_attention_startup_settings(config,args(h3_attention='legacy'),[])
        self.assertEqual(config['h3_attention'],'veda')
        for data in [dict(h3_attention='spark'),dict(h3_veda_predictor='elsewhere'),dict(duration=10),
                     dict(duration=True),dict(task_type='ref2va'),dict(resolution='2k'),dict(num_inference_steps=21),
                     dict(first_frame='image.png'),dict(attention_sparse=True),dict(num_inference_steps=50.0)]:
            errors=[];contract.validate_h3_attention_request(config,data,errors)
            self.assertTrue(errors,data)

    def test_service_uses_native_124_frames_and_50_actual_steps(self):
        value=args(num_inference_steps=40,frames=120)
        config=dict(h3_attention='spark',num_inference_steps=50)
        contract.configure_h3_attention_args(config,dict(prompt='test'),value)
        self.assertEqual(value.frames,124)
        self.assertEqual(value.num_inference_steps,50)
        spec=contract.h3_attention_parameter_spec(config)
        self.assertEqual(spec['parameters']['duration']['allowed_values'],[5])

    def test_explicit_legacy_rollback_restores_original_warmup(self):
        config=dict(h3_attention='veda')
        contract.apply_h3_attention_startup_settings(config,args(h3_attention='legacy'),['--h3_attention','legacy'])
        self.assertEqual(config['h3_attention'],'legacy')
        contract.validate_h3_attention_warmup(config,True)
        with self.assertRaisesRegex(ValueError,'50-step'):
            contract.validate_h3_attention_warmup(dict(h3_attention='spark'),True)
        contract.validate_h3_attention_warmup(dict(h3_attention='spark'),False)

    def test_optional_attention_preserves_legacy_and_installs_only_when_selected(self):
        # Check the actual installation boundary rather than freezing unrelated
        # VAE source bytes to a release before native-stream support existed.
        before = set(sys.modules)
        configure_h3_attention(object(), SimpleNamespace(h3_attention='legacy'))
        self.assertNotIn('x_diffusers.framework.transformer.minimax_h3', set(sys.modules)-before)
        for backend in ('spark', 'veda'):
            install = Mock()
            pipe = SimpleNamespace(transformer=object())
            value = args(h3_attention=backend, h3_veda_predictor=__file__)
            with patch.dict(sys.modules, {'x_diffusers.framework.transformer.minimax_h3':
                                        SimpleNamespace(install_h3_attention=install)}):
                configure_h3_attention(pipe, value)
            install.assert_called_once()
            model, config = install.call_args.args
            self.assertIs(model, pipe.transformer)
            self.assertEqual(config.key, contract.h3_attention_key(value))
            self.assertFalse(hasattr(config, 'save_path'))

    def test_request_validation_happens_before_model_session_begins(self):
        request = args(h3_attention='veda', h3_veda_predictor=__file__, phaa_num=2, save_path='out.mp4')
        pipe = SimpleNamespace(transformer=SimpleNamespace(_h3_attention_session=Mock()))
        runtime.begin_h3_attention_request(pipe, request)
        session_request = pipe.transformer._h3_attention_session.begin.call_args.args[0]
        self.assertEqual(session_request.config.key, contract.h3_attention_key(request))
        self.assertEqual((session_request.task_type, session_request.num_inference_steps), ('t2va', 50))
        self.assertFalse(hasattr(session_request, 'sp'))
        pipe.transformer._h3_attention_session.begin.reset_mock()
        request.frames = 125
        with self.assertRaises(ValueError):
            runtime.begin_h3_attention_request(pipe, request)
        pipe.transformer._h3_attention_session.begin.assert_not_called()

    def test_vendored_veda_accepts_shared_bundle_without_vllm_dependency(self):
        source=(ROOT/'x_base/x_base/operator/h3_attention/veda_vendor.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        self.assertIn('shared_bundle',[a.arg for a in init.args.args])
        self.assertNotIn('vllm',source)


if __name__=='__main__':unittest.main(verbosity=2)
