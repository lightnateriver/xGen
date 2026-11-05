"""CPU checks for H3 request-boundary cleanup without loading model weights."""
import ast
import gc
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

ROOT = Path(__file__).resolve().parents[2]
ADAPTOR = ROOT / 'x_diffusers/x_diffusers/adaptor'


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[method_name]


class Output:
    pass


class MemoryTests(unittest.TestCase):
    def test_device_gc_sync_release_order_and_observability(self):
        for backend in ('cuda', 'npu'):
            with self.subTest(backend=backend):
                events = []
                accelerator = SimpleNamespace(
                    synchronize=lambda: events.append('sync'),
                    memory_allocated=Mock(side_effect=[100, 100]),
                    memory_reserved=Mock(side_effect=[200, 100]),
                    empty_cache=lambda: events.append('empty'),
                )
                torch = SimpleNamespace(get_device_module=Mock(return_value=accelerator))
                spec = importlib.util.spec_from_file_location('h3_memory_test', ADAPTOR / 'h3_memory.py')
                memory = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, {'torch': torch,
                        'x_base.device': SimpleNamespace(device_type=lambda: backend)}):
                    spec.loader.exec_module(memory)
                with (patch.object(memory, 'gc', SimpleNamespace(collect=lambda: events.append('gc'))),
                      patch.object(memory, 'logger') as logger):
                    memory.release_request_memory()
                self.assertEqual(events, ['gc', 'sync', 'empty', 'sync'])
                torch.get_device_module.assert_called_once_with(backend)
                self.assertEqual(logger.info.call_args.args[3:], (100, 100, 200, 100))

    def test_offline_cleanup_follows_completed_base_or_refiner(self):
        for refiner in (False, True):
            with self.subTest(refiner=refiner):
                events = []
                method = load_method(ADAPTOR / 'infer_tools.py', 'InferenceManager', 'infer_single',
                                     {'__package__': 'x_diffusers.adaptor'})
                def infer(args):
                    events.append('saved')
                release = lambda: events.append('release')
                manager = SimpleNamespace(_infer_minimax_h3=infer)
                with patch.dict(sys.modules, {'x_diffusers.adaptor.h3_memory':
                                             SimpleNamespace(release_request_memory=release)}):
                    method(manager, SimpleNamespace(model='MiniMax-H3', h3_refiner=refiner))
                self.assertEqual(events, ['saved', 'release'])

    def test_offline_failed_request_does_not_report_successful_cleanup(self):
        method = load_method(ADAPTOR / 'infer_tools.py', 'InferenceManager', 'infer_single',
                             {'__package__': 'x_diffusers.adaptor'})
        manager = SimpleNamespace(_infer_minimax_h3=Mock(side_effect=RuntimeError('decode failed')))
        release = Mock()
        with patch.dict(sys.modules, {'x_diffusers.adaptor.h3_memory':
                                     SimpleNamespace(release_request_memory=release)}):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                method(manager, SimpleNamespace(model='MiniMax-H3'))
        release.assert_not_called()

    def test_online_releases_output_after_save_before_completion(self):
        for maas in (False, True):
            with self.subTest(maas=maas):
                events, outputs = [], []
                def infer(*args, **kwargs):
                    output = Output()
                    output.cycle = output
                    outputs.append(weakref.ref(output))
                    return output
                def save(*args):
                    self.assertIsNotNone(outputs[0]())
                    events.append('saved')
                def release():
                    gc.collect()
                    self.assertIsNone(outputs[0]())
                    events.append('release')
                method = load_method(ROOT / 'server/video/video_server.py', 'VideoGenerationService',
                                     '_process_task', dict(torch=SimpleNamespace(cuda=Mock()), time=time,
                                      os=SimpleNamespace(makedirs=Mock(), path=SimpleNamespace(dirname=lambda p: '.')),
                                      infer_info=Mock(), logger=Mock(), _cleanup_temp_files=Mock()))
                manager = SimpleNamespace(initialize_pipe=Mock(), infer_single=infer)
                service = SimpleNamespace(_prepare_task=lambda task: True,
                    _build_args=lambda *args: SimpleNamespace(model='MiniMax-H3'), _set_guidance_scale=Mock(),
                    inference_manager=manager, maas_health=None, _save_video=save, _publish_maas_task=save,
                    _complete_task=lambda *args: events.append('completed'))
                with patch.dict(sys.modules, {'x_diffusers.adaptor.h3_memory':
                                             SimpleNamespace(release_request_memory=release)}):
                    method(service, 'test', {}, 'test.mp4', maas_task=object() if maas else None)
                self.assertEqual(events, ['saved', 'release'] if maas else ['saved', 'release', 'completed'])

    def test_online_worker_cleans_once_before_next_command(self):
        events = []
        commands = iter([1, 2])
        method = load_method(ROOT / 'server/video/video_server.py', 'VideoGenerationService', '_worker_loop',
            dict(logger=Mock(), broadcast_int=lambda *a, **kw: next(commands), CMD_HEARTBEAT=0,
                 CMD_EXECUTE=1, CMD_SHUTDOWN=2, broadcast_dict=lambda *a, **kw: dict(data={}, save_path='test.mp4'),
                 dist=Mock(), infer_info=Mock()))
        service = SimpleNamespace(rank=1, maas_fl2va_input=False, maas_health=None,
            _build_args=lambda *args: SimpleNamespace(model='MiniMax-H3'),
            inference_manager=SimpleNamespace(initialize_pipe=Mock(),
                infer_single=lambda *args: events.append('generated')))
        with patch.dict(sys.modules, {'x_diffusers.adaptor.h3_memory':
                                     SimpleNamespace(release_request_memory=lambda: events.append('release'))}):
            method(service)
        self.assertEqual(events, ['generated', 'release'])


if __name__ == '__main__':
    unittest.main()
