"""Keep automatic H3 routing without loading an inactive resident transformer."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class WorkflowSelectionTests(unittest.TestCase):
    def test_component_loading_keeps_automatic_routing(self):
        path = Path(__file__).resolve().parents[1] / 'x_diffusers/adaptor/load_pipe.py'
        function = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'load_minimax_h3_pipe')
        route = next(n for n in function.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "task_type == 'ref2va'")
        options = next(n for n in function.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'component_options')
        loading = next(n for n in function.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == 'packed_checkpoint')
        code = compile(ast.Module(body=[route, options, loading], type_ignores=[]), str(path), 'exec')
        for task, precision, refiner, custom in (
            ('fl2va', 'fp8-bf16', False, False),
            ('t2va', 'fp8-sage', False, False),
            ('ref2va', 'fp8-sage', False, False),
            ('fl2va', 'none', True, False),
            ('fl2va', 'none', False, False),
            ('fl2va', 'fp8-bf16', False, True),
        ):
            with self.subTest(task=task, precision=precision, refiner=refiner, custom=custom):
                loader = Mock()
                args = SimpleNamespace(pretrained_model_name_or_path='weights/h3',
                                       h3_cuda_precision=precision, h3_refiner=refiner)
                exec(code, dict(args=args, task_type=task, ModularPipeline=loader, logger=Mock(),
                                dtype='bf16', packed_checkpoint=None,
                                transformer_path='weights/custom' if custom else None))
                workflow = task if task == 'ref2va' or refiner else None
                loader.from_pretrained.assert_called_once_with(
                    'weights/h3', **({'workflow': workflow} if workflow else {}))
                call = loader.from_pretrained.return_value.load_components.call_args
                self.assertIsNotNone(call)
                expected = 'fl2va' if task in ('t2va', 'fl2va') and not refiner and precision != 'none' else None
                self.assertEqual(call.kwargs.get('workflow'), expected)
                if expected is None:
                    self.assertNotIn('workflow', call.kwargs)


if __name__ == '__main__':
    unittest.main()
