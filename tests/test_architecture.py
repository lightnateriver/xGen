"""Static H3 layout and CLI checks without model runtime dependencies."""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_h3_model_components_use_two_files_per_layer(self):
        framework = ROOT / "x_diffusers/x_diffusers/framework"
        for area in ("pipeline", "transformer", "vae"):
            with self.subTest(area=area):
                names = {path.name for path in (framework / area).glob("*h3*.py")}
                self.assertEqual(names, {"minimax_h3.py", "minimax_h3_image.py"})
                self.assertEqual(list((framework / area / "h3").glob("*.py")), [])

    def test_model_settings_have_no_asset_or_device_imports(self):
        path = ROOT / "x_diffusers/x_diffusers/framework/h3_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or ""
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        self.assertFalse(
            any(
                name.split(".")[0]
                in {"torch", "diffusers", "os", "pathlib", "safetensors"}
                for name in imports
            )
        )
        self.assertEqual(
            list((ROOT / "x_diffusers/x_diffusers/contracts").glob("*.py")), []
        )

    def test_image_cli_parses_before_importing_models(self):
        path = ROOT / "example/diffusers/infer_minimax_h3_image.py"
        spec = importlib.util.spec_from_file_location("image_cli_boundary", path)
        module = importlib.util.module_from_spec(spec)
        before = set(sys.modules)
        spec.loader.exec_module(module)
        args = module.build_parser().parse_args(
            [
                "--pretrained_model_name_or_path",
                "weights",
                "--prompt",
                "test",
                "--save_path",
                "out.png",
            ]
        )
        self.assertEqual(args.num_inference_steps, 6)
        self.assertIsNone(args.image_text_int8)
        self.assertFalse(
            any(
                name.startswith(("torch", "diffusers", "x_diffusers"))
                for name in set(sys.modules) - before
            )
        )


if __name__ == '__main__':
    unittest.main()
