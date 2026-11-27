"""Check resolved switch snapshots without importing hardware-dependent modules."""
from importlib import import_module

import ast
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Keep model type identity stable while each fixture replaces only the device module.
import_module("x_diffusers.framework.h3_config")


def load_resolver():
    path = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_h3_execution_options"
    )
    namespace = dict(
        __package__="x_diffusers.adaptor",
        os=os,
        dist=SimpleNamespace(is_initialized=lambda: False, get_world_size=lambda: 1),
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["resolve_h3_execution_options"]


class InstallationPlanTests(unittest.TestCase):
    def resolve(self, args, pipe):
        with patch.dict(
            sys.modules, {"x_base.device": SimpleNamespace(device_type=lambda: "cuda")}
        ):
            return load_resolver()(args, pipe)

    def test_snapshot_is_model_owned_immutable_and_reused(self):
        args = SimpleNamespace(h3_cuda_precision="none")
        pipe = SimpleNamespace()
        first = self.resolve(args, pipe)
        self.assertIs(self.resolve(args, pipe), first)
        self.assertEqual(first.parallel.mode, "sp")
        self.assertFalse(hasattr(first.parallel, "from_args"))
        self.assertFalse(hasattr(first, "prompt"))
        with self.assertRaises(FrozenInstanceError):
            first.sp = 8

    def test_changed_installation_switch_requires_rebuild(self):
        args = SimpleNamespace(h3_cuda_precision="none")
        pipe = SimpleNamespace()
        self.resolve(args, pipe)
        args.h3_cuda_refiner_ff_chunk_size = 2048
        with self.assertRaisesRegex(ValueError, "Restart"):
            self.resolve(args, pipe)

    def test_request_content_is_not_part_of_installation_identity(self):
        args = SimpleNamespace(
            h3_cuda_precision="none", prompt="first", seed=1, frames=124
        )
        pipe = SimpleNamespace()
        first = self.resolve(args, pipe)
        args.prompt, args.seed, args.frames = "second", 2, 243
        self.assertIs(self.resolve(args, pipe), first)

    def test_fp8_preset_fusions_are_captured_once_at_the_application_boundary(self):
        args = SimpleNamespace(h3_cuda_precision="fp8-bf16")
        pipe = SimpleNamespace()
        with patch.dict(os.environ, {"XGEN_H3_FUSED_OUTPUT_ADALN": "1"}):
            first = self.resolve(args, pipe)
        self.assertIn("XGEN_H3_FUSED_OUTPUT_ADALN", first.dit_fusions)
        with patch.dict(os.environ, {"XGEN_H3_FUSED_OUTPUT_ADALN": "0"}):
            self.assertIs(self.resolve(args, pipe), first)


if __name__ == "__main__":
    unittest.main()
