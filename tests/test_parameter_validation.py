"""Parameter rules exercised after normal xGen package initialization."""
from importlib import import_module

import builtins
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import_module("x_diffusers.adaptor.config")
import_module("x_diffusers.adaptor.validation")
import_module("x_diffusers.adaptor.h3_runtime")


class ParameterValidationTests(unittest.TestCase):
    def test_cached_parameter_modules_and_default_checks_need_no_new_device_imports(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torch_npu", "diffusers"):
                raise AssertionError("Parameter import initialized " + name)
            return original(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded),
            patch.dict(os.environ, {}, clear=True),
        ):
            config = import_module("x_diffusers.adaptor.config")
            checks = import_module("x_diffusers.adaptor.validation")
            runtime = import_module("x_diffusers.adaptor.h3_runtime")
            runtime.begin_h3_attention_request(object(), SimpleNamespace())
            with patch.object(sys, "argv", ["test"]):
                args = config.parse_args(bootstrap_environment=False)
            with patch.object(
                checks,
                "_distributed",
                return_value=SimpleNamespace(is_initialized=lambda: False),
            ):
                checks.validate_pipeline_args(args, "npu")
            self.assertEqual(args.model, "Wan2.2-T2V-A14B")
            self.assertEqual(args.h3_cuda_precision, "none")

    def test_extended_cuda_dit_fusions_default_and_rollback(self):
        config = import_module("x_diffusers.adaptor.config")
        checks = import_module("x_diffusers.adaptor.validation")
        for service in (False, True):
            parser = config.argparse.ArgumentParser()
            config.add_h3_cuda_arguments(parser, service=service)
            for flags, enabled in (([], True), (["--no-h3_cuda_dit_fusions"], False)):
                args = parser.parse_args(["--h3_cuda_preset", "sage-fp8-full-parallel", *flags])
                args.model, args.task_type, args.dtype, args.sp = "MiniMax-H3", "ref2va", "bf16", 8
                checks.resolve_h3_options(args, "cuda")
                self.assertIs(args.h3_cuda_dit_fusions, enabled)
                self.assertIs(dict(checks.h3_startup_key(args))["h3_cuda_dit_fusions"], enabled)

        args = SimpleNamespace(
            model="MiniMax-H3", task_type="t2va", h3_cuda_precision="fp8-bf16",
            h3_cuda_preset="none", h3_cuda_dit_fusions=True,
        )
        checks.resolve_h3_options(args, "cuda")
        self.assertTrue(args.h3_cuda_dit_fusions)

        for backend, model in (("npu", "MiniMax-H3"), ("cuda", "Wan2.2-T2V-A14B")):
            args = SimpleNamespace(model=model, task_type="t2va")
            checks.resolve_h3_options(args, backend)
            self.assertFalse(args.h3_cuda_dit_fusions)
            args.h3_cuda_dit_fusions = True
            with self.assertRaisesRegex(ValueError, "h3_cuda_dit_fusions"):
                checks.resolve_h3_options(args, backend)

    def test_existing_conflicts_keep_their_correction_behavior(self):
        checks = import_module("x_diffusers.adaptor.validation")
        args = SimpleNamespace(
            model="Wan2.2-I2V-A14B",
            task_type="t2i",
            frames=81,
            vae_lightning="decoder",
            sp=1,
            fuse_lora=True,
            matmul_quant="int8",
            joint=False,
            ada_brighten=True,
            text_encoder_tp=2,
            fsdp="all",
            transformer_path="h3-only",
        )
        checks.check_args_conflict(args)
        self.assertEqual(args.frames, 1)
        self.assertEqual(args.vae_lightning, "none")
        self.assertFalse(args.fuse_lora)
        self.assertEqual(args.fsdp, "transformer")
        self.assertIsNone(args.transformer_path)


if __name__ == "__main__":
    unittest.main()
