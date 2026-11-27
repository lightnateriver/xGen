"""Turbo8 prepared-asset compatibility and real scheduler integration on CPU."""
from importlib import import_module

import ast
from contextlib import redirect_stderr
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from diffusers import MiniMaxH3Scheduler
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

validation = import_module("x_diffusers.adaptor.validation")
settings = import_module("x_diffusers.framework.h3_config")
prepare = import_module("tools.adaln_cache.prepare_h3_base_cache")


def load_turbo8_configurator():
    """Execute the loader's real function without accelerator patch initializers."""
    path = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    configure = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_h3_turbo8"
    )
    namespace = {"__package__": "x_diffusers.adaptor", "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[configure], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["configure_h3_turbo8"]


configure_h3_turbo8 = load_turbo8_configurator()


class Turbo8AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "transformer"
        self.model.mkdir()
        self.manifest = self.model / "turbo8.json"
        self.cache = self.root / "arbitrary-name.safetensors"
        self.metadata = {
            "turbo8_lora_sha256": settings.H3_TURBO8_LORA_SHA256,
            "num_eval_steps": "8", "num_steps": "9",
            "video_shift": "6.0", "audio_shift": "3.0", "model_variant": "T2VA",
        }
        self.manifest.write_text(
            json.dumps({"turbo8_lora_sha256": settings.H3_TURBO8_LORA_SHA256}),
            encoding="utf-8",
        )
        save_file({}, self.cache, metadata=self.metadata)
        self.args = SimpleNamespace(transformer_path=str(self.model), adaln_cache_file=str(self.cache))

    def test_matching_prepared_assets_are_accepted(self):
        validation.validate_h3_turbo8_assets(self.args)

    def test_changed_model_fingerprint_is_rejected(self):
        self.manifest.write_text(json.dumps({"turbo8_lora_sha256": "other-weights"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validation.validate_h3_turbo8_assets(self.args)

    def test_cache_requires_matching_weights_schedule_shifts_and_variant(self):
        for key, wrong in (
            ("turbo8_lora_sha256", "other-weights"),
            ("num_eval_steps", "20"),
            ("video_shift", "12.0"),
            ("audio_shift", "6.0"),
            ("model_variant", "Ref2VA"),
        ):
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    metadata = dict(self.metadata)
                    if missing:
                        metadata.pop(key)
                    else:
                        metadata[key] = wrong
                    save_file({}, self.cache, metadata=metadata)
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        validation.validate_h3_turbo8_assets(self.args)

    def test_requires_prepared_transformer_and_cache_paths(self):
        for field in ("transformer_path", "adaln_cache_file"):
            with self.subTest(field=field):
                args = SimpleNamespace(**vars(self.args))
                setattr(args, field, None)
                with self.assertRaisesRegex(ValueError, "prepared transformer"):
                    validation.validate_h3_turbo8_assets(args)
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "prepared transformer"):
            validation.validate_h3_turbo8_assets(self.args)


class Turbo8SchedulerTests(unittest.TestCase):
    def test_both_modalities_execute_eight_training_euler_steps(self):
        for dtype in (torch.float32, torch.bfloat16):
            pipe = SimpleNamespace(scheduler=MiniMaxH3Scheduler(), audio_scheduler=MiniMaxH3Scheduler())
            configure_h3_turbo8(pipe)
            for name, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
                with self.subTest(dtype=dtype, scheduler=name):
                    scheduler = getattr(pipe, name)
                    grid = torch.linspace(1.0, 0.0, 9, dtype=torch.float32)
                    sigmas = shift * grid / (1 + (shift - 1) * grid)
                    torch.testing.assert_close(scheduler.sigmas, sigmas, rtol=0, atol=0)
                    torch.testing.assert_close(scheduler.timesteps, 1 - sigmas[:-1], rtol=0, atol=0)
                    state = torch.tensor([0.25, -0.5], dtype=dtype)
                    expected = state.float()
                    for index, timestep in enumerate(scheduler.timesteps):
                        velocity = torch.tensor([0.5, -1.0], dtype=dtype) * (index + 1)
                        expected = expected + (sigmas[index] - sigmas[index + 1]) * velocity.float()
                        result = scheduler.step(velocity, timestep, state, return_dict=index % 2 == 0)
                        state = result.prev_sample if index % 2 == 0 else result[0]
                    self.assertEqual(scheduler.step_index, 8)
                    self.assertEqual(state.dtype, torch.float32)
                    torch.testing.assert_close(state, expected, rtol=0, atol=0)

    def test_changed_scheduler_point_counts_fail_before_installing_step(self):
        for name in ("scheduler", "audio_scheduler"):
            for evaluations, points in ((9, 9), (8, 8)):
                with self.subTest(scheduler=name, evaluations=evaluations, points=points):
                    original_step = object()
                    bad = SimpleNamespace(
                        set_shift=Mock(), set_timesteps=Mock(),
                        timesteps=[0] * evaluations, sigmas=[0] * points, step=original_step,
                    )
                    pipe = SimpleNamespace(scheduler=MiniMaxH3Scheduler(), audio_scheduler=MiniMaxH3Scheduler())
                    setattr(pipe, name, bad)
                    with self.assertRaisesRegex(RuntimeError, "eight DiT evaluations"):
                        configure_h3_turbo8(pipe)
                    self.assertIs(bad.step, original_step)


class Turbo8PreparationTests(unittest.TestCase):
    def test_asset_preparation_checks_shared_fingerprint_before_device_work(self):
        assets = import_module("x_diffusers.framework.pipeline.minimax_h3")
        for matches in (False, True):
            with self.subTest(matches=matches), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = ["prepare_base_cache", "--source", "unused-weights", "--output", str(root / "cache"),
                        "--model-variant", "T2VA", "--stripped-output", str(root / "stripped"),
                        "--turbo8-lora", "unused-lora"]
                npu = SimpleNamespace(set_device=Mock(side_effect=RuntimeError("device boundary")))
                fingerprint = settings.H3_TURBO8_LORA_SHA256 if matches else "other-weights"
                stderr = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(assets, "digest", return_value=fingerprint),
                    patch.object(torch, "npu", npu, create=True),
                    patch.object(torch, "set_num_threads") as threads,
                    redirect_stderr(stderr),
                ):
                    if matches:
                        with self.assertRaisesRegex(RuntimeError, "device boundary"):
                            prepare.main()
                        npu.set_device.assert_called_once_with(0)
                    else:
                        with self.assertRaises(SystemExit) as error:
                            prepare.main()
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn("checksum mismatch", stderr.getvalue())
                        npu.set_device.assert_not_called()
                        threads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
