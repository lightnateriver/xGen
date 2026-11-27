"""Exercise batch execution and the separate VAE evaluator without model weights."""

import argparse
import ast
from copy import deepcopy
from contextlib import nullcontext
import csv
import importlib.util
import io
import logging
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


files = load_file("batch_files", "x_base/x_base/utils/file.py")
config = load_file("batch_config", "x_diffusers/x_diffusers/adaptor/config.py")
info = load_file("batch_info", "x_base/x_base/utils/infer_info.py")
evaluation = load_file("vae_evaluation", "tools/evaluate_vae.py")


class BatchInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "inputs"
        for case_id in ("10", "2", "1"):
            case_dir = self.input_dir / case_id
            case_dir.mkdir(parents=True)
            (case_dir / "image.jpg").write_bytes(b"input placeholder")
            (case_dir / "prompt.txt").write_text(f"prompt, {case_id}\n", encoding="utf-8")
        parser = config.add_network_args(config.add_abilities_args(config.add_inference_args(argparse.ArgumentParser())))
        self.args = parser.parse_args([])
        vars(self.args).update(
            batch_input_dir=str(self.input_dir), batch_sample_size=0, seed=42,
            task_type="i2v", width=832, height=480, frames=81,
            vae_lightning="none", vae_output_dir="", vae_pad_mode="all_sides_valid",
            vae_pad_latent_size=8, vae_pad_content="constant", vae_use_blend=False,
        )
        self.timing = info.InferInfo()
        self.dist = SimpleNamespace(is_initialized=lambda: False, get_rank=lambda: 0, barrier=Mock())
        self.torch = SimpleNamespace(distributed=self.dist, cuda=SimpleNamespace(synchronize=Mock()))
        source = ROOT / "x_diffusers/x_diffusers/adaptor/infer_tools.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InferenceManager")
        # Execute the real batch methods; model loading and device calls are mocked.
        manager.body = [node for node in manager.body if getattr(node, "name", "") in {"infer_batch", "_get_batch_output_dir"}]
        namespace = dict(
            os=os, deepcopy=deepcopy, list_cases=files.list_cases, read_text=files.read_text,
            refiner_worker_scope=lambda *args: nullcontext(),
            infer_info=self.timing, torch=self.torch, time=time, logger=logging.getLogger("batch_test"),
        )
        exec(compile(ast.Module(body=[manager], type_ignores=[]), str(source), "exec"), namespace)
        self.manager = namespace["InferenceManager"]()
        self.manager._prepare_pipeline = Mock()
        self.calls = []

        def infer_single(args):
            self.calls.append(deepcopy(args))
            self.timing.vae_last_encode_time = "1.25"
            self.timing.vae_last_decode_time = "2.50"

        self.manager.infer_single = Mock(side_effect=infer_single)

    def options(self, mode, *extra):
        return evaluation.build_parser().parse_args([
            "--eval_mode", mode, "--eval_report_dir", str(self.root / "reports"), *extra,
        ])

    def test_normal_batch_preserves_order_inputs_paths_and_arguments(self):
        original = deepcopy(vars(self.args))
        self.assertEqual(self.manager.infer_batch(self.args), [])
        self.assertEqual([Path(args.i2v_image_path).parent.name for args in self.calls], ["1", "2", "10"])
        for args in self.calls:
            case_dir = Path(args.i2v_image_path).parent
            self.assertEqual(args.prompt, f"prompt, {case_dir.name}")
            self.assertEqual(Path(args.save_path), case_dir / "i2v/output.mp4")
        self.assertEqual(vars(self.args), original)
        self.manager._prepare_pipeline.assert_called_once_with(self.args)
        self.assertFalse((self.root / "reports").exists())

    def test_sampling_is_repeatable_and_does_not_change_global_random_state(self):
        random.seed(7)
        before = random.getstate()
        self.args.batch_sample_size = 2
        self.manager.infer_batch(self.args)
        selected = [args.i2v_image_path for args in self.calls]
        self.calls.clear()
        self.manager.infer_batch(self.args)
        self.assertEqual([args.i2v_image_path for args in self.calls], selected)
        self.assertEqual(len(selected), 2)
        self.assertEqual(random.getstate(), before)

    def test_explicit_empty_selection_does_not_run_the_whole_dataset(self):
        self.assertEqual(self.manager.infer_batch(self.args, case_dirs=[]), [])
        self.manager._prepare_pipeline.assert_not_called()
        self.manager.infer_single.assert_not_called()

    def test_failed_case_is_collected_and_following_cases_continue(self):
        self.manager.infer_single.side_effect = [RuntimeError("failed case"), None, None]
        with self.assertLogs("batch_test", level="ERROR"):
            failed = self.manager.infer_batch(self.args)
        self.assertEqual(failed, [{"case_id": "1", "error": "failed case"}])
        self.assertEqual(self.manager.infer_single.call_count, 3)

    def test_normal_lightning_layout_is_preserved(self):
        self.args.vae_lightning = "decoder"
        self.args.vae_pad_mode = "all_sides"
        self.args.vae_output_dir = "vae"
        self.manager.infer_batch(self.args, case_dirs=[str(self.input_dir / "1")])
        output_dir = self.input_dir / "1/i2v/all_sides/decoder/pad_content-constant/use_blend-False/pad_latent_size-8"
        self.assertEqual(Path(self.calls[0].save_path), output_dir / "output.mp4")
        self.assertTrue((output_dir / "vae").is_dir())

    def test_speed_matrix_reuses_identical_cases_and_writes_measurements(self):
        options = self.options("speed", "--eval_resolutions", "832x480", "--eval_frames", "17", "--eval_targets", "none,decoder")
        cases = files.list_cases(str(self.input_dir), sample_size=1, seed=42)
        self.assertEqual(evaluation.run_evaluation(self.manager, self.args, options, cases, timing_info=self.timing), [])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0].i2v_image_path, self.calls[1].i2v_image_path)
        self.assertTrue(all(args.enable_vae_time_count for args in self.calls))
        with (options.eval_report_dir / "report_speed.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], list(evaluation.SPEED_COLUMNS))
        self.assertEqual([row[3:] for row in rows[1:]], [["none", "1.25", "2.50"], ["decoder", "1.25", "2.50"]])
        self.assertFalse(hasattr(self.args, "enable_vae_time_count"))
        self.assertEqual(self.args.vae_lightning, "none")

    def test_accuracy_matrix_does_not_overwrite_combinations(self):
        options = self.options("accuracy", "--eval_resolutions", "832x480,1280x720", "--eval_frames", "17,49", "--eval_targets", "none,encoder,decoder")
        evaluation.run_evaluation(self.manager, self.args, options, [self.input_dir / "1"])
        self.assertEqual(len({args.save_path for args in self.calls}), 12)
        self.assertTrue(all(not args.enable_vae_time_count for args in self.calls))
        self.assertTrue((options.eval_report_dir / "832x480_17f/original").is_dir())
        with (options.eval_report_dir / "prompt.txt").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(list(csv.reader(stream)), [["1.mp4", "prompt, 1"]])
        self.assertTrue((options.eval_report_dir / "任务描述.txt").is_file())
        self.assertFalse((options.eval_report_dir / "report_speed.csv").exists())

    def test_evaluation_failure_keeps_combination_context(self):
        options = self.options("speed", "--eval_resolutions", "832x480", "--eval_frames", "17", "--eval_targets", "decoder")
        self.manager.infer_single.side_effect = RuntimeError("failed case")
        with self.assertLogs(level="ERROR"):
            failed = evaluation.run_evaluation(self.manager, self.args, options, [self.input_dir / "1"], timing_info=self.timing)
        self.assertEqual(failed[0], dict(case_id="1", error="failed case", width=832, height=480, frames=17, vae_lightning="decoder"))
        with (options.eval_report_dir / "report_speed.csv").open(encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.reader(stream))), 1)

    def test_nonzero_rank_does_not_write_reports_or_invoke_callbacks(self):
        self.dist.is_initialized = lambda: True
        self.dist.get_rank = lambda: 1
        options = self.options("speed", "--eval_resolutions", "832x480", "--eval_frames", "17", "--eval_targets", "none")
        evaluation.run_evaluation(self.manager, self.args, options, [self.input_dir / "1"], rank=1, timing_info=self.timing)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(options.eval_report_dir.exists())
        self.dist.barrier.assert_called()

    def test_invalid_matrix_fails_before_inference_or_report_creation(self):
        for flag, value in (("--eval_resolutions", "832x0"), ("--eval_resolutions", "invalid"), ("--eval_frames", "0"), ("--eval_frames", "17,"), ("--eval_targets", "invalid")):
            with self.subTest(flag=flag, value=value):
                options = self.options("speed", flag, value)
                with self.assertRaises(ValueError):
                    evaluation.run_evaluation(self.manager, self.args, options, [], timing_info=self.timing)
        self.manager._prepare_pipeline.assert_not_called()
        self.assertFalse((self.root / "reports").exists())

    def test_normal_parser_has_only_batch_inference_options(self):
        parser = config.add_abilities_args(config.add_inference_args(argparse.ArgumentParser()))
        args = parser.parse_args(["--batch_input_dir", "inputs", "--batch_sample_size", "2"])
        self.assertEqual((args.batch_input_dir, args.batch_sample_size), ("inputs", 2))
        self.assertFalse(hasattr(args, "enable_vae_time_count"))
        self.assertFalse(any(name.startswith("batch_test_") or name.startswith("eval_") for name in vars(args)))
        for flag in ("--batch_test_mode", "--batch_test_report_dir", "--enable_vae_time_count"):
            with self.subTest(flag=flag), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag, "speed"])

    def test_cli_forwards_only_inference_arguments_and_returns_failure_status(self):
        runtime = SimpleNamespace(
            parse_args=Mock(return_value=self.args),
            init_env=Mock(),
            InferenceManager=lambda: self.manager,
        )
        base = SimpleNamespace(infer_info=self.timing, list_cases=files.list_cases)
        self.manager.infer_single.side_effect = RuntimeError("failed case")
        modules = {"x_diffusers": runtime, "x_base": base, "torch": self.torch, "torch.distributed": self.dist}
        with patch.dict(sys.modules, modules), self.assertLogs(level="ERROR"):
            status = evaluation.main([
                "--eval_mode", "speed", "--eval_report_dir", str(self.root / "reports"),
                "--eval_resolutions", "832x480", "--eval_frames", "17", "--eval_targets", "none",
                "--batch_input_dir", str(self.input_dir), "--seed", "42",
            ])
        self.assertEqual(status, 1)
        runtime.parse_args.assert_called_once_with(argv=["--batch_input_dir", str(self.input_dir), "--seed", "42"])
        runtime.init_env.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
