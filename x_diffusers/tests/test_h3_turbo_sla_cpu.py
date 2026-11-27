"""Turbo-SLA assembly and repeated request scheduling with real CPU operators."""
from importlib import import_module

import ast
import builtins
import logging
import os
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

runtime = import_module("x_diffusers.adaptor.h3_runtime")
lora = import_module("x_diffusers.framework.lora.minimax_h3")


def load_turbo_sla_configurator():
    """Execute the loader's real function without accelerator patch initializers."""
    path = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    configure = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_h3_turbo_sla"
    )
    namespace = {"__package__": "x_diffusers.adaptor", "logger": logging.getLogger(__name__), "os": os}
    exec(compile(ast.Module(body=[configure], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["configure_h3_turbo_sla"]


configure_h3_turbo_sla = load_turbo_sla_configurator()


def attention_blocks():
    block = torch.nn.Module()
    block.attn = torch.nn.Identity()
    return torch.nn.ModuleList([block])


def make_pipe():
    transformer = torch.nn.Module()
    transformer.proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    torch.nn.init.zeros_(transformer.proj.weight)
    transformer.transformer_blocks = attention_blocks()
    transformer.token_refiner = torch.nn.Module()
    transformer.token_refiner.refiner_blocks = attention_blocks()
    return SimpleNamespace(
        transformer=transformer, scheduler=MiniMaxH3Scheduler(), audio_scheduler=MiniMaxH3Scheduler(),
        text_encoder=torch.nn.Identity(), vae=torch.nn.Identity(),
    )


class TurboSLATests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "adapter.safetensors"
        self.weights = {
            "transformer.proj.lora_down.weight": torch.tensor([[1.0, 2.0]]),
            "transformer.proj.lora_up.weight": torch.tensor([[3.0], [4.0]]),
            "transformer.proj.alpha": torch.tensor(1.0),
        }
        save_file(self.weights, self.path)
        self.args = SimpleNamespace(
            model="MiniMax-H3", task_type="t2va", h3_turbo_sla_lora=str(self.path),
            adaln_cache_file=None, num_inference_steps=4, atten="none", dtype="bf16",
            matmul_quant="none", lora_path_list=None, fuse_lora=False, phaa_num=0,
            ring_degree=1, sp=1, fsdp=None, inf_vram_blocks_num=None, cache_dit=False,
            turbo_mode="default", height=768, width=1344, frames=124,
        )

    def test_disabled_assembly_does_not_load_adapter_dependencies(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "safetensors"} or name.endswith("framework.lora.minimax_h3"):
                raise AssertionError("Disabled Turbo-SLA imported " + name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            configure_h3_turbo_sla(object(), SimpleNamespace())

    def test_lora_output_and_sparse_flags_stay_local_to_h3_attention(self):
        pipe = make_pipe()
        configure_h3_turbo_sla(pipe, self.args)
        self.assertIsInstance(pipe.transformer.proj, lora.TurboSLALinear)
        values = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(pipe.transformer.proj(values), torch.tensor([[12.0, 16.0]], dtype=torch.bfloat16))
        self.assertEqual(pipe._h3_turbo_sla_path, str(self.path.resolve()))
        self.assertEqual(pipe._h3_turbo_sla_pairs, 1)
        for block in [*pipe.transformer.transformer_blocks, *pipe.transformer.token_refiner.refiner_blocks]:
            self.assertTrue(block.attn._h3_turbo_sla)
        for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
            self.assertFalse(hasattr(module, "_h3_turbo_sla"))
        for scheduler in (pipe.scheduler, pipe.audio_scheduler):
            self.assertIs(scheduler.step.__func__, lora.training_euler_step)

    def test_repeated_requests_reset_both_four_step_schedules(self):
        for task in ("t2va", "fl2va"):
            pipe = make_pipe()
            args = SimpleNamespace(**{**vars(self.args), "task_type": task})
            configure_h3_turbo_sla(pipe, args)
            for request in range(2):
                runtime.validate_turbo_sla_schedule(pipe, args, 5)
                for name, shift in (("scheduler", 6.0), ("audio_scheduler", 3.0)):
                    with self.subTest(task=task, request=request, scheduler=name):
                        scheduler = getattr(pipe, name)
                        self.assertIsNone(scheduler.step_index)
                        grid = torch.linspace(1.0, 0.0, 5)
                        sigmas = shift * grid / (1 + (shift - 1) * grid)
                        torch.testing.assert_close(scheduler.sigmas, sigmas, rtol=0, atol=0)
                        self.assertEqual(len(scheduler.timesteps), 4)
                        self.assertEqual(scheduler.timesteps.device.type, "cpu")
                        state = torch.zeros(2)
                        velocity = torch.tensor([0.5, -1.0])
                        for timestep in scheduler.timesteps:
                            state = scheduler.step(velocity, timestep, state).prev_sample
                        self.assertEqual(scheduler.step_index, 4)
                        torch.testing.assert_close(state, velocity)

    def test_adapter_change_requires_rebuild_before_scheduler_reset(self):
        pipe = make_pipe()
        configure_h3_turbo_sla(pipe, self.args)
        other = self.root / "other.safetensors"
        save_file(self.weights, other)
        self.args.h3_turbo_sla_lora = str(other)
        with patch.object(pipe.scheduler, "set_timesteps") as video, patch.object(pipe.audio_scheduler, "set_timesteps") as audio:
            with self.assertRaisesRegex(RuntimeError, "rebuild the pipeline"):
                runtime.validate_turbo_sla_schedule(pipe, self.args, 5)
            video.assert_not_called()
            audio.assert_not_called()

    def test_wrong_scheduler_type_prevents_lora_installation(self):
        for name in ("scheduler", "audio_scheduler"):
            with self.subTest(scheduler=name):
                pipe = make_pipe()
                setattr(pipe, name, object())
                with self.assertRaisesRegex(TypeError, "requires Diffusers MiniMaxH3Scheduler"):
                    configure_h3_turbo_sla(pipe, self.args)
                self.assertIsInstance(pipe.transformer.proj, torch.nn.Linear)
                self.assertFalse(hasattr(pipe, "_h3_turbo_sla_path"))

    def test_lora_changing_cached_adaln_is_rejected_before_attention_flags(self):
        pipe = make_pipe()
        pipe.transformer.norm_out = torch.nn.Module()
        pipe.transformer.norm_out.linear = torch.nn.Linear(2, 2, bias=False)
        save_file({key.replace("proj.", "norm_out.linear."): value for key, value in self.weights.items()}, self.path)
        self.args.adaln_cache_file = "selected-cache.safetensors"
        with self.assertRaisesRegex(ValueError, "cached AdaLN"):
            configure_h3_turbo_sla(pipe, self.args)
        self.assertIsInstance(pipe.transformer.norm_out.linear, torch.nn.Linear)
        self.assertFalse(hasattr(pipe.transformer.transformer_blocks[0].attn, "_h3_turbo_sla"))

    def test_request_parameter_errors_precede_scheduler_reset(self):
        pipe = make_pipe()
        configure_h3_turbo_sla(pipe, self.args)
        for field, value in (("num_inference_steps", 8), ("fuse_lora", True), ("atten", "laser")):
            with self.subTest(field=field):
                args = SimpleNamespace(**{**vars(self.args), field: value})
                with patch.object(pipe.scheduler, "set_timesteps") as video, patch.object(pipe.audio_scheduler, "set_timesteps") as audio:
                    with self.assertRaises(ValueError):
                        runtime.validate_turbo_sla_schedule(pipe, args, 5)
                    video.assert_not_called()
                    audio.assert_not_called()

    def test_changed_scheduler_count_contract_is_rejected(self):
        for name in ("scheduler", "audio_scheduler"):
            for evaluations, points in ((5, 5), (4, 4)):
                with self.subTest(scheduler=name, evaluations=evaluations, points=points):
                    pipe = make_pipe()
                    pipe._h3_turbo_sla_path = str(self.path.resolve())
                    setattr(pipe, name, SimpleNamespace(set_timesteps=Mock(), timesteps=[0] * evaluations, sigmas=[0] * points))
                    with self.assertRaisesRegex(RuntimeError, "step-count contract changed"):
                        runtime.validate_turbo_sla_schedule(pipe, self.args, 5)


if __name__ == "__main__":
    unittest.main()
