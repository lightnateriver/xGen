"""CPU regression coverage for the H3 step-count boundary."""
import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List
import unittest

import torch
from diffusers import MiniMaxH3Scheduler, UniPCMultistepScheduler
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
ADAPTOR = ROOT / "x_diffusers/x_diffusers/adaptor"
spec = importlib.util.spec_from_file_location("h3_schedule", ADAPTOR / "h3_schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class ScheduleTests(unittest.TestCase):
    def test_h3_euler_matches_omni_grid_and_update(self):
        for steps in (1, 4, 8, 20, 50):
            for shift in (3.0, 6.0, 12.0):
                with self.subTest(steps=steps, shift=shift):
                    scheduler = MiniMaxH3Scheduler(shift=shift)
                    scheduler.set_timesteps(schedule.schedule_points(steps), device="cpu")
                    # The vLLM-Omni H3 uniform-schedule formula includes both endpoints.
                    grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
                    sigmas = shift * grid / (1 + (shift - 1) * grid)
                    torch.testing.assert_close(scheduler.sigmas, sigmas, rtol=0, atol=0)
                    torch.testing.assert_close(scheduler.timesteps, 1 - sigmas[:-1], rtol=0, atol=0)
                    for dtype in (torch.float32, torch.bfloat16):
                        scheduler.set_timesteps(schedule.schedule_points(steps), device="cpu")
                        state = torch.tensor([0.25, -0.5], dtype=dtype)
                        velocity = torch.tensor([0.5, -1], dtype=dtype)
                        for index, timestep in enumerate(scheduler.timesteps):
                            denoised = state + (1 - timestep.to(dtype)) * velocity
                            ratio = sigmas[index + 1] / sigmas[index]
                            expected = (ratio * state.float() + (1 - ratio) * denoised.float()).to(dtype)
                            state = scheduler.step(velocity, timestep, state).prev_sample
                            torch.testing.assert_close(state, expected, rtol=0, atol=0)
                        self.assertEqual(scheduler.step_index, steps)

    def test_adaln_reads_actual_steps_from_metadata_independent_of_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "arbitrary-name.safetensors"
            for metadata in ({"num_steps": "21", "num_eval_steps": "20"}, {"num_steps": "21"}):
                save_file({}, cache, metadata=metadata)
                self.assertEqual(schedule.schedule_points(20, cache), 21)
                with self.assertRaisesRegex(ValueError, "evaluation count"):
                    schedule.schedule_points(21, cache)
            save_file({}, cache, metadata={"num_steps": "21", "num_eval_steps": "21"})
            with self.assertRaisesRegex(ValueError, "sigma point"):
                schedule.cache_steps(cache)
            cache.write_bytes(b"broken cache")
            with self.assertRaisesRegex(ValueError, "metadata"):
                schedule.cache_steps(cache)

    def test_online_h3_boundary_and_wan_passthrough(self):
        tree = ast.parse((ROOT / "server/video/video_server.py").read_text(encoding="utf-8"))
        names = {"_mmh3_build_infer_params", "_validate_mmh3_steps"}
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(torch=torch, Dict=Dict, Any=Any, List=List,
                         schedule_points=schedule.schedule_points, cache_steps=schedule.cache_steps)
        exec(compile(ast.Module(body=methods, type_ignores=[]), "video_server.py", "exec"), namespace)
        server = SimpleNamespace(default_config={})
        for steps in (1, 20, 50):
            args = SimpleNamespace(num_inference_steps=steps, prompt="test", width=1344, height=768, frames=124, seed=42)
            errors = []
            self.assertEqual(namespace["_validate_mmh3_steps"](server, {"num_inference_steps": steps}, errors), steps)
            self.assertFalse(errors)
            params = namespace["_mmh3_build_infer_params"](None, args)
            self.assertEqual(params["num_inference_steps"], steps + 1)
            self.assertEqual(args.num_inference_steps, steps)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "named-51step.safetensors"
            save_file({}, cache, metadata={"num_steps": "21", "num_eval_steps": "20"})
            server.default_config["adaln_cache_file"] = str(cache)
            validate = namespace["_validate_mmh3_steps"]
            errors = []
            self.assertEqual(validate(server, {}, errors), 20)
            self.assertFalse(errors)
            for value in (0, True, "20", 21, 50):
                errors = []
                validate(server, {"num_inference_steps": value}, errors)
                self.assertTrue(errors)

        # Wan's inference path forwards the argument unchanged to UniPC.
        adaptor_tree = ast.parse((ADAPTOR / "infer_tools.py").read_text(encoding="utf-8"))
        wan = next(node for node in ast.walk(adaptor_tree) if isinstance(node, ast.FunctionDef) and node.name == "infer_single")
        params = next(node.value for node in wan.body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "infer_params")
        values = {ast.literal_eval(key): value for key, value in zip(params.keys, params.values)}
        self.assertEqual(ast.unparse(values["num_inference_steps"]), "args.num_inference_steps")
        scheduler = UniPCMultistepScheduler(use_flow_sigmas=True)
        scheduler.set_timesteps(20)
        self.assertEqual(len(scheduler.timesteps), 20)


if __name__ == "__main__":
    unittest.main()
