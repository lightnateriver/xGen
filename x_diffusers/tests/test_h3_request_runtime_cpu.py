"""Shared CLI/HTTP request lifetime and input translation without model imports."""

from contextlib import contextmanager
import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from x_diffusers.adaptor.h3_runtime import (
    build_video_parameters,
    load_keyframes,
    run_video_request,
    validate_turbo_sla_schedule,
)


class RequestRuntimeTests(unittest.TestCase):
    def test_disabled_turbo_sla_needs_no_scheduler_or_accelerator_imports(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "safetensors"}:
                raise AssertionError("Disabled Turbo-SLA imported " + name)
            return original_import(name, *args, **kwargs)

        args = SimpleNamespace(h3_turbo_sla_lora=None)
        with patch("builtins.__import__", side_effect=guarded):
            validate_turbo_sla_schedule(object(), args, 51)
        self.assertEqual(vars(args), {"h3_turbo_sla_lora": None})

    def test_steps_seed_and_keyframe_order_preserve_the_request(self):
        args = SimpleNamespace(
            prompt="test",
            width=1344,
            height=768,
            frames=124,
            num_inference_steps=20,
            seed=42,
            keyframe_image="first.png",
            keyframe_last_image="last.png",
        )
        before = vars(args).copy()
        torch = Mock()
        load_image = Mock(side_effect=lambda path: "loaded:" + path)
        with patch.dict(
            sys.modules,
            {"torch": torch, "diffusers.utils": SimpleNamespace(load_image=load_image)},
        ):
            parameters = build_video_parameters(args, include_keyframes=False)
            load_image.assert_not_called()
            load_keyframes(args, parameters)
        self.assertEqual(parameters["num_inference_steps"], 21)
        self.assertEqual(parameters["num_frames"], 124)
        self.assertEqual(parameters["image"], "loaded:first.png")
        self.assertEqual(parameters["last_image"], "loaded:last.png")
        self.assertEqual(
            [call.args[0] for call in load_image.call_args_list],
            ["first.png", "last.png"],
        )
        torch.Generator.return_value.manual_seed.assert_called_once_with(42)
        self.assertEqual(vars(args), before)

    def scopes(self, events):
        def scope(name):
            @contextmanager
            def managed(*args):
                events.append("enter:" + name)
                try:
                    yield
                finally:
                    events.append("exit:" + name)

            return managed

        return {
            "x_diffusers.adaptor.h3_runtime": SimpleNamespace(
                H3_MAX_PROMPT_LENGTH=7000, base_temporal_padding=scope("padding"),
                fusion_context=scope("fusion"),
                condition_overlap_context=scope("overlap"),
            ),
            "x_diffusers.framework.transformer.minimax_h3": SimpleNamespace(
                request_conditioning=scope("conditioning")
            ),
        }

    def test_enabled_contexts_unwind_on_success_and_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                events = []
                result = object()

                def generate(**kwargs):
                    events.append("inference")
                    if failed:
                        raise RuntimeError("primary inference failure")
                    return result

                pipe = Mock(side_effect=generate)
                pipe.transformer._h3_attention_session.begin.side_effect = lambda args: (
                    events.append("begin")
                )
                args = SimpleNamespace(
                    h3_attention="veda",
                    h3_veda_predictor=__file__,
                    model="MiniMax-H3",
                    task_type="t2va",
                    dtype="bf16",
                    atten="laser",
                    sp=8,
                    num_inference_steps=50,
                    frames=124,
                    width=1344,
                    height=768,
                    save_fps=24,
                    phaa_num=2,
                    save_path="result.mp4",
                    h3_condition_overlap=True,
                    h3_cuda_precompute_conditioning=True,
                )
                parameters = {"num_frames": 124, "prompt": "test"}
                output = ["videos", "audio", "sampling_rate"]
                modules = self.scopes(events)
                contexts = modules.pop("x_diffusers.adaptor.h3_runtime")
                with patch.dict(sys.modules, modules), patch.dict(
                    run_video_request.__globals__, base_temporal_padding=contexts.base_temporal_padding,
                    fusion_context=contexts.fusion_context,
                    condition_overlap_context=contexts.condition_overlap_context,
                ):
                    if failed:
                        with self.assertRaisesRegex(
                            RuntimeError, "primary inference failure"
                        ):
                            run_video_request(pipe, args, parameters, output=output)
                    else:
                        self.assertIs(
                            run_video_request(pipe, args, parameters, output=output),
                            result,
                        )
                self.assertEqual(
                    events,
                    [
                        "begin",
                        "enter:padding",
                        "enter:fusion",
                        "enter:overlap",
                        "enter:conditioning",
                        "inference",
                        "exit:conditioning",
                        "exit:overlap",
                        "exit:fusion",
                        "exit:padding",
                    ],
                )
                pipe.assert_called_once_with(output=output, **parameters)
                self.assertIs(pipe.call_args.kwargs["output"], output)

    def test_default_request_does_not_activate_optional_contexts(self):
        events = []
        modules = self.scopes(events)
        modules.pop("x_diffusers.framework.transformer.minimax_h3")
        pipe = Mock(return_value={"videos": "result"})
        contexts = modules.pop("x_diffusers.adaptor.h3_runtime")
        with patch.dict(sys.modules, modules), patch.dict(
            run_video_request.__globals__, base_temporal_padding=contexts.base_temporal_padding,
            fusion_context=contexts.fusion_context,
            condition_overlap_context=contexts.condition_overlap_context,
        ):
            result = run_video_request(
                pipe, SimpleNamespace(), {"num_frames": 124}, output=["videos"]
            )
        self.assertEqual(result, {"videos": "result"})
        self.assertEqual(
            events, ["enter:padding", "enter:fusion", "exit:fusion", "exit:padding"]
        )
        pipe.transformer._h3_attention_session.begin.assert_not_called()


if __name__ == "__main__":
    unittest.main()
