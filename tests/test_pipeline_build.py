"""Verify real CLI/HTTP construction ordering without loading model weights."""

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SOURCE = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
definitions = [node for node in tree.body
               if getattr(node, "name", None) in ("PipelineBuildOptions", "_finish_pipeline")]
namespace = dict(__name__=__name__, dataclass=dataclass, Callable=Callable)
# Execute the real loader definitions while keeping vendor imports outside this CPU gate.
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
build = SimpleNamespace(**namespace)


def finish_pipeline(pipe, args, options, **callbacks):
    with patch.dict(build._finish_pipeline.__globals__, callbacks):
        return build._finish_pipeline(pipe, args, options)


class PipelineBuildTests(unittest.TestCase):
    def fixture(self):
        events = []
        pipe = object()
        args = SimpleNamespace(
            model="MiniMax-H3",
            lora_path_list=["lora"],
            fuse_lora=True,
            lora_scale_weight_list=[0.5],
            fsdp=None,
        )
        callbacks = {
            name: Mock(
                side_effect=lambda *a, _name=name, **kw: events.append(_name) or pipe
            )
            for name in (
                "update_lora",
                "update_pipe",
                "pipe_to_device",
                "update_scheduler",
            )
        }
        return pipe, args, events, callbacks

    def test_cli_build_loads_lora_before_wrapping_and_places_last(self):
        pipe, args, events, callbacks = self.fixture()
        result = finish_pipeline(
            pipe,
            args,
            build.PipelineBuildOptions(initial_lora_paths=["old"]),
            **callbacks,
        )
        self.assertIs(result, pipe)
        self.assertEqual(events, ["update_lora", "update_pipe", "pipe_to_device"])
        callbacks["update_lora"].assert_called_once_with(
            pipe, "MiniMax-H3", ["old"], ["lora"], fuse_lora=True, weights_list=[0.5]
        )

    def test_image_build_configures_scheduler_before_lora(self):
        pipe, args, events, callbacks = self.fixture()
        finish_pipeline(
            pipe,
            args,
            build.PipelineBuildOptions(configure_scheduler=True),
            **callbacks,
        )
        self.assertEqual(
            events, ["update_scheduler", "update_lora", "update_pipe", "pipe_to_device"]
        )

    def test_video_service_retains_setup_before_unfused_lora(self):
        pipe, args, events, callbacks = self.fixture()
        options = build.PipelineBuildOptions(
            lora_after_setup=True,
            fsdp_for_lora="text_encoder",
            fuse_lora=False,
            lora_transformer_list=["transformer"],
        )
        finish_pipeline(pipe, args, options, **callbacks)
        self.assertEqual(events, ["update_pipe", "update_lora", "pipe_to_device"])
        self.assertEqual(args.fsdp, "text_encoder")
        callbacks["update_lora"].assert_called_once_with(
            pipe,
            "MiniMax-H3",
            None,
            ["lora"],
            fuse_lora=False,
            weights_list=[0.5],
            lora_transformer_list=["transformer"],
        )

    def test_service_without_lora_keeps_fsdp_and_skips_overlay(self):
        pipe, args, events, callbacks = self.fixture()
        args.lora_path_list = None
        finish_pipeline(
            pipe,
            args,
            build.PipelineBuildOptions(
                lora_after_setup=True, fsdp_for_lora="text_encoder"
            ),
            **callbacks,
        )
        self.assertEqual(events, ["update_pipe", "pipe_to_device"])
        self.assertIsNone(args.fsdp)

    def test_setup_failure_stops_placement_and_later_lora(self):
        pipe, args, events, callbacks = self.fixture()
        callbacks["update_pipe"].side_effect = RuntimeError("setup failed")
        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            finish_pipeline(
                pipe,
                args,
                build.PipelineBuildOptions(lora_after_setup=True),
                **callbacks,
            )
        callbacks["update_lora"].assert_not_called()
        callbacks["pipe_to_device"].assert_not_called()

    def test_auxiliary_pipelines_are_prepared_before_main_device_placement(self):
        pipe, args, events, callbacks = self.fixture()
        options = build.PipelineBuildOptions(
            before_placement=lambda: events.append("auxiliary")
        )
        finish_pipeline(pipe, args, options, **callbacks)
        self.assertEqual(
            events, ["update_lora", "update_pipe", "auxiliary", "pipe_to_device"]
        )

    def test_public_loader_builds_components_then_finishes_once(self):
        path = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_pipe"
        )
        pipe, args, events, callbacks = self.fixture()
        namespace = dict(
            __package__="x_diffusers.adaptor",
            initialize_framework=lambda: events.append("register"),
            **callbacks,
            _load_pipe_components=Mock(
                side_effect=lambda value: events.append("components") or pipe
            ),
            _finish_pipeline=lambda pipe, args, options: finish_pipeline(
                pipe, args, options, **callbacks),
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        self.assertIs(namespace["load_pipe"](args), pipe)
        self.assertEqual(
            events, ["register", "components", "update_lora", "update_pipe", "pipe_to_device"]
        )

    def test_public_image_loader_uses_local_builder_after_option_validation(self):
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "load_pipe")
        events = []
        pipe, args, _, _ = self.fixture()
        args.model = "MiniMax-H3-Image"
        load_image = Mock(side_effect=lambda *a, **kw: events.append("image") or pipe)
        configure_attention = Mock(side_effect=lambda *a: events.append("attention"))
        components, finish = Mock(), Mock()
        namespace = dict(
            __package__="x_diffusers.adaptor", device_type=lambda: "npu",
            initialize_framework=lambda: events.append("register"),
            dist=SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 4),
            torch=SimpleNamespace(device=lambda backend, rank: (backend, rank),
                                  get_device_module=lambda backend: SimpleNamespace(current_device=lambda: 0)),
            load_minimax_h3_image_pipe=load_image,
            configure_h3_image_attention=configure_attention,
            _load_pipe_components=components, _finish_pipeline=finish,
        )
        config = SimpleNamespace(
            resolve_image_text_options=lambda *a: events.append("resolve"),
            validate_h3_image_options=lambda *a: events.append("validate"),
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
        with patch.dict(sys.modules, {"x_diffusers.adaptor.validation": config}):
            self.assertIs(namespace["load_pipe"](args), pipe)
        self.assertEqual(events, ["register", "resolve", "validate", "attention", "image"])
        configure_attention.assert_called_once_with(args, "npu")
        load_image.assert_called_once_with(args, device=("npu", 0), world=4)
        components.assert_not_called()
        finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
