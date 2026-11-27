"""Check merged Image entry and component policies without accelerator imports."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from x_diffusers.adaptor.validation import resolve_image_text_options


ROOT = Path(__file__).resolve().parents[2]
ADAPTOR = ROOT / "x_diffusers/x_diffusers/adaptor/h3_runtime.py"
PIPELINE = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
CLI = ROOT / "example/diffusers/infer_minimax_h3_image.py"


def load_function(path, name, namespace):
    # Exercise real orchestration functions without importing optional model packages.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class ImageAssemblyTests(unittest.TestCase):
    def test_text_sharding_quantization_and_placement_order(self):
        for tp, quantized, resident, pinned, expected in (
            (1, False, False, False, ["offload"]),
            (1, True, False, True, ["quantize", "offload"]),
            (1, True, True, False, ["quantize", "resident"]),
            (1, False, True, False, ["resident"]),
            (4, False, False, False, ["shard", "resident"]),
            (4, True, False, False, ["shard", "quantize", "resident"]),
            (4, True, True, True, ["shard", "quantize", "resident"]),
        ):
            with self.subTest(tp=tp, quantized=quantized, resident=resident, pinned=pinned):
                events = []
                encoder = Mock()
                encoder.to.side_effect = lambda device: events.append("resident")
                pipe = SimpleNamespace(text_encoder=encoder)
                offload = Mock(side_effect=lambda *args, **kwargs: events.append("offload"))
                shard = Mock(side_effect=lambda *args, **kwargs: events.append("shard"))
                quantize = Mock(side_effect=lambda *args: events.append("quantize"))
                manager = Mock()
                modules = {
                    "diffusers.hooks": SimpleNamespace(apply_group_offloading=offload),
                    "x_base": SimpleNamespace(ParallelManager=manager, enable_tp=shard),
                    "x_diffusers.framework.transformer.minimax_h3_image": SimpleNamespace(
                        quantize_image_text_encoder=quantize),
                }
                namespace = {"__package__": "_image_assembly.adaptor",
                             "torch": SimpleNamespace(device=lambda value: value)}
                configure = load_function(PIPELINE, "_configure_h3_image_text_encoder", namespace)
                options = SimpleNamespace(text_encoder_tp=tp, image_text_int8=quantized,
                                          image_text_resident=resident, image_vision_sp=False)
                with patch.dict(sys.modules, modules):
                    configure(pipe, options, device="npu:0", world=tp, low_cpu_mem_usage=not pinned)
                self.assertEqual(events, expected)
                if tp > 1:
                    shard.assert_called_once_with(encoder, tp, target_layers=range(50))
                    self.assertIs(pipe._xgen_image_tp_manager, manager.return_value)
                if quantized:
                    quantize.assert_called_once_with(encoder)
                if "offload" in expected:
                    self.assertEqual(offload.call_args.kwargs["low_cpu_mem_usage"], not pinned)
                    self.assertEqual(offload.call_args.kwargs["offload_device"], "cpu")
                    self.assertEqual(offload.call_args.kwargs["onload_device"], "npu:0")
                    encoder.to.assert_not_called()
                else:
                    offload.assert_not_called()
                    encoder.to.assert_called_once_with("npu:0")

    def test_public_entries_resolve_precision_before_validation_or_device_access(self):
        class ReachedValidation(Exception):
            pass

        for entry, path in (("load_pipe", PIPELINE), ("run_image", CLI)):
            for backend, explicit, expected in (("npu", None, True), ("npu", False, False),
                                                 ("cuda", None, False), ("cuda", False, False)):
                with self.subTest(entry=entry, backend=backend, explicit=explicit):
                    args = SimpleNamespace(model="MiniMax-H3-Image", image_text_int8=explicit)
                    validate = Mock(side_effect=ReachedValidation)
                    torch = Mock(inference_mode=lambda: nullcontext())
                    namespace = {"__package__": "x_diffusers.adaptor",
                                 "device_type": lambda: backend, "torch": torch,
                                 "dist": SimpleNamespace(is_initialized=lambda: False),
                                 "validate_h3_image_request": validate}
                    namespace["resolve_image_text_options"] = resolve_image_text_options
                    run = load_function(path, entry, namespace)
                    modules = {"PIL": SimpleNamespace(Image=Mock()),
                               "x_base": SimpleNamespace(attention_manager=Mock()),
                               "x_diffusers.adaptor.h3_runtime": SimpleNamespace(generate_h3_images=Mock()),
                               "x_diffusers.adaptor.load_pipe": SimpleNamespace(configure_h3_image_attention=Mock()),
                               "x_diffusers.adaptor.validation": SimpleNamespace(
                                   resolve_image_text_options=resolve_image_text_options,
                                   validate_h3_image_options=validate, validate_h3_image_request=validate),
                               "torch": torch, "torch.distributed": namespace["dist"],
                               "x_base.device": SimpleNamespace(device_type=lambda: backend)}
                    torch.distributed = namespace["dist"]
                    with patch.dict(sys.modules, modules), self.assertRaises(ReachedValidation):
                        run(args)
                    self.assertIs(args.image_text_int8, expected)
                    validate.assert_called_once_with(args, 1)
                    torch.device.assert_not_called()
                    torch.get_device_module.assert_not_called()

    def test_image_request_clears_conditions_on_success_and_failure(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                clear = Mock()
                pipe = Mock(transformer=object())
                def invoke(**kwargs):
                    self.assertEqual(clear.call_count, 1)
                    if fails:
                        raise RuntimeError("image generation failed")
                    return "images"
                pipe.side_effect = invoke
                generate = load_function(ADAPTOR, "generate_h3_images", {})
                with patch.dict(sys.modules, {
                    "torch": SimpleNamespace(inference_mode=lambda: nullcontext()),
                    "x_diffusers.framework.transformer.minimax_h3_image":
                        SimpleNamespace(clear_image_condition_cache=clear)
                }):
                    parameters = dict(prompt="test", width=512, height=512,
                                      min_generation_area=256, generator=object(), references=["ref"])
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "image generation failed"):
                            generate(pipe, **parameters)
                    else:
                        self.assertEqual(generate(pipe, **parameters), "images")
                self.assertEqual(clear.call_count, 2)
                self.assertEqual(pipe.call_args.kwargs["image"], ["ref"])

    def image_cli_fixture(self, directory, *, rank=0, fails=False):
        args = SimpleNamespace(
            image_path_list=["first.png", "second.png"], atten="none", prompt="test",
            width=2048, height=2048, min_generation_area=4194304, seed=42,
            warmup=1, repeat=2, save_path=str(Path(directory) / "nested/result.png"),
            sp=2, text_encoder_tp=2, image_text_int8=None,
        )
        pipe = SimpleNamespace(transformer=SimpleNamespace(_xgen_image_sp_state={"sequence_length": 128}))
        image = Mock()
        generate = Mock(return_value=[image], side_effect=RuntimeError("request failed") if fails else None)
        load = Mock(return_value=pipe)
        generator = Mock()
        generator.manual_seed.return_value = object()
        accelerator = Mock()
        accelerator.max_memory_allocated.return_value = 128
        dist = SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2,
                               get_rank=lambda: rank, all_reduce=Mock(), ReduceOp=SimpleNamespace(MAX=object()))
        torch = SimpleNamespace(
            inference_mode=lambda: nullcontext(), distributed=dist, int64="int64",
            Generator=Mock(return_value=generator), get_device_module=Mock(return_value=accelerator),
            tensor=lambda value, **kwargs: SimpleNamespace(item=lambda: value),
        )
        opened = Mock(side_effect=lambda path: nullcontext(SimpleNamespace(convert=lambda mode: "rgb:" + path)))
        modules = {
            "torch": torch, "torch.distributed": dist,
            "PIL": SimpleNamespace(Image=SimpleNamespace(open=opened)),
            "x_base": SimpleNamespace(attention_manager=Mock()),
            "x_base.device": SimpleNamespace(device_type=lambda: "npu"),
            "x_diffusers.adaptor.validation": SimpleNamespace(
                resolve_image_text_options=resolve_image_text_options, validate_h3_image_request=Mock()),
            "x_diffusers.adaptor.load_pipe": SimpleNamespace(load_pipe=load, configure_h3_image_attention=Mock()),
            "x_diffusers.adaptor.h3_runtime": SimpleNamespace(generate_h3_images=generate),
        }
        run = load_function(CLI, "run_image", {"Path": Path, "json": json, "time": time})
        return SimpleNamespace(args=args, pipe=pipe, image=image, generate=generate, load=load,
                               generator=generator, torch=torch, dist=dist, opened=opened, modules=modules, run=run)

    def test_cli_owns_warmup_numbered_outputs_metrics_and_pipeline_reuse(self):
        for rank, reused in ((0, False), (0, True), (1, True)):
            with self.subTest(rank=rank, reused=reused), tempfile.TemporaryDirectory() as directory:
                fixture = self.image_cli_fixture(directory, rank=rank)
                with patch.dict(sys.modules, fixture.modules), patch("builtins.print") as printed:
                    result = fixture.run(fixture.args, pipe=fixture.pipe if reused else None)
                self.assertEqual(fixture.generate.call_count, 3)
                self.assertEqual(fixture.generator.manual_seed.call_count, 3)
                self.assertTrue(all(call.args == (42,) for call in fixture.generator.manual_seed.call_args_list))
                self.assertTrue(all(call.kwargs == {"device": "cpu"} for call in fixture.torch.Generator.call_args_list))
                self.assertEqual([call.args[0] for call in fixture.opened.call_args_list], ["first.png", "second.png"])
                for request in fixture.generate.call_args_list:
                    self.assertIs(request.args[0], fixture.pipe)
                    self.assertEqual(request.kwargs["references"], ["rgb:first.png", "rgb:second.png"])
                    self.assertIs(request.kwargs["generator"], fixture.generator.manual_seed.return_value)
                outputs = [str(Path(directory) / f"nested/result-{index:02d}.png") for index in range(2)]
                self.assertEqual([record["output"] for record in result["records"]], outputs)
                self.assertEqual(result["pipeline_reused"], reused)
                self.assertEqual(result["peak_allocated_bytes"], 128)
                self.assertEqual(result["workflow"], "edit")
                self.assertEqual(fixture.dist.all_reduce.call_count, 4)
                self.assertTrue(all(call.kwargs["op"] is fixture.dist.ReduceOp.MAX for call in fixture.dist.all_reduce.call_args_list))
                manifest = Path(fixture.args.save_path).with_suffix(".json")
                if rank == 0:
                    self.assertEqual([str(call.args[0]) for call in fixture.image.save.call_args_list], outputs)
                    self.assertEqual(json.loads(manifest.read_text()), result)
                    printed.assert_called_once_with(json.dumps(result), flush=True)
                else:
                    fixture.image.save.assert_not_called()
                    printed.assert_not_called()
                    self.assertFalse(manifest.exists())
                if reused:
                    fixture.load.assert_not_called()
                else:
                    fixture.load.assert_called_once_with(fixture.args, model="MiniMax-H3-Image")

    def test_cli_propagates_failed_requests_without_writing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.image_cli_fixture(directory, fails=True)
            with patch.dict(sys.modules, fixture.modules), patch("builtins.print") as printed:
                with self.assertRaisesRegex(RuntimeError, "request failed"):
                    fixture.run(fixture.args, pipe=fixture.pipe)
            fixture.image.save.assert_not_called()
            printed.assert_not_called()
            self.assertFalse(Path(fixture.args.save_path).with_suffix(".json").exists())

    def test_image_attention_selection_is_owned_by_loader(self):
        configure = load_function(PIPELINE, "configure_h3_image_attention", {})
        for backend, selected, available, error in (
            ("npu", "laser", True, None),
            ("npu", "none", False, None),
            ("cuda", "laser", True, ValueError),
            ("npu", "laser", False, RuntimeError),
        ):
            with self.subTest(backend=backend, selected=selected, available=available):
                manager = Mock()
                with patch.dict(sys.modules, {
                    "x_base": SimpleNamespace(attention_manager=manager),
                    "x_base.operator.attention": SimpleNamespace(LA_AVAILABLE=available),
                }):
                    if error:
                        with self.assertRaises(error):
                            configure(SimpleNamespace(atten=selected), backend)
                        manager.enable_laser_attention.assert_not_called()
                        manager.disable_laser_attention.assert_not_called()
                    else:
                        configure(SimpleNamespace(atten=selected), backend)
                        if selected == "laser":
                            manager.enable_laser_attention.assert_called_once_with()
                        else:
                            manager.disable_laser_attention.assert_called_once_with()

    def test_cli_accepts_upstream_and_local_options_together(self):
        spec = importlib.util.spec_from_file_location(
            "image_cli_assembly", ROOT / "example/diffusers/infer_minimax_h3_image.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = cli.build_parser().parse_args([
            "--pretrained_model_name_or_path", "weights", "--prompt", "test", "--save_path", "output.png",
            "--image_text_int8", "--no-image_text_int8", "--image_text_resident",
            "--image_vae_decode_batch_size", "4", "--image_vae_gather_once",
            "--image_adaln_table_file", "adaln.safetensors", "--image_pinned_cpu_offload",
        ])
        self.assertFalse(args.image_text_int8)
        self.assertTrue(args.image_text_resident)
        self.assertEqual(args.image_vae_decode_batch_size, 4)
        self.assertTrue(args.image_vae_gather_once)
        self.assertEqual(args.image_adaln_table_file, "adaln.safetensors")
        self.assertTrue(args.image_pinned_cpu_offload)


if __name__ == "__main__":
    unittest.main()
