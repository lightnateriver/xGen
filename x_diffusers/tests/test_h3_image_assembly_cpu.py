"""Check merged Image entry and component policies without accelerator imports."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
ADAPTOR = ROOT / "x_diffusers/x_diffusers/adaptor/h3_image.py"
PIPELINE = ROOT / "x_diffusers/x_diffusers/framework/pipeline/minimax_h3_image.py"


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
                    "_image_assembly.transformer.h3_image_optimizations": SimpleNamespace(
                        quantize_image_text_encoder=quantize),
                }
                namespace = {"__package__": "_image_assembly.pipeline",
                             "torch": SimpleNamespace(device=lambda value: value)}
                configure = load_function(PIPELINE, "_configure_text_encoder", namespace)
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

        for entry in ("load_pipeline", "run_image"):
            for backend, explicit, expected in (("npu", None, True), ("npu", False, False),
                                                 ("cuda", None, False), ("cuda", False, False)):
                with self.subTest(entry=entry, backend=backend, explicit=explicit):
                    args = SimpleNamespace(image_text_int8=explicit)
                    validate = Mock(side_effect=ReachedValidation)
                    torch = Mock()
                    namespace = {"device_type": lambda: backend, "torch": torch,
                                 "dist": SimpleNamespace(is_initialized=lambda: False),
                                 "validate_options": validate}
                    load_function(ADAPTOR, "resolve_image_text_options", namespace)
                    run = load_function(ADAPTOR, entry, namespace)
                    modules = {"PIL": SimpleNamespace(Image=Mock()),
                               "x_base": SimpleNamespace(attention_manager=Mock())}
                    with patch.dict(sys.modules, modules), self.assertRaises(ReachedValidation):
                        run(args)
                    self.assertIs(args.image_text_int8, expected)
                    validate.assert_called_once_with(args, 1)
                    torch.device.assert_not_called()
                    torch.get_device_module.assert_not_called()

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
