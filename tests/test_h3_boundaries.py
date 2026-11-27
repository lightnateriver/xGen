"""H3 policy/mechanism integration with fake devices and real application code."""
from importlib import import_module

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

settings = import_module("x_diffusers.framework.h3_config")
schedule = import_module("x_diffusers.adaptor.validation")
refiner = import_module("x_diffusers.adaptor.validation")



from x_diffusers.adaptor import h3_runtime as _refiner_runtime
from x_diffusers.adaptor import validation as _parameter_validation

class ModelBoundaryTests(unittest.TestCase):
    def test_model_sigma_points_and_application_cache_matching(self):
        for steps in (1, 4, 8, 20, 50):
            self.assertEqual(settings.schedule_points(steps), steps + 1)
            self.assertEqual(schedule.h3_schedule_points(steps), steps + 1)
        with patch.object(schedule, "h3_adaln_cache_steps", return_value=20):
            self.assertEqual(schedule.h3_schedule_points(20, "cache"), 21)
            with self.assertRaisesRegex(ValueError, "evaluation count"):
                schedule.h3_schedule_points(21, "cache")
        for invalid in (True, 0, -1, 2.5, "20"):
            with self.assertRaises(ValueError):
                settings.schedule_points(invalid)

    def test_deployment_duration_choices_are_separate_from_model_geometry(self):
        self.assertEqual(settings.refiner_shape(6)["output_frames"], 144)
        with self.assertRaises(ValueError):
            refiner.h3_refiner_profile(6)
        for seconds, generation, latent in (
            (5, 124, 37),
            (10, 243, 72),
            (15, 362, 107),
        ):
            result = refiner.h3_refiner_profile(seconds)
            self.assertEqual(
                (result["generation_frames"], result["latent_frames"]),
                (generation, latent),
            )

    def test_long_video_padding_restores_instance_after_failure(self):
        class Pipe:
            fps = 24
            max_duration = 15

        pipe, other = Pipe(), Pipe()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with _refiner_runtime.base_temporal_padding(pipe, 362):
                self.assertGreater(pipe.max_duration, 15)
                self.assertEqual(other.max_duration, 15)
                self.assertTrue(pipe._h3_base_long_video)
                raise RuntimeError("inference failed")
        self.assertIs(type(pipe), Pipe)
        self.assertFalse(hasattr(pipe, "_h3_base_long_video"))
        with _refiner_runtime.base_temporal_padding(pipe, 124):
            self.assertIs(type(pipe), Pipe)

    def test_attention_settings_are_immutable_and_do_not_contain_startup_namespace(
        self,
    ):
        config = settings.H3AttentionConfig("veda", "predictor", False, 0.7)
        self.assertEqual(config.key, ("veda", "predictor", False, 0.7))
        with self.assertRaises(FrozenInstanceError):
            config.backend = "spark"
        self.assertFalse(hasattr(config, "model"))


class VaeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.dist = SimpleNamespace(is_initialized=lambda: False)
        self.torch = SimpleNamespace(
            distributed=self.dist, compile=Mock(side_effect=lambda fn, **kw: fn)
        )
        self.devices = patch.dict(
            sys.modules, {"torch": self.torch, "torch.distributed": self.dist}
        )
        self.devices.start()
        self.addCleanup(self.devices.stop)
        path = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        configure = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "configure_h3_vae_backend"
        )
        namespace = {"__package__": "x_diffusers.adaptor", "logger": Mock()}
        exec(compile(ast.Module(body=[configure], type_ignores=[]), str(path), "exec"), namespace)
        self.configure_vae = namespace["configure_h3_vae_backend"]
        self.compiler = import_module("x_diffusers.framework.vae.minimax_h3")
        self.tile = import_module("x_diffusers.framework.vae.minimax_h3")
        compiler_torch = patch.dict(self.compiler.__dict__, torch=self.torch)
        compiler_torch.start()
        self.addCleanup(compiler_torch.stop)

    def options(self, **overrides):
        values = dict(
            model="MiniMax-H3",
            h3_vae_backend="legacy",
            h3_vae_model_path=None,
            h3_vae_trust_remote_code=False,
            vae_lightning="none",
            sp=8,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_legacy_and_lightning_do_not_load_native_decoder(self):
        factory = Mock(
            side_effect=AssertionError("native decoder should stay unloaded")
        )
        with patch.dict(
            sys.modules,
            {
                "x_diffusers.framework.vae.minimax_h3": SimpleNamespace(
                    OmniH3Decoder=factory
                )
            },
        ):
            for backend in ("cuda", "npu"):
                for selected in ("legacy", "lightning"):
                    args = self.options(h3_vae_backend=selected)
                    pipe = SimpleNamespace()
                    _parameter_validation.validate_h3_vae(args, backend)
                    self.configure_vae(pipe, args, backend)
                    self.assertFalse(hasattr(pipe, "_h3_vae_decode_backend"))
                    self.assertEqual(
                        args.vae_lightning,
                        "decoder" if selected == "lightning" else "none",
                    )
        factory.assert_not_called()

    def test_native_decoder_is_reused_and_reconfiguration_requires_restart(self):
        args = self.options(
            h3_vae_backend="native-stream",
            h3_vae_model_path="weights/vae",
            h3_vae_trust_remote_code=True,
            h3_vae_torch_compile=True,
        )
        installed = SimpleNamespace(
            path=args.h3_vae_model_path, torch_compile_enabled=True
        )
        factory = Mock(return_value=installed)
        with patch.dict(
            sys.modules,
            {
                "x_diffusers.framework.vae.minimax_h3": SimpleNamespace(
                    OmniH3Decoder=factory
                )
            },
        ):
            _parameter_validation.validate_h3_vae(args, "cuda")
            pipe = SimpleNamespace()
            self.configure_vae(pipe, args, "cuda:0")
            self.configure_vae(pipe, args, "cuda:0")
            factory.assert_called_once_with(
                "weights/vae",
                "cuda:0",
                trust_remote_code=True,
                fp16_linears=True,
                torch_compile=True,
            )
            args.h3_vae_model_path = "other"
            with self.assertRaisesRegex(ValueError, "Restart"):
                self.configure_vae(pipe, args, "cuda:0")

    def test_compiler_validation_precedes_mechanism_and_wrapping_is_idempotent(self):
        args = self.options(h3_vae_backend="native-stream", h3_vae_torch_compile=True)
        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            _parameter_validation.validate_h3_vae_compile(args, "npu")
        self.torch.compile.assert_not_called()
        decoder = SimpleNamespace(forward=lambda value: value + 1)
        self.compiler.compile_decoder_tile_subgraph(decoder)
        self.compiler.compile_decoder_tile_subgraph(decoder)
        self.assertEqual(decoder.forward(3), 4)
        self.torch.compile.assert_called_once()

    def test_native_decoder_rejects_partial_distributed_world(self):
        args = self.options(
            h3_vae_backend="native-stream",
            h3_vae_model_path="weights/vae",
            h3_vae_trust_remote_code=True,
        )
        self.dist.is_initialized = lambda: True
        self.dist.get_world_size = lambda: 4
        with self.assertRaisesRegex(ValueError, "complete SP world"):
            _parameter_validation.validate_h3_vae(args, "npu")


def load_legacy_codec_function(name, namespace):
    """Execute the real routing code without importing accelerator patch roots."""
    path = ROOT / "x_diffusers/x_diffusers/framework/vae/minimax_h3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    function.returns = None
    for argument in ast.walk(function.args):
        if isinstance(argument, ast.arg):
            argument.annotation = None
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class LegacyCodecTests(unittest.TestCase):
    def test_default_video_decode_uses_existing_decoder(self):
        original = Mock(return_value=object())
        call = load_legacy_codec_function(
            "_decode_long_cuda_video", {"_original_video_decode_step_call": original}
        )
        block, pipe, state = object(), SimpleNamespace(), object()
        self.assertIs(call(block, pipe, state), original.return_value)
        original.assert_called_once_with(block, pipe, state)

    def test_legacy_and_full_sp_keep_original_audio_video_order(self):
        for lightning, distributed, mode in (
            ("none", False, "vae_mgr"),
            ("none", True, "chunks"),
            ("decoder", True, "full"),
        ):
            original = Mock(return_value=object())
            namespace = {
                "_original_decode_step_call": original,
                "infer_info": SimpleNamespace(vae_lightning=lightning),
                "_SP_MODE": mode,
                "torch": SimpleNamespace(
                    distributed=SimpleNamespace(is_initialized=lambda: distributed)
                ),
            }
            call = load_legacy_codec_function("_decode_step_call_route_b", namespace)
            block, pipe, state = object(), SimpleNamespace(), object()
            self.assertIs(call(block, pipe, state), original.return_value)
            original.assert_called_once_with(block, pipe, state)

    def test_native_decoder_routes_before_legacy_controls(self):
        backend = SimpleNamespace(
            decode_step=Mock(return_value=object()),
            decode_video=Mock(return_value=object()),
        )
        pipe = SimpleNamespace(_h3_vae_decode_backend=backend)
        for function, target in (
            ("_decode_step_call_route_b", backend.decode_step),
            ("_decode_long_cuda_video", backend.decode_video),
        ):
            call = load_legacy_codec_function(function, {})
            block, state = object(), object()
            self.assertIs(call(block, pipe, state), target.return_value)
            target.assert_called_once_with(block, pipe, state)

    def test_single_rank_sp_falls_back_without_collectives(self):
        for function, original_name, arguments in (
            (
                "_sp_block_forward",
                "_original_block_forward",
                (object(), object(), object()),
            ),
            ("_sp_decoder_forward", "_original_decoder_forward", (object(), object())),
        ):
            original = Mock(return_value=object())
            call = load_legacy_codec_function(
                function, {original_name: original, "_CURRENT_SP_MANAGER": None}
            )
            self.assertIs(call(*arguments), original.return_value)
            original.assert_called_once_with(*arguments)


class RefinerMediaTests(unittest.TestCase):
    def test_reference_assets_are_loaded_then_classified_then_selected(self):
        adaptor = import_module("x_diffusers.adaptor.h3_runtime")
        for conditioning in ("video", "both"):
            events = []
            refs = ["image", "audio"]
            paths = ["image.png", "audio.wav"]
            pipe = object()
            media = SimpleNamespace(
                load_references=Mock(
                    side_effect=lambda value: events.append("load") or refs
                ),
                reference_conditioning=Mock(
                    side_effect=lambda value: events.append("classify") or conditioning
                ),
                select_adaln=Mock(side_effect=lambda *a: events.append("select")),
            )
            args = SimpleNamespace(task_type="ref2va", reference_paths=paths)
            with (patch.object(adaptor, "load_h3_references", media.load_references),
                  patch.object(adaptor, "select_h3_adaln", media.select_adaln),
                  patch.object(_parameter_validation, "h3_reference_conditioning", media.reference_conditioning)):
                parameters, audio = adaptor.prepare_h3_draft_media(pipe, args)
            self.assertIs(parameters["references"], refs)
            self.assertEqual(audio, conditioning == "both")
            self.assertEqual(events, ["load", "classify", "select"])
            media.load_references.assert_called_once_with(paths)
            media.select_adaln.assert_called_once_with(pipe, args, conditioning)

    def test_first_frame_and_text_paths_keep_media_order_and_skip_unused_loads(self):
        adaptor = import_module("x_diffusers.adaptor.h3_runtime")
        image = object()
        load = Mock(return_value=image)
        with patch.dict(
            sys.modules, {"diffusers.utils": SimpleNamespace(load_image=load)}
        ):
            parameters, audio = adaptor.prepare_h3_draft_media(
                object(), SimpleNamespace(task_type="fl2va", keyframe_image="first.png")
            )
            self.assertIs(parameters["image"], image)
            self.assertFalse(audio)
            self.assertEqual(
                adaptor.prepare_h3_draft_media(
                    object(), SimpleNamespace(task_type="t2va")
                ),
                ({}, False),
            )
        load.assert_called_once_with("first.png")


class VaeTileInstallerTests(unittest.TestCase):
    def test_parallel_selection_keeps_native_fallback_and_checks_heads(self):
        path = ROOT / "x_diffusers/x_diffusers/framework/vae/minimax_h3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = [node for node in tree.body
                       if isinstance(node, ast.FunctionDef)
                       and node.name in ("install_vae_parallel", "decode_clip_tiles")]
        namespace = dict(MethodType=MethodType)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        install = namespace["install_vae_parallel"]
        manager = SimpleNamespace(sp_size=4)
        for heads, tiles in ((16, False), (16, True), (15, True)):
            with self.subTest(heads=heads, tiles=tiles):
                decode = Mock(return_value="native")
                vae = SimpleNamespace(
                    decoder=SimpleNamespace(transformer_blocks=[SimpleNamespace(
                        attn=SimpleNamespace(heads=heads))]),
                    _decode_clip=decode, use_tiling=False,
                )
                if heads == 15:
                    with self.assertRaisesRegex(ValueError, "divide evenly"):
                        install(vae, manager, tile_parallel=tiles)
                    self.assertIs(vae._decode_clip, decode)
                    self.assertFalse(hasattr(vae, "_h3_cuda_vae_parallel"))
                else:
                    install(vae, manager, tile_parallel=tiles)
                    self.assertIs(vae._h3_cuda_vae_parallel, manager)
                    self.assertEqual(vae._decode_clip("latent"), "native")
                    decode.assert_called_once_with("latent")
                    if tiles:
                        self.assertIs(vae._decode_clip.__self__, vae)
                    else:
                        self.assertIs(vae._decode_clip, decode)


if __name__ == "__main__":
    unittest.main()
