"""Refiner delivery, shared media and explicit model registration regression tests."""
import ast
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x_diffusers"))
from x_diffusers import framework
from x_diffusers.adaptor import h3_runtime as runtime, validation
from x_diffusers.framework.h3_config import H3RefinerConfig, H3RefinerRequest, H3RefinerResult, refiner_shape


class PackageInitializationTests(unittest.TestCase):
    def test_import_and_argument_parsing_do_not_import_model_runtime(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "x_diffusers"))
        result = subprocess.run([sys.executable, "-c", """
import sys
import x_diffusers
from x_diffusers import parse_args
args = parse_args(argv=['--seed', '123'], bootstrap_environment=False)
assert args.seed == 123
assert not any(name.split('.')[0] in {'torch', 'diffusers', 'x_base', 'transformers'} for name in sys.modules)
assert not any(name.startswith('x_diffusers.framework.pipeline.') for name in sys.modules)
"""], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_registration_keeps_order_and_runs_once(self):
        with patch.object(framework, "_initialized", False), patch.object(framework, "import_module") as register:
            framework.initialize_framework()
            framework.initialize_framework()
        self.assertEqual([call.args[0] for call in register.call_args_list], list(framework._REGISTRATION_MODULES))
        modules = list(framework._REGISTRATION_MODULES)
        self.assertLess(modules.index(".transformer.attention_backend"), modules.index(".transformer.minimax_h3"))
        self.assertLess(modules.index(".vae.wan"), modules.index(".pipeline.pipeline"))
        self.assertLess(modules.index(".pipeline.pipeline"), modules.index(".pipeline.qwenimage"))

    def test_failed_registration_is_not_marked_ready(self):
        with patch.object(framework, "_initialized", False), patch.object(framework, "import_module", side_effect=ImportError("missing runtime")):
            with self.assertRaises(ImportError):
                framework.initialize_framework()
            self.assertFalse(framework._initialized)


class RefinerBoundaryTests(unittest.TestCase):
    def test_worker_scope_is_opt_in_and_covers_caller_setup(self):
        prepare = Mock()
        guard = Mock()
        guard.__enter__ = Mock(return_value=guard)
        guard.__exit__ = Mock(return_value=False)
        manager = SimpleNamespace(_h3_refiner_failure_guard=guard)
        with patch.dict(sys.modules, {"x_diffusers.adaptor.load_pipe": SimpleNamespace(prepare_h3_refiner=prepare)}):
            with runtime.refiner_worker_scope(manager, SimpleNamespace(h3_refiner=False)):
                pass
            prepare.assert_not_called()
            args = SimpleNamespace(h3_refiner=True)
            with self.assertRaisesRegex(OSError, "output setup"), runtime.refiner_worker_scope(manager, args):
                raise OSError("output setup")
        prepare.assert_called_once_with(manager, args)
        self.assertIs(guard.__exit__.call_args.args[0], OSError)

    def test_cli_output_directory_error_is_guarded_and_does_not_return_success(self):
        from x_diffusers.adaptor.config import parse_args

        source = ROOT / "x_diffusers/x_diffusers/adaptor/infer_tools.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InferenceManager")
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "infer")
        guard = Mock()
        guard.__enter__ = Mock(return_value=guard)
        guard.__exit__ = Mock(return_value=False)
        namespace = dict(refiner_worker_scope=Mock(return_value=guard), deepcopy=deepcopy, infer_info=Mock(),
                         logger=Mock(), os=SimpleNamespace(makedirs=Mock(side_effect=PermissionError("output setup")),
                                                         path=os.path), torch=Mock(), time=time)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        manager = SimpleNamespace(init_args=None, _prepare_pipeline=Mock(), infer_single=Mock())
        args = parse_args(argv=["--model", "MiniMax-H3", "--task_type", "t2va", "--sp", "8", "--h3_refiner"], bootstrap_environment=False)
        with self.assertRaisesRegex(PermissionError, "output setup"):
            namespace["infer"](manager, args)
        self.assertIs(guard.__exit__.call_args.args[0], PermissionError)
        manager.infer_single.assert_not_called()
        namespace["torch"].distributed.barrier.assert_not_called()

    def test_http_setup_is_guarded_but_publication_failure_is_not_a_worker_failure(self):
        from typing import Any, Dict

        source = ROOT / "server/video/video_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VideoGenerationService")
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_process_task")
        for fail_setup in (True, False):
            with self.subTest(fail_setup=fail_setup):
                guard = Mock()
                guard.__enter__ = Mock(return_value=guard)
                guard.__exit__ = Mock(return_value=False)
                manager = SimpleNamespace(initialize_pipe=Mock(), infer_single=Mock(return_value={}),
                                          sr_pipe=SimpleNamespace(failed=False))
                events = []
                health = SimpleNamespace(model_loaded=False, phase="generating")

                def advance(phase, **unused):
                    events.append(phase)
                    health.phase = phase
                    return health

                health.advance = advance
                service = SimpleNamespace(
                    inference_manager=manager, maas_health=health, maas=False,
                    _build_args=Mock(return_value=SimpleNamespace(h3_refiner=True)), _set_guidance_scale=Mock(),
                    _publish_maas_task=Mock(side_effect=OSError("publish failed")),
                    tasks_lock=nullcontext(), _fail_maas_jobs=Mock(), default_config={"h3_refiner": True},
                )
                task = SimpleNamespace(task_id="id", finish=Mock())
                make_directory = Mock(side_effect=PermissionError("setup failed") if fail_setup else None)
                namespace = dict(Dict=Dict, Any=Any,
                                 refiner_worker_scope=Mock(side_effect=lambda *a: events.append("prepare") or guard),
                                 infer_info=Mock(), os=SimpleNamespace(makedirs=make_directory, path=os.path),
                                 torch=Mock(), time=time, logger=Mock(), StorageError=OSError,
                                 _cleanup_temp_files=Mock())
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
                result = namespace["_process_task"](service, "id", {}, "output/video.mp4", maas_task=task)
                self.assertFalse(result)
                self.assertLess(events.index("loading"), events.index("prepare"))
                if fail_setup:
                    self.assertIs(guard.__exit__.call_args.args[0], PermissionError)
                    manager.infer_single.assert_not_called()
                    service._fail_maas_jobs.assert_called_once()
                else:
                    self.assertIsNone(guard.__exit__.call_args.args[0])
                    manager.infer_single.assert_called_once()
                    service._fail_maas_jobs.assert_not_called()
                    self.assertEqual(task.finish.call_args.kwargs["code"], "storage_failed")

    def test_writer_crops_frames_and_commits_only_complete_outputs(self):
        class Chunk:
            def __init__(self, frames):
                self.shape = (1, 3, frames, 2, 2)

            def __getitem__(self, key):
                return Chunk(min(self.shape[2], key[2].stop))

        for failed in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "video.mp4"
                output.write_bytes(b"existing output")
                metrics = SimpleNamespace(span=lambda *a, **kw: nullcontext(), host_span=lambda *a: nullcontext())
                writer = runtime.RefinerVideoWriter(output, metrics)
                work = Path(writer.work.name)
                stream = Mock()

                def encode():
                    if failed:
                        raise RuntimeError("encoder failed")
                    writer.staged.write_bytes(b"complete output")

                stream.close.side_effect = encode
                pixels = Mock()
                pixels.cpu.return_value.numpy.return_value = ["frame1", "frame2"]
                convert = Mock(return_value=pixels)
                modules = {
                    "x_base.vae_parallelism.audio_utils": SimpleNamespace(save_audio_to_wav=Mock()),
                    "x_base.vae_parallelism.save_av_stream": SimpleNamespace(SaveAVStream=Mock(return_value=stream)),
                    "x_diffusers.framework.vae.minimax_h3": SimpleNamespace(refiner_rgb_frames=convert),
                }
                with patch.dict(sys.modules, modules):
                    writer.start(None, 32000, 2)
                    writer.enqueue(Chunk(5))
                    self.assertEqual(convert.call_args.args[0].shape[2], 2)
                    self.assertEqual(writer.sent, 2)
                    try:
                        if failed:
                            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                                writer.finish()
                            self.assertEqual(output.read_bytes(), b"existing output")
                        else:
                            writer.finish()
                            self.assertEqual(output.read_bytes(), b"complete output")
                    finally:
                        writer.close()
                self.assertFalse(work.exists())
                self.assertEqual(stream.abort.call_count, int(failed))

    def test_config_and_request_are_explicit_and_immutable(self):
        config = H3RefinerConfig("weights", "prepared", "cuda", "prefetch", 20)
        request = H3RefinerRequest("prompt", 42, 5)
        with self.assertRaises(FrozenInstanceError):
            config.loading = "sync"
        with self.assertRaises(FrozenInstanceError):
            request.seed = 1
        self.assertFalse(hasattr(config, "args"))
        self.assertFalse(hasattr(request, "save_path"))

    def test_model_compute_order_and_context_restoration_on_success_and_failure(self):
        path = ROOT / "x_diffusers/x_diffusers/framework/pipeline/minimax_h3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "H3RefinerPipeline")
        call = next(node for node in cls.body if getattr(node, "name", "") == "__call__")
        namespace = dict(
            time=time, torch=Mock(), refiner_shape=refiner_shape, H3RefinerRequest=H3RefinerRequest,
            H3RefinerResult=H3RefinerResult, H3RefinerOutput=object, logger=logging.getLogger("refiner_test"),
        )
        exec(compile(ast.Module(body=[call], type_ignores=[]), str(path), "exec"), namespace)
        for failed in (False, True):
            events = []
            rope = SimpleNamespace(is_rope_fused=True)
            phaa = SimpleNamespace(get_phaa_split_num=lambda: 4, set_phaa_split_num=Mock())
            owner = SimpleNamespace(
                failed=False, rank=0, base_steps=20, completed=0, startup_seconds=1.0,
                accelerator=Mock(), metrics=Mock(), base_weights=Mock(),
                identity={"resident": 1}, identities=lambda: {"resident": 1},
            )
            owner.metrics.resolve.return_value = []
            output = object()

            def draft(*unused):
                events.append("base")
                owner.calls["h3"] = 20
                return "latent", "audio"

            def conditions(*unused):
                events.append("conditioning")
                return "conditions"

            def refine(*unused):
                events.append("refiner")
                if failed:
                    raise RuntimeError("refiner failed")
                owner.calls["refiner"] = 3
                return "final"

            def decode(request, spec, final, audio, sink):
                events.append("decode")
                self.assertIs(sink, output)
                return None, "waveform"

            owner._draft, owner._conditions, owner._refine, owner._decode = draft, conditions, refine, decode
            invoke = MethodType(namespace["__call__"], owner)
            with patch.dict(sys.modules, {"x_base": SimpleNamespace(rope_manager=rope), "x_base.phaa": phaa}):
                if failed:
                    with self.assertLogs("refiner_test", level="ERROR"), self.assertRaisesRegex(RuntimeError, "refiner failed"):
                        invoke(H3RefinerRequest("prompt", 42, 5), output=output)
                    self.assertTrue(owner.failed)
                else:
                    result = invoke(H3RefinerRequest("prompt", 42, 5), output=output)
                    self.assertEqual(result.audio, "waveform")
                    self.assertEqual(result.metrics["actual_dit"], {"h3": 20, "refiner": 3})
            self.assertEqual(events, ["base", "conditioning", "refiner"] + ([] if failed else ["decode"]))
            self.assertTrue(rope.is_rope_fused)
            phaa.set_phaa_split_num.assert_called_with(4)
            self.assertFalse(owner.metrics.active)

    def test_application_owns_paths_metrics_and_writer_cleanup(self):
        spec = importlib.util.spec_from_file_location("boundary_files", ROOT / "x_base/x_base/utils/file.py")
        files = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(files)
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(prompt="prompt", seed=42, h3_refiner_seconds=5, phaa_num=2, enable_audio=True,
                                       save_path=str(Path(directory) / "output.mp4"))
                inference = SimpleNamespace(vae_lightning="decoder", update_info=Mock())
                writer = Mock()
                refiner = Mock(metrics=object())
                refiner.return_value = H3RefinerResult(metrics={"actual_dit": {"h3": 20, "refiner": 3}})
                if failed:
                    refiner.side_effect = RuntimeError("writer failed")
                guard = Mock()
                guard.__enter__ = Mock(return_value=guard)
                guard.__exit__ = Mock(return_value=False)
                manager = SimpleNamespace(pipe=SimpleNamespace(transformer=object()), sr_pipe=refiner,
                                          _h3_refiner_failure_guard=guard)
                dist = SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: 0, barrier=Mock())
                modules = {
                    "x_diffusers.adaptor.load_pipe": SimpleNamespace(prepare_h3_refiner=Mock()),
                    "x_base": SimpleNamespace(infer_info=inference),
                    "x_base.utils.file": files, "torch": SimpleNamespace(distributed=dist), "torch.distributed": dist,
                }
                with (patch.dict(sys.modules, modules),
                      patch.object(validation, "validate_h3_refiner_settings"),
                      patch.object(validation, "validate_h3_refiner_inputs"),
                      patch.object(runtime, "prepare_h3_draft_media", return_value=({"references": ["loaded"]}, True)),
                      patch.object(runtime, "RefinerVideoWriter", return_value=writer),
                      patch.object(runtime, "begin_h3_attention_request"),
                      patch.object(runtime, "fusion_context", return_value=nullcontext())):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, "writer failed"):
                            runtime.infer_h3_refiner(manager, args)
                        self.assertTrue(refiner.failed)
                        guard.fail.assert_called_once()
                    else:
                        result = runtime.infer_h3_refiner(manager, args)
                        self.assertEqual(result["saved_path"], str(Path(args.save_path).resolve()))
                        record = json.loads(Path(args.save_path).with_suffix(".metrics.rank0.json").read_text())
                        self.assertEqual(record, result["h3_refiner_metrics"])
                        guard.fail.assert_not_called()
                request = refiner.call_args.args[0]
                self.assertIsInstance(request, H3RefinerRequest)
                self.assertTrue(request.reference_audio)
                self.assertEqual(request.media, {"references": ["loaded"]})
                self.assertFalse(hasattr(request, "save_path"))
                writer.close.assert_called_once_with()
                guard.__enter__.assert_called_once()
                guard.__exit__.assert_called_once()
                dist.barrier.assert_not_called()
                self.assertEqual(inference.vae_lightning, "decoder")


    def test_startup_reuses_guard_and_marks_errors_before_loading_fatal(self):
        source = ROOT / "x_diffusers/x_diffusers/adaptor/load_pipe.py"
        function = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                        if isinstance(n, ast.FunctionDef) and n.name == "prepare_h3_refiner")
        prepare = Mock(return_value="ready")
        namespace = dict(_prepare_h3_refiner=prepare)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        guard = Mock()
        guard.__enter__ = Mock(return_value=guard)
        guard.__exit__ = Mock(return_value=False)
        factory = Mock(return_value=guard)
        distributed = SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 8)
        modules = {"torch": SimpleNamespace(distributed=distributed), "torch.distributed": distributed,
                   "x_base.utils.distributed": SimpleNamespace(DistributedFailureGuard=factory)}
        manager, args = SimpleNamespace(), object()
        with patch.dict(sys.modules, modules):
            self.assertEqual(namespace["prepare_h3_refiner"](manager, args), "ready")
            prepare.side_effect = OSError("asset failure")
            with self.assertRaisesRegex(OSError, "asset failure"):
                namespace["prepare_h3_refiner"](manager, args)
        factory.assert_called_once_with()
        self.assertEqual(guard.__enter__.call_count, 2)
        self.assertTrue(manager._h3_refiner_failed)
        guard.fail.assert_called_once()

    def test_preparation_metrics_and_cleanup_errors_notify_guard(self):
        for stage in ("media", "writer_init", "metrics", "cleanup"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(prompt="test", seed=42, h3_refiner_seconds=5, phaa_num=2,
                                       save_path=str(Path(directory) / "output.mp4"))
                info = SimpleNamespace(vae_lightning="decoder", update_info=Mock())
                refiner = Mock(metrics=object(), failed=False)
                refiner.return_value = H3RefinerResult(metrics={})
                guard = Mock()
                guard.__enter__ = Mock(return_value=guard)
                guard.__exit__ = Mock(return_value=False)
                manager = SimpleNamespace(pipe=SimpleNamespace(transformer=None), sr_pipe=refiner,
                                          _h3_refiner_failure_guard=guard)
                writer, metrics = Mock(), Mock()
                distributed = SimpleNamespace(get_rank=lambda: 0, barrier=Mock())
                modules = {"torch": SimpleNamespace(distributed=distributed), "torch.distributed": distributed,
                           "x_base": SimpleNamespace(infer_info=info),
                           "x_base.utils.file": SimpleNamespace(atomic_json=metrics),
                           "x_diffusers.adaptor.load_pipe": SimpleNamespace(prepare_h3_refiner=Mock())}
                with (patch.dict(sys.modules, modules), patch.object(validation, "validate_h3_refiner_settings"),
                      patch.object(validation, "validate_h3_refiner_inputs"),
                      patch.object(runtime, "prepare_h3_draft_media", return_value=({}, False)) as media,
                      patch.object(runtime, "RefinerVideoWriter", return_value=writer) as factory,
                      patch.object(runtime, "begin_h3_attention_request")):
                    {"media": media, "writer_init": factory, "metrics": metrics, "cleanup": writer.close}[stage].side_effect = OSError(stage)
                    with self.assertRaisesRegex(OSError, stage):
                        runtime.infer_h3_refiner(manager, args)
                guard.fail.assert_called_once()
                self.assertTrue(refiner.failed)
                self.assertEqual(info.vae_lightning, "decoder")
                distributed.barrier.assert_not_called()
                guard.__exit__.assert_called_once()


class SharedMediaTests(unittest.TestCase):
    def test_order_soundtrack_and_resize_use_one_request_preparation(self):
        args = SimpleNamespace(reference_paths=["image.png", "video.mp4", "audio.wav"], enable_audio=False)
        parameters = {}
        with (patch.object(runtime, "load_h3_references", return_value=["image", "video", "audio"]) as load,
              patch.object(validation, "h3_reference_conditioning", return_value="both") as classify,
              patch.object(validation, "h3_reference_image_resize_mode", return_value="match"),
              patch.object(runtime, "select_h3_adaln") as select):
            self.assertEqual(runtime.prepare_h3_media(object(), args, parameters), "both")
        load.assert_called_once_with(args.reference_paths)
        classify.assert_called_once_with(args)
        self.assertEqual(parameters, dict(references=["image", "video", "audio"], h3_reference_image_resize_mode="match"))
        self.assertEqual(select.call_args.args[-1], "both")

    def test_text_and_keyframes_choose_none_or_video_conditioning(self):
        for parameters, expected in (({}, "none"), ({"image": object()}, "video"), ({"last_image": object()}, "video")):
            with patch.object(runtime, "select_h3_adaln") as select:
                self.assertEqual(runtime.prepare_h3_media(object(), SimpleNamespace(), parameters), expected)
                self.assertEqual(select.call_args.args[-1], expected)

    def test_npu_table_refreshes_when_source_changes_even_with_same_conditioning(self):
        pipe = SimpleNamespace(transformer=SimpleNamespace(_adaln_cache=SimpleNamespace(_conditioning_type="video", _source_file="old")))
        args = SimpleNamespace(adaln_cache_file="new.safetensors")
        install = Mock()
        with patch.dict(sys.modules, {"x_base.adaln_cache": SimpleNamespace(install_adaln_cache=install)}):
            runtime.select_h3_adaln(pipe, args, "video")
        install.assert_called_once_with(pipe, args.adaln_cache_file, conditioning_type="video")

    def test_http_and_cli_delegate_without_private_media_or_adaln_implementations(self):
        for name in ("server/video/video_server.py", "x_diffusers/x_diffusers/adaptor/infer_tools.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            self.assertFalse({node.name for node in methods} & {"_mmh3_load_references", "_mmh3_switch_adaln_cache", "_probe_video_has_audio"})
            method = next(node for node in methods if node.name == "_infer_minimax_h3")
            self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "prepare_h3_media" for node in ast.walk(method)))


class ComfyBoundaryTests(unittest.TestCase):
    def test_model_modules_have_no_application_imports(self):
        for area in ("transformer", "text_encoder", "vae"):
            path = ROOT / f"x_comfyui/x_comfyui/framework/{area}/minimax_h3.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any("adaptor" in name for name in imports), path)



if __name__ == "__main__":
    unittest.main()
