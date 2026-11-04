"""CPU tests of the actual Flask routes. Run directly in the service environment."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from pathlib import Path
import queue
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

from PIL import Image

from maas_fl2va import FIXED_FIELDS, MAX_BODY_BYTES, MAX_IMAGE_BYTES, validate_startup
import video_server
from video_server import VideoGenerationService
from x_diffusers.adaptor.h3_refiner_config import base_temporal_padding


def image_bytes(width=64, height=64, image_format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format=image_format)
    return buffer.getvalue()


def data_uri(payload, mime="png"):
    return f"data:image/{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def padded_png(size):
    png = image_bytes()
    payload = b"\0" * (size - len(png) - 12)
    chunk = b"vpAg" + payload
    return png[:-12] + struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk)) + png[-12:]


class MaaSInputTests(unittest.TestCase):
    def setUp(self):
        self.service = VideoGenerationService(maas_fl2va_input=True)
        self.service.is_initialized = True
        self.client = self.service.app.test_client()
        self.good = {"prompt": "test", "first_frame": data_uri(image_bytes())}

    def post(self, route, payload):
        body = {"id": "job-1", "input": payload} if route == "/v1/tasks" else payload
        return self.client.post(route, json=body)

    def assert_rejected(self, response, status=400, field=None):
        self.assertEqual(response.status_code, status, response.get_json())
        body = response.get_json()
        self.assertEqual(body["error"], "invalid_input")
        if field:
            self.assertEqual(body["field"], field)
        self.assertFalse(self.service.maas_pending)
        self.assertEqual(self.service.tasks, {})

    def test_numeric_and_enum_rejections_on_both_routes(self):
        invalid = {
            "duration": (None, True, "5", 5.0, 5.9, 4, 6, 10, 11, 15),
            "seed": (None, True, 1.9, "42", -1, 2**31),
            "num_inference_steps": (None, True, "51", 51.0, 0, 50, 52),
            "resolution": (None, "1080p", "2k", 768, []),
            "aspect_ratio": (None, "9:16"),
            "model": (None, "other", {}),
            "task_type": ("t2va", "ref2va"),
            "enable_audio": (None, 0, "false"),
        }
        for route in ("/generate", "/v1/tasks"):
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(route=route, field=field, value=value):
                        self.assert_rejected(self.post(route, dict(self.good, **{field: value})), field=field)

    def test_text_and_forbidden_fields_on_both_routes(self):
        for route in ("/generate", "/v1/tasks"):
            for value in (None, "", " \n", [], "a" * 2001, "a\0b", "\ud800"):
                with self.subTest(route=route, value=repr(value)[:30]):
                    self.assert_rejected(self.post(route, dict(self.good, prompt=value)), field="prompt")
            for field in ("width", "height", "frames", "save_fps", "ref_path", "last_frame", "save_path",
                          "audio_save_path", "atten", "parameters"):
                with self.subTest(route=route, field=field):
                    self.assert_rejected(self.post(route, dict(self.good, **{field: "injected"})), field="input")

    def test_bad_wire_bodies_on_both_routes(self):
        for route in ("/generate", "/v1/tasks"):
            for raw in (b"{", b"{\xff}", b"[]", b"null", b'{"input":{"seed":1,"seed":2}}',
                        b'{"seed":NaN}', b'{"seed":Infinity}', b'[' * 10000):
                with self.subTest(route=route, body=raw[:32]):
                    self.assert_rejected(self.client.post(route, data=raw, content_type="application/json"))
            self.assert_rejected(self.client.post(route, data="{}"), 415, "body")
            self.assert_rejected(self.client.post(route, data=b" " * (MAX_BODY_BYTES + 1),
                                                  content_type="application/json"), 413, "body")

    def test_invalid_envelope(self):
        for body in ({"id": "job-1"}, {"id": "job-1", "input": "text"},
                     {"id": True, "input": self.good}, {"id": "../out", "input": self.good},
                     {"id": "a" * 129, "input": self.good}, {"input": self.good},
                     {"id": "job-1", "input": self.good, "created_at": 10**30},
                     {"id": "job-1", "input": self.good, "created_at": None},
                     {"id": "job-1", "input": self.good, "created_at": 0},
                     {"id": "job-1", "input": self.good, "config": {}}):
            with self.subTest(body=str(body)[:45]):
                self.assert_rejected(self.client.post("/v1/tasks", json=body))

    def test_bad_images_before_queue(self):
        for image, status in ((None, 400), ("", 400), ("https://example.invalid/a.png", 400),
                              ("/input.png", 400), ("data:image/png;base64,@@@", 400),
                              (data_uri(image_bytes()[:40]), 400),
                              (data_uri(image_bytes(image_format="JPEG")), 400),
                              (data_uri(image_bytes(31, 32)), 400),
                              (data_uri(image_bytes(129, 32)), 400),
                              (data_uri(image_bytes(4097, 2047)), 413),
                              (data_uri(image_bytes(4096, 2049)), 413),
                              (data_uri(padded_png(MAX_IMAGE_BYTES + 1)), 413)):
            with self.subTest(image=str(image)[:28], status=status):
                self.assert_rejected(self.post("/v1/tasks", dict(self.good, first_frame=image)), status, "first_frame")

    def test_inclusive_boundaries_and_normalized_queue_data(self):
        cases = (("/v1/tasks", "中" * 1999, 0, data_uri(padded_png(MAX_IMAGE_BYTES - 1))),
                 ("/generate", "🙂" * 2000, 2**31 - 1, data_uri(padded_png(MAX_IMAGE_BYTES))),
                 ("/generate", "--valid-leading-option-text", 42, data_uri(image_bytes(4096, 2048))),
                 ("/generate", "test", 42, data_uri(image_bytes(4096, 2047))),
                 ("/generate", "test", 42, data_uri(image_bytes(128, 32, "JPEG"), "jpeg")),
                 ("/generate", "test", 42, data_uri(image_bytes(32, 32))))
        for route, prompt, seed, first_frame in cases:
            with self.subTest(route=route, prompt_length=len(prompt), seed=seed):
                response = self.post(route, dict(self.good, prompt=prompt, seed=seed, first_frame=first_frame))
                self.assertEqual(response.status_code, 200, response.get_json())
                job, _ = self.service._claim_maas_task()
                self.assertEqual(job["data"], dict(FIXED_FIELDS, prompt=prompt, seed=seed,
                                                   first_frame=first_frame, enable_audio=True))
                self.assertIn(f"--prompt={prompt}", self.service._config_to_arg_list({"prompt": prompt}))

    def test_body_limit_and_read_only_profile_config(self):
        body = json.dumps(self.good).encode("utf-8")
        for length in (MAX_BODY_BYTES - 1, MAX_BODY_BYTES):
            response = self.client.post("/generate", data=body.ljust(length), content_type="application/json")
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.post("/config", json={"model": "other"}).status_code, 405)

    def test_startup_profile_rejects_incompatible_settings(self):
        validate_startup(FIXED_FIELDS, "cuda", False)
        for settings, backend, warmup in ((FIXED_FIELDS, "npu", False), (FIXED_FIELDS, "cuda", True),
                                          (dict(FIXED_FIELDS, h3_refiner=True), "cuda", False),
                                          (dict(FIXED_FIELDS, num_inference_steps=50), "cuda", False)):
            with self.assertRaises(ValueError):
                validate_startup(settings, backend, warmup)

    def test_peer_rank_cleans_decoded_first_frame(self):
        path = Path(video_server._resolve_input_file(self.good["first_frame"]))
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertTrue(path.is_file())
        self.service.inference_manager = Mock()
        args = SimpleNamespace(model="MiniMax-H3", keyframe_image=str(path))
        with patch.object(video_server, "broadcast_int", side_effect=[video_server.CMD_EXECUTE,
                                                                    video_server.CMD_SHUTDOWN]), \
             patch.object(video_server, "broadcast_dict", return_value={"data": {}, "save_path": "video.mp4"}), \
             patch.object(video_server.dist, "barrier"), patch.object(video_server, "infer_info"), \
             patch.object(self.service, "_build_args", return_value=args):
            self.service._worker_loop()
        self.service.inference_manager.infer_single.assert_called_once_with(args)
        self.assertFalse(path.exists())

    def test_disabled_profile_keeps_legacy_parser(self):
        legacy = VideoGenerationService()
        self.assertEqual(legacy.task_queue.maxsize, 100)
        legacy.default_config.update(FIXED_FIELDS)
        legacy.is_initialized = True
        response = legacy.app.test_client().post("/generate", json=dict(self.good, seed="42"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(legacy.task_queue.get_nowait()["data"]["seed"], 42)


class H3BaseDurationTests(unittest.TestCase):
    def setUp(self):
        config = Path(__file__).resolve().parents[2] / "example/diffusers/configs/h3_npu_base.example.json"
        self.service = VideoGenerationService(str(config))
        self.service.is_initialized = True
        self.client = self.service.app.test_client()

    def test_base_duration_spec(self):
        spec = self.client.get("/params").json["parameters"]["duration"]
        self.assertEqual(spec["constraints"], {"min": 5, "max": 15})
        wan_spec = VideoGenerationService().app.test_client().get("/params").json["parameters"]["duration"]
        self.assertEqual(wan_spec["constraints"], {"allowed_values": [5, 10]})

    def test_both_base_endpoints_accept_fifteen_seconds(self):
        payload = {"prompt": "test", "task_type": "t2va", "duration": 15}
        for endpoint, request, status in (
            ("/generate", payload, "accepted"),
            ("/v1/tasks", {"id": "base-15s", "input": payload}, "pending"),
        ):
            with self.subTest(endpoint=endpoint):
                response = self.client.post(endpoint, json=request)
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["status"], status)
                task = self.service.task_queue.get_nowait()["data"]
                self.assertEqual(task["duration"], 15)
                self.assertEqual(task["frames"], 360)

    def test_base_duration_outside_range_or_fractional_is_rejected(self):
        for duration in (4, 15.5, 16):
            with self.subTest(duration=duration):
                response = self.client.post("/generate", json={"prompt": "test", "duration": duration})
                self.assertEqual(response.status_code, 400)
        response = self.client.post("/v1/tasks", json={"id": "base-16s", "input": {"prompt": "test", "duration": 16}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "failed")
        self.assertTrue(self.service.task_queue.empty())

    def test_padding_is_local_to_the_fifteen_second_request(self):
        class BasePipeline:
            fps = 24
            max_duration = 15

        pipe = BasePipeline()
        other = BasePipeline()
        with base_temporal_padding(pipe, 240):
            self.assertEqual(pipe.max_duration, 15)
        with self.assertRaises(RuntimeError):
            with base_temporal_padding(pipe, 360):
                self.assertEqual(pipe.max_duration * pipe.fps, 362)
                self.assertTrue(pipe._h3_base_long_video)
                self.assertEqual(other.max_duration, 15)
                raise RuntimeError("inference error")
        self.assertIs(type(pipe), BasePipeline)
        self.assertFalse(hasattr(pipe, "_h3_base_long_video"))

    def test_base_save_trims_padded_video_and_audio(self):
        frames = list(range(362))
        audio = video_server.np.zeros((1, 3620))
        helpers = SimpleNamespace(save_audio_to_wav=Mock(return_value="result.wav"), cleanup_audio=Mock())
        with patch.dict("sys.modules", {"x_base.vae_parallelism.audio_utils": helpers}):
            with patch.object(self.service, "_save_mmh3_video_ffmpeg") as save_video:
                args = SimpleNamespace(frames=360, enable_audio=True)
                self.service._save_minimax_h3_output(
                    {"videos": [frames], "audio": audio, "sampling_rate": 240}, "result.mp4", args
                )
        save_video.assert_called_once_with(frames[:360], "result.wav", "result.mp4", args)
        self.assertEqual(helpers.save_audio_to_wav.call_args.args[0].shape[-1], 3600)


class MaaSHealthTests(unittest.TestCase):
    def setUp(self):
        self.service = VideoGenerationService(maas_fl2va_input=True)
        self.service.is_initialized = True
        self.service.world_size = 8
        self.service.default_config.update(model="MiniMax-H3", sp=8)
        self.client = self.service.app.test_client()

    def test_legacy_health_response_and_routes_unchanged(self):
        client = VideoGenerationService().app.test_client()
        self.assertNotIn("ready", client.get("/health").get_json())
        self.assertEqual(client.get("/ready").status_code, 404)

    def test_starting_busy_timeout_failed_and_draining(self):
        with patch("maas_health.monotonic", return_value=100):
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.get("/ready").status_code, 503)
            self.service.maas_health = self.service.maas_health.advance("loading", group_confirmed=True)
            self.assertEqual(self.client.get("/ready").get_json()["reason"], "model_loading")
            self.service.maas_health = self.service.maas_health.advance(
                "generating", model_loaded=True, warmed=True,
            )
        with patch("maas_health.monotonic", return_value=1100):
            self.assertEqual(self.client.get("/ready").status_code, 200)
        with patch("maas_health.monotonic", return_value=1300):
            response = self.client.get("/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.get_json()["reason"], "progress_timeout")
            self.assertEqual(self.client.post("/v1/tasks", json={}).status_code, 503)
            self.assertEqual(self.client.post("/generate", json={}).status_code, 503)
        self.service.maas_health = self.service.maas_health.advance("idle", failed=True)
        self.assertEqual(self.client.get("/ready").get_json()["reason"], "worker_failed")
        self.service.shutdown = True
        self.assertEqual(self.client.get("/ready").status_code, 503)
        self.assertEqual(self.client.get("/health").get_json()["state"], "draining")

    def test_phase_deadlines_and_untimed_phases(self):
        for phase, deadline in (("loading", 600), ("generating", 1200), ("saving", 180)):
            with self.subTest(phase=phase):
                with patch("maas_health.monotonic", return_value=100):
                    self.service.maas_health = self.service.maas_health.advance(phase, group_confirmed=True)
                with patch("maas_health.monotonic", return_value=100 + deadline - 0.001):
                    self.assertEqual(self.client.get("/health").get_json()["state"], "starting")
                with patch("maas_health.monotonic", return_value=100 + deadline):
                    self.assertEqual(self.client.get("/ready").get_json()["reason"], "progress_timeout")
        for phase in ("idle", "dispatching", "confirming_workers"):
            with self.subTest(phase=phase), patch("maas_health.monotonic", return_value=100):
                self.service.maas_health = self.service.maas_health.advance(phase, group_confirmed=True)
                with patch("maas_health.monotonic", return_value=7300):
                    snapshot = self.client.get("/ready").get_json()
                    self.assertEqual((snapshot["phase"], snapshot["state"]), (phase, "starting"))

    def test_dispatcher_waits_for_peer_confirmation_after_warmup(self):
        self.assertEqual(self.client.post("/v1/tasks", json={
            "id": "warmup", "input": {"prompt": "test", "first_frame": data_uri(image_bytes())},
        }).status_code, 200)
        claim = self.service._claim_maas_task

        def finish_job(*_, **_kwargs):
            self.service.maas_health = self.service.maas_health.advance("saving", model_loaded=True)
            return True

        def broadcast(command, **_):
            if command == video_server.CMD_HEARTBEAT:
                self.assertEqual(self.client.get("/ready").status_code, 503)
                self.assertEqual(self.client.get("/ready").get_json()["phase"], "confirming_workers")

        def next_job():
            if not self.service.maas_health.warmed:
                return claim()
            self.assertEqual(self.client.get("/ready").status_code, 200)
            self.service.shutdown = True
            raise queue.Empty

        with patch.object(self.service, "_claim_maas_task", side_effect=next_job), \
             patch.object(self.service, "_process_task", side_effect=finish_job), \
             patch.object(video_server, "broadcast_int", side_effect=broadcast), \
             patch.object(video_server, "broadcast_dict"), patch.object(video_server.dist, "barrier"), \
             patch.object(video_server.torch.cuda, "empty_cache"):
            self.service._task_processor_thread()
        self.assertTrue(self.service.maas_health.warmed)

    def test_peer_failure_is_not_swallowed(self):
        with patch.object(video_server, "broadcast_int", side_effect=RuntimeError("rank fault")):
            with self.assertRaisesRegex(RuntimeError, "rank fault"):
                self.service._worker_loop()


class MaaSJobTests(unittest.TestCase):
    def setUp(self):
        self.service = VideoGenerationService(maas_fl2va_input=True)
        self.service.is_initialized = True
        self.client = self.service.app.test_client()
        self.input = {'prompt': 'test', 'first_frame': data_uri(image_bytes())}

    def submit(self, task_id='job', **changes):
        return self.service.app.test_client().post('/v1/tasks', json={
            'id': task_id, 'input': {**self.input, **changes},
        })

    def test_concurrent_submissions_and_conflict(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            responses = list(workers.map(lambda _: self.submit(), range(16)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(len(self.service.maas_pending), 1)
        self.assertEqual(self.submit(seed=43).status_code, 409)

    def test_capacity_counts_waiting_only_and_cancel_frees_slot(self):
        self.submit('running')
        _, running = self.service._claim_maas_task()
        with ThreadPoolExecutor(max_workers=8) as workers:
            responses = list(workers.map(lambda i: self.submit(f'pending-{i}'), range(11)))
        self.assertEqual(sorted(response.status_code for response in responses), [200] * 10 + [503])
        overflow = next(response.get_json() for response in responses if response.status_code == 503)
        self.assertEqual(overflow['error'], 'queue_full')
        self.assertNotIn(overflow['id'], self.service.tasks)
        first, second, *_ = self.service.maas_pending
        self.assertEqual(self.client.get(f'/v1/tasks/{second}').get_json()['queue_ahead'], 1)
        self.assertEqual(self.client.delete(f'/v1/tasks/{first}').get_json()['status'], 'cancelled')
        self.assertEqual(self.client.get(f'/status/{second}').get_json()['queue_ahead'], 0)
        self.assertEqual(self.submit(overflow['id']).status_code, 200)
        self.assertEqual(running.status, 'running')
        job, _ = self.service._claim_maas_task()
        self.assertEqual(job['task_id'], second)

    def test_ten_minute_deadline_and_elapsed_feedback(self):
        with patch('video_server.time.monotonic', return_value=100):
            for i in range(10):
                self.assertEqual(self.submit(f'queued-{i}').status_code, 200)
        with patch('video_server.time.monotonic', return_value=699.999):
            before = self.client.get('/v1/tasks/queued-9').get_json()
            self.assertEqual((before['status'], before['queue_elapsed_s'], before['queue_ahead']),
                             ('pending', 599.999, 9))
            self.assertEqual(self.submit('replacement').status_code, 503)
        with patch('video_server.time.monotonic', return_value=700):
            self.assertEqual(self.submit('replacement').status_code, 200)
            expired = self.client.get('/v1/tasks/queued-9').get_json()
            self.assertEqual((expired['error_code'], expired['timings_s']['queue']), ('queue_expired', 600))
            self.assertNotIn('queue_elapsed_s', expired)
            self.assertEqual(list(self.service.maas_pending), ['replacement'])
            self.assertEqual(self.client.get('/metrics').get_json()['tasks']['pending'], 1)

    def test_custom_queue_capacity_and_timeout(self):
        self.service = VideoGenerationService(maas_fl2va_input=True, maas_queue_capacity=1,
                                              maas_queue_timeout_s=30)
        self.service.is_initialized = True
        with patch('video_server.time.monotonic', return_value=100):
            self.assertEqual(self.submit('first').status_code, 200)
            self.assertEqual(self.submit('next').status_code, 503)
        with patch('video_server.time.monotonic', return_value=130):
            self.assertEqual(self.submit('next').status_code, 200)
        self.assertEqual(self.service.tasks['first'].error_code, 'queue_expired')

    def test_cancelled_and_expired_jobs_never_broadcast(self):
        self.service.world_size = 8
        self.submit('cancel')
        self.submit('expire')
        self.assertEqual(self.client.delete('/v1/tasks/cancel').get_json()['status'], 'cancelled')
        self.assertEqual(self.client.post('/cancel/cancel').status_code, 200)
        expired = self.service.tasks['expire']
        expired.queued_at -= video_server.MAAS_QUEUE_TIMEOUT_S
        def broadcast_heartbeat(command, **_):
            if command == video_server.CMD_HEARTBEAT:
                self.service.shutdown = True

        with patch.object(video_server, 'broadcast_int', side_effect=broadcast_heartbeat) as broadcast:
            self.service._task_processor_thread()
        self.assertNotIn(video_server.CMD_EXECUTE, [call.args[0] for call in broadcast.call_args_list])
        self.assertEqual(self.client.get('/v1/tasks/expire').get_json()['error_code'], 'queue_expired')

    def test_claim_wins_against_cancellation_and_terminal_is_stable(self):
        self.submit()
        _, task = self.service._claim_maas_task()
        self.assertIsNotNone(task)
        self.assertEqual(self.client.delete('/v1/tasks/job').status_code, 409)
        self.service._fail_maas_jobs('generating')
        self.assertFalse(task.finish('completed', result={'infer_time': 3}))
        response = self.client.get('/v1/tasks/job').get_json()
        self.assertEqual((response['status'], response['error_code'], response['failure_stage']),
                         ('failed', 'worker_lost', 'generating'))

    def test_storage_failure_keeps_terminal_failed_and_releases_lock(self):
        from tempfile import TemporaryDirectory

        self.submit()
        _, task = self.service._claim_maas_task()
        self.service.inference_manager = Mock()
        args = SimpleNamespace(model='MiniMax-H3', enable_audio=True)
        def publish(path, *_args, **_kwargs):
            Path(path).write_bytes(b'encoded')
        def fail_storage(*_args):
            self.assertEqual(self.client.get('/v1/tasks/job').get_json()['status'], 'running')
            raise video_server.StorageError('mock object not readable')

        with TemporaryDirectory() as root, patch.object(self.service, '_build_args', return_value=args), \
             patch.object(video_server.torch.cuda, 'synchronize'), patch.object(video_server, 'infer_info'), \
             patch.object(video_server, 'publish_video', side_effect=publish), \
             patch.object(video_server, 'publish_obs', side_effect=fail_storage), \
             patch.dict(video_server.os.environ, {'OBS_LOCAL_PATH': root}):
            path = str(Path(root) / 'result.mp4')
            self.assertFalse(self.service._process_task('job', {}, path, maas_task=task))
            self.assertFalse(Path(path).exists())
        response = self.client.get('/v1/tasks/job').get_json()
        self.assertEqual((response['status'], response['error_code']), ('failed', 'storage_failed'))
        self.assertEqual(self.client.get('/download/job').status_code, 400)

    def test_success_becomes_downloadable_and_reports_timings(self):
        from tempfile import TemporaryDirectory

        self.submit()
        _, task = self.service._claim_maas_task()
        self.service.inference_manager = Mock()
        args = SimpleNamespace(model='MiniMax-H3', enable_audio=True)
        def publish(path, *_args, **_kwargs):
            self.assertEqual(self.client.get('/download/job').status_code, 400)
            Path(path).write_bytes(b'encoded video')

        with TemporaryDirectory() as root, patch.object(self.service, '_build_args', return_value=args), \
             patch.object(video_server.torch.cuda, 'synchronize'), patch.object(video_server, 'infer_info'), \
             patch.object(video_server, 'publish_video', side_effect=publish), \
             patch.dict(video_server.os.environ, {'OBS_LOCAL_PATH': ''}):
            path = str(Path(root) / 'result.mp4')
            self.assertTrue(self.service._process_task('job', {}, path, maas_task=task))
            with self.client.get('/download/job') as download:
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.data, b'encoded video')
        response = self.client.get('/v1/tasks/job').get_json()
        self.assertEqual(response['status'], 'succeeded')
        self.assertIsNone(response['error_code'])
        self.assertEqual(set(response['timings_s']), {'queue', 'inference', 'encode', 'delivery', 'total'})


if __name__ == "__main__":
    logging.disable(logging.INFO)
    unittest.main(verbosity=2)
