"""V2 HTTP contracts, real admission/queue, persistence, callbacks and FFmpeg mocks.

No model generation is performed. Run alongside the existing MaaS CPU regressions.
"""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from maas_artifacts import watermark_converter
from maas_media import cleanup
from maas_v2 import parse_content
from maas_v2_backends import MockBackends
from maas_v2_store import TaskHistory, RETENTION_S
from video_server import VideoGenerationService


def picture(width=256, height=256):
    data = io.BytesIO()
    Image.new('RGB', (width, height), '#336699').save(data, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()


def request_body(**changes):
    return dict(dict(model='MiniMax-H3', content=[dict(type='text', text='Two people talk and stand up.')],
                     resolution='768P', duration=5, ratio='16:9'), **changes)


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.root = TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.service = self.new_service()
        self.client = self.service.app.test_client()

    def new_service(self, **options):
        # Do not replace the real routing, queue or task lifecycle.
        with patch.object(VideoGenerationService, '_setup_signal_handlers'):
            service = VideoGenerationService(maas=True, maas_state_dir=str(Path(self.root.name)/'history'), **options)
        service.default_config.update(model='MiniMax-H3', task_type='t2va', service_mode='fl2va', num_inference_steps=20)
        service.is_initialized = True
        return service

    def tearDown(self):
        self.service._cleanup_on_exit()

    def post(self, body=None, path='/v2/video_generation'):
        return self.client.post(path, json=request_body() if body is None else body)

    def query(self, identifier):
        return self.client.get('/v2/query/video_generation/'+identifier)

    def create(self, body=None):
        response = self.post(body)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(set(response.json), {'task_id'})
        return response.json['task_id']

    def finish(self):
        job, task = self.service._claim_maas_task()
        output = Path(self.root.name)/(task.task_id+'.mp4')
        output.write_bytes(b'fixture: serializer/download test, not a generated video')
        with self.service.tasks_lock:
            task.finish('completed', result=dict(video_path=str(output), infer_time=.1))
        cleanup(job['data'])
        return task.task_id

    def assert_error(self, response, status=400):
        self.assertEqual(response.status_code, status, response.json)
        self.assertEqual(set(response.json), {'type', 'error', 'request_id'})
        self.assertEqual(response.json['type'], 'error')
        self.assertEqual(response.json['error']['http_code'], str(status))
        self.assertEqual(response.headers['X-Request-ID'], response.json['request_id'])

    def test_create_query_and_old_protocol_are_distinct(self):
        identifier = self.create(request_body(duration=4))
        task = self.query(identifier).json['task']
        self.assertEqual((task['status'], task['duration'], task['resolution']), ('queued', 4, '768P'))
        self.assertIs(type(task['created_at']), int)
        self.assertLess(abs(task['created_at']-time.time()), 3)
        self.assertEqual(self.client.get('/v1/tasks/'+identifier).json['status'], 'pending')
        job, running = self.service._claim_maas_task()
        self.assertEqual(self.query(identifier).json['task']['status'], 'running')
        running.finish('failed', code='injected', stage='generating')
        self.assertEqual(self.query(identifier).json['task']['error']['code'], 'injected')
        cleanup(job['data'])

    def test_success_url_object_download_and_usage(self):
        identifier = self.create()
        self.assertEqual(self.finish(), identifier)
        task = self.query(identifier).json['task']
        self.assertEqual(task['status'], 'succeeded')
        self.assertEqual(set(task['content']), {'url'})
        self.assertNotIn('output', task)
        self.assertNotIn('video_path', json.dumps(task))
        self.assertEqual(task['usage']['output_seconds'], 5)
        with self.client.get(task['content']['url']) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'fixture', response.data)

    def test_required_fields_enums_and_wire_errors(self):
        bodies = []
        for field in ('model', 'content', 'duration', 'resolution', 'ratio'):
            body = request_body(); body.pop(field); bodies.append(body)
        bodies += [request_body(duration=value) for value in (3, 16, 4.5, True, '5', 10**400)]
        bodies += [request_body(resolution=value) for value in ('768p', '480P', None)]
        bodies += [request_body(ratio=value) for value in ('adaptive', 'bad', [])]
        bodies += [request_body(aigc_watermark=value) for value in ('true', 1)]
        bodies += [request_body(callback_url=None)]
        bodies += [request_body(seed=42), request_body(num_inference_steps=1)]
        for body in bodies:
            with self.subTest(body=str(body)[:160]):
                self.assert_error(self.post(body))
        for raw in ('{"model":"MiniMax-H3","model":"other"}', '{"x":NaN}', '[', 'null'):
            self.assert_error(self.client.post('/v2/video_generation', data=raw, content_type='application/json'))
        self.assert_error(self.client.post('/v2/video_generation', data='{}'))
        self.assertFalse(self.service.maas_pending)

    def test_utf16_limits_on_content_text(self):
        for units in (6999, 7000, 7001):
            text = '🙂'*(units//2)+'a'*(units%2)
            response = self.post(request_body(content=[dict(type='text', text=text)]))
            if units <= 7000:
                self.assertEqual(response.status_code, 200, response.json)
            else:
                self.assert_error(response)

    def test_integral_json_numbers_match_official_cli(self):
        identifier = self.create(request_body(duration=4.0))
        task = self.query(identifier).json['task']
        self.assertIs(type(task['duration']), int)
        self.assertEqual(task['duration'], 4)

    def test_first_last_and_five_references_adaptive(self):
        for roles in (['first_frame'], ['last_frame'], ['first_frame', 'last_frame']):
            with self.subTest(roles=roles):
                content = [dict(type='text', text='test')]+[
                    dict(type='image_url', image_url=dict(url=picture(512, 256)), role=role) for role in roles]
                identifier = self.create(request_body(content=content))
                job, task = self.service._claim_maas_task()
                self.assertEqual(job['data']['aspect_ratio'], 'adaptive')
                args = self.service._build_args(job['data'])
                self.assertEqual(args._maas_watermark, False)
                self.assertTrue(args.keyframe_image if 'first_frame' in roles else args.keyframe_last_image)
                task.finish('failed', code='test_done', stage='test'); cleanup(job['data'])
        self.service.default_config.update(task_type='ref2va', service_mode='ref2va')
        content = [dict(type='text', text='test')]+[dict(type='image_url', role='reference_image', image_url=dict(url=picture())) for _ in range(5)]
        identifier = self.create(request_body(content=content, ratio='adaptive'))
        job, task = self.service._claim_maas_task()
        self.assertEqual(len(job['data']['ref_path']), 5)
        self.assertEqual(job['data']['_maas_size'], (992, 992))
        task.finish('failed', code='test_done', stage='test'); cleanup(job['data'])
        self.assertEqual(self.query(identifier).json['task']['ratio'], 'adaptive')

    def test_content_roles_counts_and_type_mismatch(self):
        text = dict(type='text', text='test')
        image = dict(type='image_url', role='reference_image', image_url=dict(url=picture()))
        video = dict(type='video_url', role='reference_video', video_url=dict(url=picture()))
        self.service.default_config.update(task_type='ref2va', service_mode='ref2va')
        for content in ([text]+[image]*10, [text]+[video]*4, [text]+[image]*9+[video]*3+
                        [dict(type='audio_url', role='reference_audio', audio_url=dict(url=picture()))],
                        [text, image, dict(image, role='last_frame')], [text, dict(image, role=[])],
                        [text, dict(image, image_url={'url': ''})], [text, video]):
            with self.subTest(content=len(content)):
                self.assert_error(self.post(request_body(content=content)))
        self.assertFalse(self.service.maas_pending)

    def test_cancel_delete_semantics(self):
        identifier = self.create()
        response = self.client.delete('/v2/video_generation/'+identifier)
        self.assertEqual(response.json, dict(task_id=identifier, action='cancelled', status='cancelled'))
        self.assertEqual(self.query(identifier).json['task']['status'], 'cancelled')
        self.assert_error(self.client.delete('/v2/video_generation/'+identifier))
        second = self.create()
        self.finish()
        response = self.client.delete('/v2/video_generation/'+second)
        self.assertEqual(response.json['action'], 'deleted')
        self.assert_error(self.query(second))
        self.assert_error(self.client.delete('/v2/video_generation/missing'))

    def test_running_cannot_be_cancelled(self):
        identifier = self.create()
        job, task = self.service._claim_maas_task()
        self.assert_error(self.client.delete('/v2/video_generation/'+identifier))
        self.assertEqual(task.status, 'running')
        task.finish('failed', code='injected', stage='test'); cleanup(job['data'])
        self.assertEqual(self.client.delete('/v2/video_generation/'+identifier).json['action'], 'deleted')

    def test_queue_full_and_expiry_use_v2_errors_and_status(self):
        self.service.maas_queue_capacity = 1
        identifier = self.create()
        self.assert_error(self.post(), 429)
        self.service.tasks[identifier].queued_at -= 601
        task = self.query(identifier).json['task']
        self.assertEqual((task['status'], task['error']['code']), ('failed', 'queue_expired'))
        self.assertEqual(self.post().status_code, 200)

    def test_list_pagination_filters_and_multiple_ids(self):
        first, second = self.create(), self.create()
        self.client.delete('/v2/video_generation/'+first)
        response = self.client.get('/v2/query/video_generation?filter.status=queued&page_size=1')
        self.assertEqual(response.json['total'], 1)
        self.assertEqual(response.json['items'][0]['id'], second)
        response = self.client.get('/v2/query/video_generation', query_string=[('filter.task_ids', first), ('filter.task_ids', second)])
        self.assertEqual(response.json['total'], 2)
        for query in ('page_size=0', 'page_num=0', 'page_num=1.5', 'filter.status=pending', 'unknown=1'):
            self.assert_error(self.client.get('/v2/query/video_generation?'+query))

    def test_callbacks_challenge_and_ordered_query_bodies(self):
        sent = []
        def transport(url, body):
            sent.append(body)
            return body if 'challenge' in body else None
        self.service.maas_v2.callbacks.transport = transport
        identifier = self.create(request_body(callback_url='https://callbacks.example.com/h3'))
        self.finish()
        self.service.maas_v2.callbacks.close()
        self.assertIn('challenge', sent[0])
        self.assertEqual([body['task']['status'] for body in sent[1:]], ['queued', 'running', 'succeeded'])
        from urllib.parse import urlsplit
        latest = self.query(identifier).json
        for body in (sent[-1], latest):
            body['task']['content']['url'] = urlsplit(body['task']['content']['url']).path
        self.assertEqual(sent[-1], latest)

    def test_bad_callback_does_not_admit_or_leak_media(self):
        self.service.maas_v2.callbacks.transport = lambda *_: {'challenge': 'wrong'}
        self.assert_error(self.post(request_body(callback_url='https://callbacks.example.com/h3')))
        self.assertFalse(self.service.tasks)

    def test_restart_preserves_terminal_and_fails_unfinished_without_retry(self):
        completed = self.create(); self.finish()
        old_url = self.query(completed).json['task']['content']['url']
        queued = self.create()
        for job in self.service.maas_pending.values(): cleanup(job['data'])
        self.service.maas_pending.clear(); self.service.tasks.clear()
        self.service.maas_v2.close()
        self.service = self.new_service(); self.client = self.service.app.test_client()
        self.assertEqual(self.query(completed).json['task']['status'], 'succeeded')
        with self.client.get(old_url) as download:
            self.assertEqual(download.status_code, 200)
        record = self.query(queued).json['task']
        self.assertEqual((record['status'], record['error']['code']), ('failed', 'engine_restarted'))
        self.assertFalse(self.service.maas_pending)

    def test_history_expiry_and_exclusive_directory(self):
        identifier = self.create()
        other = TaskHistory(Path(self.root.name)/'history')
        with self.assertRaises(RuntimeError): other.get(identifier)
        other.close()
        store = self.service.maas_v2.history
        doc, meta = store.get(identifier)
        doc['created_at'] = time.time()-RETENTION_S-1
        store.put(doc, meta)
        self.assertIsNone(store.get(identifier))

    def test_transient_metadata_failure_does_not_rerun_or_orphan_admitted_media(self):
        identifier = self.create()
        with patch.object(self.service.maas_v2.history, 'put', side_effect=OSError('injected metadata write failure')):
            with self.assertRaises(OSError): self.service._claim_maas_task()
        self.assertIn(identifier, self.service.maas_pending)
        self.service._fail_maas_jobs('dispatching')
        self.assertEqual(self.query(identifier).json['task']['status'], 'failed')
        self.assertFalse(self.service.maas_pending)

    def test_mock_backends_are_explicit_and_ir_is_not_real_enhancement(self):
        body = request_body(); body.pop('resolution')
        self.assert_error(self.post(body, '/v2/h3_context_ir'), 422)
        self.assert_error(self.post(request_body(resolution='2K')), 422)
        self.service.maas_v2.backend = MockBackends()
        response = self.post(body, '/v2/h3_context_ir')
        self.assertEqual(response.headers['X-XGen-Mock'], 'ir')
        result = self.query(response.json['task_id'])
        self.assertEqual(result.headers['X-XGen-Mock'], 'ir')
        self.assertEqual(result.json['task']['content']['prompt'], body['content'][0]['text'])
        self.assertEqual(result.json['task']['task_type'], 'h3_context_ir')
        self.assertNotIn('usage', result.json['task'])
        self.assertFalse(self.service.maas_pending)

    def test_2k_mock_outputs_real_labelled_media_without_gpu(self):
        self.service.maas_v2.backend = MockBackends()
        response = self.post(request_body(resolution='2K', duration=4))
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.headers['X-XGen-Mock'], '2k')
        identifier = response.json['task_id']
        record = self.query(identifier)
        self.assertEqual(record.json['task']['status'], 'succeeded', record.json)
        _, metadata = self.service.maas_v2.history.get(identifier)
        probe = subprocess.run([os.getenv('FFPROBE_PATH', 'ffprobe'), '-v', 'error', '-show_streams', '-of', 'json', metadata['local_path']],
                               capture_output=True, text=True, check=True, timeout=15)
        video = next(s for s in json.loads(probe.stdout)['streams'] if s['codec_type'] == 'video')
        self.assertEqual((video['width'], video['height'], int(video['nb_frames'])), (2560, 1440, 96))
        self.assertFalse(self.service.maas_pending)

    def test_mock_failure_remains_failed_and_never_exposes_a_video(self):
        self.service.maas_v2.backend = MockBackends()
        with patch.object(self.service.maas_v2.backend, 'video_2k', side_effect=OSError('injected')):
            identifier = self.create(request_body(resolution='2K'))
        record = self.query(identifier).json['task']
        self.assertEqual(record['status'], 'failed')
        self.assertNotIn('content', record)
        self.assert_error(self.client.get('/v2/video_outputs/'+identifier), 401)

    def test_mm_file_resolver_and_extra_text_are_explicit(self):
        body = request_body(content=[dict(type='text', text='test'), dict(type='image_url', image_url=dict(url='mm_file://file_1'))])
        self.assert_error(self.post(body), 422)
        normalized, _ = parse_content(body, file_base='https://assets.example.com/files')
        self.assertEqual(normalized['first_frame'], 'https://assets.example.com/files/file_1')
        self.assert_error(self.post(request_body(content=[dict(type='text', text='a'), dict(type='text', text='b')])), 422)

    def test_watermark_is_opt_in_and_does_not_modify_source_frames(self):
        identifier = self.create(request_body(aigc_watermark=True))
        job, task = self.service._claim_maas_task()
        args = self.service._build_args(job['data'])
        self.assertTrue(args._maas_watermark)
        frame = np.zeros((768, 1344, 3), dtype=np.uint8)
        marked = watermark_converter(lambda value: value, 1344, 768)(frame)
        self.assertEqual(int(frame.sum()), 0)
        self.assertGreater(int(marked.sum()), 0)
        # Exercise the real ffmpeg writer, including the optional overlay branch.
        args.model = 'MiniMax-H3'; args.save_fps = 24
        path = Path(self.root.name)/'watermarked.mp4'
        self.service._save_mmh3_video_ffmpeg([frame]*2, None, str(path), args)
        pixels = subprocess.run([os.getenv('FFMPEG_PATH', 'ffmpeg'), '-v', 'error', '-i', str(path),
                                 '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                capture_output=True, check=True, timeout=15).stdout
        decoded = np.frombuffer(pixels, dtype=np.uint8).reshape(768, 1344, 3)
        self.assertGreater(float(decoded[705:745, 1190:1320, 0].mean()), 100)
        task.finish('failed', code='test_done', stage='test'); cleanup(job['data'])

    def test_without_maas_old_server_does_not_register_v2(self):
        with patch.object(VideoGenerationService, '_setup_signal_handlers'):
            service = VideoGenerationService()
        self.assertEqual(service.app.test_client().post('/v2/video_generation', json=request_body()).status_code, 404)
        self.assertIsNone(getattr(service, 'maas_v2', None))

    def test_optional_internal_bearer_token(self):
        self.service.maas_v2.api_key = 'test-only-token'
        self.assert_error(self.post(), 401)
        response = self.client.post('/v2/video_generation', json=request_body(),
                                    headers={'Authorization': 'Bearer test-only-token'})
        self.assertEqual(response.status_code, 200, response.json)

    def test_signed_download_does_not_require_internal_bearer_and_rejects_tampering(self):
        identifier = self.create(); self.finish()
        self.service.maas_v2.api_key = 'test-only-token'
        response = self.client.get('/v2/query/video_generation/'+identifier, headers={'Authorization': 'Bearer test-only-token'})
        url = response.json['task']['content']['url']
        with self.client.get(url) as download:
            self.assertEqual(download.status_code, 200)
        self.assert_error(self.client.get(url+'bad'), 401)
        self.assert_error(self.client.get('/v2/video_outputs/'+identifier), 401)
        with patch('maas_v2.time.time', return_value=time.time()+3601):
            self.assert_error(self.client.get(url), 401)

    def test_claim_and_cancel_share_one_atomic_boundary(self):
        from concurrent.futures import ThreadPoolExecutor
        import queue
        identifier = self.create()
        def claim():
            try:
                return self.service._claim_maas_task()
            except queue.Empty:
                return None
        def cancel():
            with self.service.app.test_client() as client:
                return client.delete('/v2/video_generation/'+identifier).status_code
        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = executor.submit(claim)
            cancelled = executor.submit(cancel)
            job = claimed.result(); status = cancelled.result()
        if job is None:
            self.assertEqual(status, 200)
            self.assertEqual(self.query(identifier).json['task']['status'], 'cancelled')
        else:
            self.assertEqual(status, 400)
            payload, task = job
            self.assertEqual(task.status, 'running')
            task.finish('failed', code='test_done', stage='test'); cleanup(payload['data'])

    def test_cli_guards_run_before_model_initialization(self):
        import contextlib
        import sys
        from video_server import main
        for flags in (['--maas', '--maas-queue-capacity', '0'],
                      ['--maas-fl2va-input', '--maas-queue-capacity', '0'],
                      ['--maas-mock-backends'], ['--maas-state-dir', 'state']):
            with self.subTest(flags=flags), patch.object(sys, 'argv', ['video_server.py']+flags), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_openapi_and_http_error_envelopes(self):
        response = self.client.get('/v2/openapi.json')
        self.assertEqual(response.status_code, 200, response.json)
        spec = response.json
        self.assertEqual(set(spec['paths']), {'/v2/video_generation', '/v2/h3_context_ir',
            '/v2/query/video_generation', '/v2/query/video_generation/{task_id}', '/v2/video_generation/{task_id}'})
        schema = spec['components']['schemas']['VideoGenerationV2Req']
        self.assertEqual(schema['properties']['model']['enum'], ['MiniMax-H3'])
        self.assertEqual(schema['properties']['resolution']['enum'], ['768P', '2K'])
        self.assertEqual(schema['properties']['duration']['enum'], list(range(4, 16)))
        pending = [spec]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if '$ref' in item:
                    target = spec
                    for part in item['$ref'][2:].split('/'):
                        target = target[part.replace('~1', '/').replace('~0', '~')]
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        self.assert_error(self.client.get('/v2/video_generation'))
        self.assert_error(self.client.post('/v2/video_regeneration', json={}))


if __name__ == '__main__':
    unittest.main()
