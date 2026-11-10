"""MiniMax-H3 V2 protocol adapter. GPU execution uses the existing MaaS queue."""
from collections import Counter
import hmac
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import quote
import uuid

from flask import Blueprint, g, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from maas import MAX_BODY_BYTES, RATIOS, normalize
from maas_fl2va import InvalidInput, _read_json
from maas_media import cleanup
from maas_v2_backends import MockBackends
from maas_v2_callbacks import Callbacks
from maas_v2_store import STATUSES, TaskHistory
from x_diffusers.adaptor.h3_refiner_config import H3_MAX_PROMPT_LENGTH, h3_prompt_length

logger = logging.getLogger(__name__)
FIELDS = {'model', 'content', 'resolution', 'duration', 'ratio', 'callback_url', 'aigc_watermark'}
KINDS = {'image_url': 'image', 'video_url': 'video', 'audio_url': 'audio'}
ROLES = {'image_url': {'first_frame', 'last_frame', 'reference_image'},
         'video_url': {'reference_video'}, 'audio_url': {'reference_audio'}}


def parse_content(body, *, ir=False, file_base=None):
    fields = FIELDS-{'resolution', 'aigc_watermark'} if ir else FIELDS
    required = {'model', 'content', 'duration'} | (set() if ir else {'resolution'})
    if not isinstance(body, dict) or body.keys()-fields or required-body.keys():
        raise InvalidInput('body', 'Missing required fields or unknown request fields')
    if body['model'] != 'MiniMax-H3':
        raise InvalidInput('model', 'Only MiniMax-H3 is supported')
    duration = body['duration']
    # JSON Schema integer and the official CLI's Number.isInteger also accept
    # integral JSON numbers such as 4.0, unlike Python's type(value) is int.
    if type(duration) not in (int, float) or not 4 <= duration <= 15 or duration != int(duration):
        raise InvalidInput('duration', 'Expected an integer from 4 to 15')
    if 'callback_url' in body and (not isinstance(body['callback_url'], str) or not body['callback_url']):
        raise InvalidInput('callback_url', 'Expected a nonempty callback URL string')
    if not ir and body['resolution'] not in ('768P', '2K'):
        raise InvalidInput('resolution', 'Expected 768P or 2K')
    if 'aigc_watermark' in body and type(body['aigc_watermark']) is not bool:
        raise InvalidInput('aigc_watermark', 'Expected a boolean')
    items = body['content']
    if not isinstance(items, list) or not items:
        raise InvalidInput('content', 'Expected a nonempty content array')
    texts, references, media_types, frames = [], [], [], {}
    counts = Counter()
    images = sum(isinstance(item, dict) and item.get('type') == 'image_url' for item in items)
    for item in items:
        if not isinstance(item, dict):
            raise InvalidInput('content', 'Every content item must be an object')
        kind = item.get('type')
        if kind == 'text':
            text = item.get('text')
            if (not isinstance(text, str) or not text.strip() or '\0' in text
                    or h3_prompt_length(text) > H3_MAX_PROMPT_LENGTH):
                raise InvalidInput('content.text', 'Expected nonblank text of at most 7000 UTF-16 code units')
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                raise InvalidInput('content.text', 'Invalid Unicode text') from None
            if item.keys()-{'type', 'text'}:
                raise InvalidInput('content', 'Text items only accept type and text')
            texts.append(text)
            continue
        if not isinstance(kind, str) or kind not in KINDS:
            raise InvalidInput('content.type', 'Expected text/image_url/video_url/audio_url')
        if item.keys()-{'type', kind, 'role'}:
            raise InvalidInput('content', 'Unexpected media item fields')
        value = item.get(kind)
        if not isinstance(value, dict) or set(value) != {'url'} or not isinstance(value['url'], str) or not value['url']:
            raise InvalidInput('content', 'Media items require a nested url string')
        source = value['url']
        if source.startswith('mm_file://'):
            identifier = source[len('mm_file://'):]
            if not file_base or not identifier or any(c in identifier for c in '/?#'):
                raise InvalidInput('content', 'Platform file resolver is not configured or file ID is invalid', 422)
            source = file_base.rstrip('/')+'/'+quote(identifier, safe='')
        role = item.get('role')
        if role is None and kind == 'image_url' and images == 1:
            role = 'first_frame'
        if not isinstance(role, str) or role not in ROLES[kind]:
            raise InvalidInput('content.role', 'Missing or incompatible media role')
        counts[KINDS[kind]] += 1
        if role in ('first_frame', 'last_frame'):
            if role in frames:
                raise InvalidInput('content.role', 'At most one first frame and one last frame')
            frames[role] = source
        else:
            references.append(source)
            media_types.append(KINDS[kind])
    if not texts:
        raise InvalidInput('content', 'A nonempty text item is required')
    if len(texts) != 1:
        # Preserve semantics: the model accepts a single prompt. Never discard or
        # silently merge independently ordered text items.
        raise InvalidInput('content', 'This engine currently supports exactly one text item', 422)
    if frames and references:
        raise InvalidInput('content.role', 'Frame inputs and reference inputs cannot be combined')
    if len(references) > 12 or (not frames and any(counts[k] > n for k, n in [('image', 9), ('video', 3), ('audio', 3)])):
        raise InvalidInput('content', 'At most 9 reference images, 3 videos, 3 audios and 12 media files')
    mode = 'ref2va' if references else ('fl2va' if frames else 't2va')
    ratio = body.get('ratio', 'adaptive')
    if not isinstance(ratio, str) or ratio not in RATIOS+('adaptive',) or (mode == 't2va' and ratio == 'adaptive'):
        raise InvalidInput('ratio', 'T2VA requires a concrete ratio; other modes may use adaptive')
    result = dict(model='MiniMax-H3', task_type=mode, prompt=texts[0], duration=int(duration),
                  resolution='768P', aspect_ratio=ratio, ref_path=references, **frames)
    if not ir:
        result['aigc_watermark'] = body.get('aigc_watermark', False)
    return result, media_types


class MaaSV2:
    def __init__(self, service, *, directory=None, mock_backends=False):
        self.service = service
        self.history = TaskHistory(directory or os.getenv('XGEN_MAAS_STATE_DIR', 'outputs/maas-state'))
        self.callbacks = Callbacks()
        self.backend = MockBackends() if mock_backends else None
        self.mock_slots = threading.BoundedSemaphore(2)
        self.file_base = os.getenv('XGEN_MAAS_FILE_BASE_URL')
        self.download_key = None
        self.api_key = os.getenv('XGEN_MAAS_API_KEY')
        self.url_ttl_s = int(os.getenv('XGEN_MAAS_URL_TTL_S', '3600'))
        if not 1 <= self.url_ttl_s <= 7*24*3600:
            raise ValueError('XGEN_MAAS_URL_TTL_S must be between 1 and 604800')
        self.blueprint = Blueprint('h3_v2', __name__, url_prefix='/v2')
        self._routes()
        service.app.register_blueprint(self.blueprint)

        def http_error(error):
            if request.path.startswith('/v2/'):
                return self.error(error.description, 400)
            return error.get_response()
        service.app.register_error_handler(HTTPException, http_error)

    def close(self, wait=True):
        self.callbacks.close(wait=wait)
        self.history.close()

    @staticmethod
    def error(message, status=400):
        category = {400: 'bad_request_error', 401: 'authorized_error', 402: 'insufficient_balance_error',
                    422: 'unprocessable_entity_error', 429: 'rate_limit_error', 500: 'server_error'}.get(status, 'server_error')
        identifier = getattr(g, 'h3_request_id', None) or uuid.uuid4().hex
        response = jsonify(type='error', error=dict(type=category, message=message, http_code=str(status)), request_id=identifier)
        response.headers['X-Request-ID'] = identifier
        return response, status

    def _save(self, document, metadata):
        self.history.put(document, metadata)
        self.callbacks.notify(metadata.get('callback_url'), document)

    def _signature(self, identifier, expires):
        # Persistent signing key; never included in task records or responses.
        with self.history.lock:
            if self.download_key is None:
                with self.history.connection():
                    path = self.history.directory/'download.key'
                    try:
                        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    except FileExistsError:
                        pass
                    else:
                        with os.fdopen(descriptor, 'wb') as key_file:
                            key_file.write(os.urandom(32))
                    key = path.read_bytes()
                    if len(key) != 32:
                        raise RuntimeError('Invalid download signing key')
                    self.download_key = key
        return hmac.new(self.download_key, f'{identifier}:{expires}'.encode(), hashlib.sha256).hexdigest()

    def _video_url(self, identifier, metadata):
        expires = int(time.time())+self.url_ttl_s
        return metadata['base_url']+'/v2/video_outputs/'+identifier+f'?expires={expires}&signature='+self._signature(identifier, expires)

    def task_changed(self, task):
        metadata = task.v2_metadata
        document = dict(id=task.task_id, model='MiniMax-H3', status={'pending': 'queued', 'completed': 'succeeded'}.get(task.status, task.status),
                        created_at=int(task.created_at.timestamp()), updated_at=int(time.time()),
                        resolution=metadata['resolution'], duration=metadata['duration'], ratio=metadata['ratio'],
                        task_type='generation', modality='video')
        if task.status == 'completed':
            metadata['local_path'] = str(Path(task.result['video_path']).resolve())
            document['content'] = dict(url=self._video_url(task.task_id, metadata))
            usage = metadata['usage']
            seconds = math.ceil(usage['input_video_seconds'])
            document['usage'] = dict(input_seconds=seconds, output_seconds=metadata['duration'],
                                     total_seconds=seconds+metadata['duration'], input_image_count=usage['input_image_count'])
            if usage['input_audio_seconds']:
                document['usage']['input_audio_seconds'] = math.floor(usage['input_audio_seconds']+.5)
        elif task.status == 'failed':
            document['error'] = dict(code=task.error_code or 'engine_error', message='Task failed during '+str(task.failure_stage or 'inference'))
        self._save(document, metadata)

    def record(self, identifier):
        with self.service.tasks_lock:
            self.service._expire_maas_pending(time.monotonic())
            record = self.history.get(identifier)
            task = self.service.tasks.get(identifier)
            if record is not None and task is not None and getattr(task, 'on_change', None):
                status = {'pending': 'queued', 'completed': 'succeeded'}.get(task.status, task.status)
                if record[0]['status'] != status:
                    # Reconcile metadata after a transient storage error. This
                    # never reruns inference or restores latent state.
                    task.on_change(task)
                    record = self.history.get(identifier)
        if record is None:
            raise InvalidInput('task_id', 'Invalid task_id')
        document, metadata = record
        if document['status'] == 'succeeded' and document['modality'] == 'video':
            document['content'] = dict(url=self._video_url(identifier, metadata))
        return record

    def _response(self, value, metadata=None):
        response = jsonify(value)
        if metadata and metadata.get('mock'):
            response.headers['X-XGen-Mock'] = metadata['mock']
        return response

    def create(self, *, ir=False):
        if not self.service.is_initialized or self.service.shutdown:
            return self.error('Service is unavailable', 500)
        body = _read_json(request, MAX_BODY_BYTES)
        values, kinds = parse_content(body, ir=ir, file_base=self.file_base)
        body = dict(body, duration=values['duration'])
        mock = ir or body['resolution'] == '2K'
        if mock and self.backend is None:
            raise InvalidInput('model', 'IR/2K provider is not configured; mock backends require explicit startup opt-in', 422)
        if not self.service.maas_admission.acquire(blocking=False):
            return self.error('Media admission is busy', 429)
        normalized = None
        try:
            config = self.service.default_config
            if mock:
                config = dict(config, task_type=values['task_type'], service_mode=values['task_type'])
            normalized = normalize(values, config, media_types=kinds)
            self.callbacks.verify(body.get('callback_url'))
            identifier = uuid.uuid4().hex
            metadata = dict(resolution=body.get('resolution'), duration=body['duration'],
                ratio=normalized['aspect_ratio'], callback_url=body.get('callback_url'),
                base_url=(self.service.default_config.get('external_base_url') or request.url_root).rstrip('/'),
                usage=normalized['_maas_usage'])
            if mock:
                metadata['mock'] = 'ir' if ir else '2k'
                return self._mock(identifier, body, normalized, metadata, ir=ir)
            normalized['_maas_v2_metadata'] = metadata
            response, status = self.service._submit_maas_task(identifier, normalized)
            if status != 200:
                payload = response.get_json()
                return self.error('Engine admission rejected: '+str(payload.get('error', 'unavailable')),
                                  429 if payload.get('error') in ('queue_full', 'too_many_tasks') else 500)
            return self._response(dict(task_id=identifier))
        finally:
            self.service.maas_admission.release()
            if normalized is not None and not normalized.get('_maas_adopted'):
                cleanup(normalized)

    def _mock(self, identifier, body, normalized, metadata, *, ir):
        if not self.mock_slots.acquire(blocking=False):
            return self.error('Mock backend is busy', 429)
        document = dict(id=identifier, model='MiniMax-H3', status='queued', created_at=int(time.time()), updated_at=int(time.time()),
                        task_type='h3_context_ir' if ir else 'generation', modality='text' if ir else 'video',
                        duration=body['duration'], ratio=normalized['aspect_ratio'])
        if not ir:
            document['resolution'] = '2K'
        try:
            self._save(document, metadata)
            document.update(status='running', updated_at=int(time.time()))
            self._save(document, metadata)
            if ir:
                document['content'] = self.backend.context_ir(body, normalized)
            else:
                from maas_artifacts import publish_video
                width, height = normalized['_maas_size']
                if (width, height) in ((1344, 768), (768, 1344)):
                    width, height = (2560, 1440) if width > height else (1440, 2560)
                else:
                    ratio = width/height
                    width = int(math.sqrt(2560*1440*ratio)/32)*32
                    height = int(math.sqrt(2560*1440/ratio)/32)*32
                path = self.history.directory/'mock-videos'/(identifier+'.mp4')
                path.parent.mkdir(parents=True, exist_ok=True)
                publish_video(path, lambda target: self.backend.video_2k(target, width, height, body['duration'], request=body, normalized=normalized),
                              require_audio=True, expected=(width, height, body['duration']*24))
                metadata['local_path'] = str(path.resolve())
                document['content'] = dict(url=self._video_url(identifier, metadata))
            document.update(status='succeeded', updated_at=int(time.time()))
            self._save(document, metadata)
        except Exception:
            logger.exception('Mock H3 backend failed for task %s', identifier)
            document.pop('content', None)
            document.update(status='failed', updated_at=int(time.time()), error=dict(code='mock_backend_failed', message='Mock backend failed'))
            self._save(document, metadata)
        finally:
            self.mock_slots.release()
        return self._response(dict(task_id=identifier), metadata)

    def _routes(self):
        bp = self.blueprint

        @bp.before_request
        def request_id():
            g.h3_request_id = uuid.uuid4().hex
            if request.endpoint != 'h3_v2.download' and self.api_key and not hmac.compare_digest(request.headers.get('Authorization', '').encode(),
                                                        ('Bearer '+self.api_key).encode()):
                return self.error('Invalid or missing Bearer token', 401)

        @bp.after_request
        def response_id(response):
            response.headers['X-Request-ID'] = g.h3_request_id
            if self.backend is not None:
                response.headers['X-XGen-Mock-Enabled'] = 'ir,2k'
            return response

        @bp.errorhandler(InvalidInput)
        def invalid(error):
            status = error.status if error.status in (422, 429, 500) else 400
            return self.error(str(error), status)

        @bp.errorhandler(Exception)
        def failed(error):
            if isinstance(error, HTTPException):
                return self.error(error.description, 400)
            logger.exception('V2 request failed request_id=%s', g.h3_request_id)
            return self.error('Internal server error', 500)

        bp.add_url_rule('/video_generation', 'create', self.create, methods=['POST'])
        bp.add_url_rule('/h3_context_ir', 'ir', lambda: self.create(ir=True), methods=['POST'])

        @bp.get('/openapi.json')
        def openapi():
            path = Path(__file__).resolve().parents[2]/'docs/openapi/h3-maas-v2.json'
            return jsonify(json.loads(path.read_text(encoding='utf-8')))

        @bp.get('/query/video_generation/<task_id>')
        def query(task_id):
            document, metadata = self.record(task_id)
            return self._response(dict(task=document), metadata)

        @bp.get('/query/video_generation')
        def query_list():
            allowed = {'page_num', 'page_size', 'filter.status', 'filter.task_ids', 'filter.model', 'filter.task_type'}
            if request.args.keys()-allowed:
                raise InvalidInput('query', 'Unknown query parameters')
            for key in request.args:
                if key != 'filter.task_ids' and len(request.args.getlist(key)) != 1:
                    raise InvalidInput('query', 'Only filter.task_ids accepts repeated values')
            try:
                page = int(request.args.get('page_num', '1'))
                size = int(request.args.get('page_size', '20'))
                if page < 1 or not 1 <= size <= 1000:
                    raise ValueError()
            except ValueError:
                raise InvalidInput('query', 'Expected page_num >= 1 and page_size between 1 and 1000') from None
            filters = {key[7:]: request.args.getlist(key) if key == 'filter.task_ids' else value
                       for key, value in request.args.items() if key.startswith('filter.')}
            if 'status' in filters and filters['status'] not in STATUSES:
                raise InvalidInput('filter.status', 'Unknown task status')
            if 'task_type' in filters and filters['task_type'] not in ('generation', 'h3_context_ir', 'regeneration'):
                raise InvalidInput('filter.task_type', 'Unknown task type')
            with self.service.tasks_lock:
                self.service._expire_maas_pending(time.monotonic())
                result, metadatas = self.history.list(filters, page, size)
                for document in result['items']:
                    if document['status'] == 'succeeded' and document['modality'] == 'video':
                        document['content'] = dict(url=self._video_url(document['id'], metadatas[document['id']]))
            return self._response(result)

        @bp.delete('/video_generation/<task_id>')
        def delete(task_id):
            # Same lock as claim/expire: cancellation cannot race a successful claim.
            with self.service.tasks_lock:
                self.service._expire_maas_pending(time.monotonic())
                record = self.history.get(task_id)
                if record is None:
                    raise InvalidInput('task_id', 'Invalid task_id')
                document, metadata = record
                if document['status'] == 'queued':
                    task = self.service.tasks.get(task_id)
                    if task is None or task_id not in self.service.maas_pending:
                        raise InvalidInput('task_id', 'Task cannot be cancelled')
                    cleanup(self.service.maas_pending.pop(task_id)['data'])
                    task.timings_s['queue'] = round(time.monotonic()-task.queued_at, 3)
                    task.finish('cancelled', code='cancelled_by_user', stage='queue')
                    action = 'cancelled'
                elif document['status'] in ('succeeded', 'failed'):
                    self.history.delete(task_id)
                    self.service.tasks.pop(task_id, None)
                    action = 'deleted'
                else:
                    raise InvalidInput('task_id', 'Running and cancelled tasks cannot be cancelled or deleted')
            return self._response(dict(task_id=task_id, action=action, status=action), metadata)

        @bp.get('/video_outputs/<task_id>')
        def download(task_id):
            try:
                expires = int(request.args['expires'])
                signature = request.args['signature']
                if expires < time.time() or not hmac.compare_digest(signature.encode(), self._signature(task_id, expires).encode()):
                    raise ValueError()
            except (KeyError, ValueError):
                return self.error('Invalid or expired download URL', 401)
            document, metadata = self.record(task_id)
            path = metadata.get('local_path')
            if document['status'] != 'succeeded' or not path or not Path(path).is_file():
                raise InvalidInput('task_id', 'Video is unavailable')
            response = send_file(path, mimetype='video/mp4', as_attachment=True, download_name=task_id+'.mp4')
            if metadata.get('mock'):
                response.headers['X-XGen-Mock'] = metadata['mock']
            return response
