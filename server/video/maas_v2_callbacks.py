"""Best-effort V2 callbacks, independent of the GPU worker and inference queue."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import logging
import ssl
import time
import uuid

from maas_fl2va import InvalidInput
from maas_media import public_target

logger = logging.getLogger(__name__)


def post_json(url, payload, timeout=3):
    deadline = time.monotonic() + timeout
    parsed, port, ip = public_target(url)
    connection = http.client.HTTPConnection(ip, port, timeout=max(.01, deadline-time.monotonic()))
    try:
        connection.connect()
        if parsed.scheme == 'https':
            connection.sock = ssl.create_default_context().wrap_socket(connection.sock, server_hostname=parsed.hostname)
        if time.monotonic() >= deadline:
            raise TimeoutError('Callback deadline exceeded')
        connection.sock.settimeout(max(.01, deadline-time.monotonic()))
        resource = (parsed.path or '/') + ('?'+parsed.query if parsed.query else '')
        connection.request('POST', resource, body=json.dumps(payload).encode(),
            headers={'Host': parsed.netloc, 'Content-Type': 'application/json', 'Accept-Encoding': 'identity'})
        response = connection.getresponse()
        raw = response.read(64*1024+1)
        if response.status != 200 or len(raw) > 64*1024 or time.monotonic() > deadline:
            raise ValueError('Callback must acknowledge within 3 seconds with HTTP 200')
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return raw.decode('utf-8', errors='replace')
    finally:
        connection.close()


class Callbacks:
    def __init__(self, transport=post_json):
        self.transport = transport
        # FIFO notifications; this executor never runs generation or model code.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='h3-callback')

    def verify(self, url):
        if url is None:
            return
        if not isinstance(url, str) or not url:
            raise InvalidInput('callback_url', 'Expected a public HTTP(S) callback URL')
        challenge = uuid.uuid4().hex
        started = time.monotonic()
        try:
            response = self.transport(url, dict(challenge=challenge))
            if time.monotonic()-started > 3 or response != dict(challenge=challenge):
                raise ValueError('Invalid challenge acknowledgement')
        except Exception:
            raise InvalidInput('callback_url', 'Callback must return the unchanged challenge within 3 seconds') from None

    def notify(self, url, document):
        if url:
            self.executor.submit(self._send, url, json.loads(json.dumps(dict(task=document))))

    def _send(self, url, body):
        try:
            self.transport(url, body)
        except Exception:
            # No inference retry and no effect on the task's terminal state.
            logger.warning('H3 callback delivery failed for task %s', body['task']['id'])

    def close(self, wait=True):
        self.executor.shutdown(wait=wait, cancel_futures=not wait)
