"""Seven-day V2 metadata and results; never stores weights or inline media."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
import time


STATUSES = ('queued', 'running', 'succeeded', 'failed', 'cancelled')
RETENTION_S = 7 * 24 * 3600


class TaskHistory:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.initialized = False
        self.lease = None

    def close(self):
        with self.lock:
            if self.lease is not None:
                self.lease.close()
                self.lease = None

    def _lease(self):
        if self.lease is not None:
            return
        lease = (self.directory/'owner.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if lease.seek(0, 2) == 0:
                    lease.write(b'0'); lease.flush()
                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lease.close()
            raise RuntimeError('V2 history directory is already in use; configure a unique --maas-state-dir') from None
        self.lease = lease

    @contextmanager
    def connection(self):
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._lease()
            connection = sqlite3.connect(self.directory / 'tasks.sqlite3', timeout=5)
            try:
                connection.execute('CREATE TABLE IF NOT EXISTS tasks '
                                   '(id TEXT PRIMARY KEY, created REAL NOT NULL, document TEXT NOT NULL, metadata TEXT NOT NULL)')
                connection.execute('DELETE FROM tasks WHERE created <= ?', (time.time()-RETENTION_S,))
                if not self.initialized:
                    # A restarted worker cannot resume unfinished inference. No retry.
                    for identifier, raw in connection.execute('SELECT id, document FROM tasks').fetchall():
                        document = json.loads(raw)
                        if document['status'] in ('queued', 'running'):
                            document.update(status='failed', updated_at=int(time.time()),
                                            error=dict(code='engine_restarted', message='Engine restarted before completion'))
                            connection.execute('UPDATE tasks SET document=? WHERE id=?', (json.dumps(document), identifier))
                    self.initialized = True
                yield connection
                connection.commit()
            finally:
                connection.close()

    def put(self, document, metadata):
        with self.connection() as connection:
            connection.execute('INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?)',
                (document['id'], document['created_at'], json.dumps(document), json.dumps(metadata)))

    def get(self, identifier):
        with self.connection() as connection:
            row = connection.execute('SELECT document, metadata FROM tasks WHERE id=?', (identifier,)).fetchone()
        return (json.loads(row[0]), json.loads(row[1])) if row else None

    def delete(self, identifier):
        with self.connection() as connection:
            connection.execute('DELETE FROM tasks WHERE id=?', (identifier,))

    def list(self, filters, page, size):
        with self.connection() as connection:
            rows = [(json.loads(row[0]), json.loads(row[1])) for row in connection.execute('SELECT document, metadata FROM tasks ORDER BY created DESC, id DESC')]
        documents = [doc for doc, _ in rows]
        metadata = {doc['id']: meta for doc, meta in rows}
        for field, value in filters.items():
            if field == 'task_ids':
                documents = [doc for doc in documents if doc['id'] in value]
            else:
                documents = [doc for doc in documents if doc.get(field) == value]
        result = dict(items=documents[(page-1)*size:page*size], total=len(documents))
        return result, {doc['id']: metadata[doc['id']] for doc in result['items']}
