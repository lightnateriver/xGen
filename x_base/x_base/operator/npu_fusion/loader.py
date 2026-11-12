"""One process-wide load per resolved library path; no device initialization."""
from pathlib import Path
import threading

import torch

_loaded = set()
_lock = threading.Lock()


def load_library(path):
    path = str(Path(path).resolve())
    with _lock:
        if path not in _loaded:
            torch.ops.load_library(path)
            _loaded.add(path)


def loaded_libraries():
    with _lock:
        return tuple(sorted(_loaded))
