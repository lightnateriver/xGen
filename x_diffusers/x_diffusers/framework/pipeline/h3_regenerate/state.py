"""Small CPU-only helpers shared by the HTTP service and GPU worker."""
import json
from pathlib import Path


def atomic_json(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temp.replace(path)
