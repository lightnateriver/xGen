"""Prepared-model integrity and offline refresh validation (CPU only)."""
import hashlib
import json
from pathlib import Path


def validate_base_schedule(steps, metadata=None, *, turbo8=False):
    allowed = (8,) if turbo8 else (20, 50)
    if type(steps) is not int or steps not in allowed:
        raise ValueError("Resident Base schedule does not match the selected inference profile")
    if metadata is not None and (int(metadata.get("num_steps", "-1")) != steps + 1
                                 or int(metadata.get("num_eval_steps", steps)) != steps):
        raise ValueError("Base AdaLN cache evaluation count differs from the configured schedule")
    return steps


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_digests(folder):
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file()
                   and p.suffix in (".json", ".safetensors", ".bin"))
    if not (folder / "config.json").is_file() or not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError(f"Missing model config or weights: {folder}")
    return {p.relative_to(folder).as_posix(): digest(p) for p in files}


def prepared_cache(folder):
    folder = Path(folder)
    record = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    name = record.get("cache_file", "adaln_cache.safetensors")
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Invalid prepared cache filename")
    cache = folder / name
    if digest(cache) != record["cache_sha256"]:
        raise ValueError("Prepared AdaLN cache checksum mismatch")
    return cache


def validate_refresh(output, fingerprint, base_digests):
    output = Path(output)
    prepared_cache(output)
    record = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if record.get("source_base_sha256") != base_digests:
        raise ValueError("Base weights changed or legacy manifest; prepare into a new directory")
    if record.get("adapters_sha256") != fingerprint:
        raise ValueError("Adapter weights changed; prepare into a new directory")
    if record.get("transformer_sha256") != model_digests(output / "transformer"):
        raise ValueError("Prepared transformer changed; prepare into a new directory")
    return record
