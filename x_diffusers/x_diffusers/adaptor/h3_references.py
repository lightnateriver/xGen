"""Local Ref2VA media limits and ordered loading shared by Base and Refiner."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import subprocess


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".ogg"}
LIMITS = {"image": 9, "video": 3, "audio": 3}
MAX_FILES = 12
MIN_SECONDS = 2
MAX_SECONDS = 15


def media_kind(path):
    suffix = Path(path).suffix.lower()
    for kind, extensions in (("image", IMAGE_EXTS), ("video", VIDEO_EXTS), ("audio", AUDIO_EXTS)):
        if suffix in extensions:
            return kind
    raise ValueError(f"Unsupported Ref2VA reference type: {suffix or '(no extension)'}")


def probe_clip(path, kind):
    try:
        result = subprocess.run(
            [os.environ.get("FFPROBE_PATH", "ffprobe"), "-v", "error", "-show_streams",
             "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        streams = [stream for stream in info.get("streams", ())
                   if not stream.get("disposition", {}).get("attached_pic")]
        stream = next(stream for stream in streams if stream.get("codec_type") == kind)
        duration = stream.get("duration")
        if duration in (None, "N/A"):
            duration = info.get("format", {}).get("duration")
        seconds = Decimal(str(duration))
        if not seconds.is_finite():
            raise ValueError("Nonfinite stream duration")
    except (OSError, subprocess.SubprocessError, ValueError, InvalidOperation, StopIteration) as exc:
        raise ValueError(f"Could not probe {kind} reference {Path(path).name}: {exc}") from exc
    return seconds, any(stream.get("codec_type") == "audio" for stream in streams)


def validate_references(paths):
    """Check local media before model loading. A video soundtrack is part of that file."""
    if not isinstance(paths, (list, tuple)) or not paths or any(not isinstance(p, str) or not p for p in paths):
        raise ValueError("Ref2VA requires a nonempty list of reference file paths")
    kinds = [media_kind(path) for path in paths]
    counts = Counter(kinds)
    if len(paths) > MAX_FILES:
        raise ValueError(f"Ref2VA accepts at most {MAX_FILES} reference files total")
    for kind, maximum in LIMITS.items():
        if counts[kind] > maximum:
            raise ValueError(f"Ref2VA accepts at most {maximum} {kind} references")
    if not (counts["image"] or counts["video"]):
        raise ValueError("Audio references require at least one image or video reference")
    files = [Path(path).resolve() for path in paths]
    if len(set(files)) != len(files):
        raise ValueError("Ref2VA reference file paths must be distinct")

    has_audio = False
    totals = Counter()
    for path, kind in zip(files, kinds):
        if not path.is_file():
            raise ValueError(f"Ref2VA reference file does not exist: {path.name}")
        if kind == "image":
            from PIL import Image

            with Image.open(path) as image:
                if min(image.size) < 1 or image.width * image.height > 4096 * 4096:
                    raise ValueError("Reference images must contain 1..4096*4096 pixels")
                image.verify()
            continue
        seconds, audio = probe_clip(path, kind)
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"Each {kind} reference must be {MIN_SECONDS}–{MAX_SECONDS} seconds: {path.name}")
        totals[kind] += seconds
        if totals[kind] > MAX_SECONDS:
            raise ValueError(f"Combined {kind} reference duration must be at most {MAX_SECONDS} seconds")
        has_audio |= audio
    return "both" if has_audio else "video"


def load_references(paths):
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference,
    )

    loaders = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference,
               "audio": MiniMaxH3AudioReference}
    return [loaders[media_kind(path)].from_file(path) for path in paths]


def reference_conditioning(args):
    paths = getattr(args, "reference_paths", None)
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Ref2VA requires a nonempty list of reference file paths")
    paths = tuple(paths)
    cached = getattr(args, "_h3_reference_conditioning", None)
    if cached is None or cached[0] != paths:
        cached = (paths, validate_references(paths))
        args._h3_reference_conditioning = cached
    return cached[1]


def select_reference_adaln(pipe, args):
    select_adaln(pipe, args, reference_conditioning(args))


def select_adaln(pipe, args, conditioning):
    cache = getattr(pipe.transformer, "_adaln_cache", None)
    if cache is not None and cache._conditioning_type != conditioning:
        from x_base.adaln_cache import install_adaln_cache

        install_adaln_cache(pipe, args.adaln_cache_file, conditioning_type=conditioning)
