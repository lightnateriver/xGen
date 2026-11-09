"""Opt-in request contract for the CUDA H3 FL2VA MaaS profile."""

import base64
import binascii
import io
import json
import re

from PIL import Image
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from x_diffusers.adaptor.h3_refiner_config import H3_MAX_PROMPT_LENGTH, h3_prompt_length


MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 2048
MAX_PROMPT_LENGTH = H3_MAX_PROMPT_LENGTH
MAX_SEED = 2**31 - 1
MAX_ENCODED_IMAGE_BYTES = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
# A UTF-16 code unit can occupy six bytes as a JSON Unicode escape.
MAX_BODY_BYTES = MAX_ENCODED_IMAGE_BYTES + MAX_PROMPT_LENGTH * 6 + 32 * 1024
FIXED_FIELDS = {
    "model": "MiniMax-H3", "task_type": "fl2va", "resolution": "768p",
    "aspect_ratio": "16:9", "duration": 5, "num_inference_steps": 50,
}
INPUT_FIELDS = FIXED_FIELDS.keys() | {"prompt", "first_frame", "seed", "enable_audio"}
IMAGE_HEADERS = {"data:image/jpeg;base64": "JPEG", "data:image/png;base64": "PNG"}


class InvalidInput(ValueError):
    def __init__(self, field, message, status=400):
        super().__init__(message)
        self.field = field
        self.status = status


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidInput("body", "Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(_value):
    raise InvalidInput("body", "Non-finite JSON number")


def _read_json(request, max_body_bytes=MAX_BODY_BYTES):
    if not request.is_json:
        raise InvalidInput("body", "Expected a JSON content type", 415)
    if request.content_length is not None and request.content_length > max_body_bytes:
        raise InvalidInput("body", "Request body exceeds the byte limit", 413)
    try:
        body = request.stream.read(max_body_bytes + 1)
    except RequestEntityTooLarge:
        raise InvalidInput("body", "Request body exceeds the byte limit", 413) from None
    except BadRequest:
        raise InvalidInput("body", "Could not read the request body") from None
    if len(body) > max_body_bytes:
        raise InvalidInput("body", "Request body exceeds the byte limit", 413)
    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except InvalidInput:
        raise
    except (ValueError, RecursionError):
        raise InvalidInput("body", "Expected valid UTF-8 JSON") from None


def _validate_first_frame(value):
    field = "first_frame"
    if not isinstance(value, str):
        raise InvalidInput(field, "Expected an inline JPEG or PNG data URI")
    header, _, encoded = value.partition(",")
    image_format = IMAGE_HEADERS.get(header)
    if image_format is None or not encoded:
        raise InvalidInput(field, "Expected an inline JPEG or PNG data URI")
    if len(encoded) > MAX_ENCODED_IMAGE_BYTES:
        raise InvalidInput(field, "Image exceeds the byte limit", 413)
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise InvalidInput(field, "Invalid base64 image") from None
    if len(decoded) > MAX_IMAGE_BYTES:
        raise InvalidInput(field, "Image exceeds the byte limit", 413)
    try:
        with Image.open(io.BytesIO(decoded), formats=[image_format]) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS or max(width, height) > 4096:
                raise InvalidInput(field, "Image exceeds the decoded pixel or edge limit", 413)
            if min(width, height) < 32 or max(width, height) > 4 * min(width, height):
                raise InvalidInput(field, "Image is too small or its aspect ratio exceeds 4:1")
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidInput(field, "Expected a still image")
            image.verify()
        with Image.open(io.BytesIO(decoded), formats=[image_format]) as image:
            image.load()
    except InvalidInput:
        raise
    except Image.DecompressionBombError:
        raise InvalidInput(field, "Image exceeds the decoded pixel limit", 413) from None
    except (OSError, ValueError, SyntaxError):
        raise InvalidInput(field, "Image is corrupt or does not match its declared format") from None


def _validate_input(value):
    if not isinstance(value, dict):
        raise InvalidInput("input", "Expected a JSON object")
    if value.keys() - INPUT_FIELDS:
        raise InvalidInput("input", "Unsupported input field")
    for field, expected in FIXED_FIELDS.items():
        actual = value.get(field, expected)
        if type(actual) is not type(expected) or actual != expected:
            raise InvalidInput(field, f"Expected {expected}")
    prompt = value.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or h3_prompt_length(prompt) > MAX_PROMPT_LENGTH or "\x00" in prompt:
        raise InvalidInput("prompt", f"Expected 1 to {MAX_PROMPT_LENGTH} UTF-16 code units of non-blank text without NUL")
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidInput("prompt", "Invalid Unicode text") from None
    seed = value.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise InvalidInput("seed", "Expected an integer from 0 to 2147483647")
    enable_audio = value.get("enable_audio", True)
    if type(enable_audio) is not bool:
        raise InvalidInput("enable_audio", "Expected a boolean")
    first_frame = value.get("first_frame")
    _validate_first_frame(first_frame)
    return dict(FIXED_FIELDS, prompt=prompt, first_frame=first_frame, seed=seed, enable_audio=enable_audio)


def read_submission(request, *, internal=False):
    body = _read_json(request)
    if not internal:
        return _validate_input(body)
    if not isinstance(body, dict) or body.keys() - {"id", "input", "created_at"}:
        raise InvalidInput("body", "Expected an object with id, input and optional created_at")
    task_id = body.get("id")
    if not isinstance(task_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id) is None:
        raise InvalidInput("id", "Expected 1 to 128 ASCII letters, digits, underscores or hyphens starting with a letter or digit")
    created_at = body.get("created_at")
    if "created_at" in body and (type(created_at) is not int or not 0 < created_at < 4102444800000):
        raise InvalidInput("created_at", "Expected a positive Unix timestamp in milliseconds before 2100")
    return task_id, created_at, _validate_input(body.get("input"))


def validate_startup(config, backend, warmup):
    if (backend != "cuda" or config.get("model") != FIXED_FIELDS["model"]
            or config.get("task_type") != FIXED_FIELDS["task_type"]
            or config.get("service_mode") == "ref2va"
            or config.get("h3_refiner") or config.get("h3_attention", "legacy") != "legacy"
            or config.get("num_inference_steps") != FIXED_FIELDS["num_inference_steps"] or warmup):
        raise ValueError("MaaS FL2VA input profile requires CUDA H3 Base FL2VA with 51 schedule points and no legacy warmup")


def parameter_spec():
    fields = {
        key: {"type": "integer" if type(value) is int else "string", "default": value, "allowed_values": [value]}
        for key, value in FIXED_FIELDS.items()
    }
    fields.update(
        prompt={"type": "string", "required": True, "max_length": MAX_PROMPT_LENGTH,
                "length_unit": "utf16_code_units"},
        first_frame={"type": "string", "required": True, "mime_types": ["image/jpeg", "image/png"],
                     "max_bytes": MAX_IMAGE_BYTES, "max_pixels": MAX_IMAGE_PIXELS,
                     "min_edge": 32, "max_edge": 4096, "max_aspect_ratio": 4},
        seed={"type": "integer", "default": 42, "min": 0, "max": MAX_SEED},
        enable_audio={"type": "boolean", "default": True},
    )
    return {"profile": "maas-fl2va-768p-5s", "parameters": fields, "max_body_bytes": MAX_BODY_BYTES}
