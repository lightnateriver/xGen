"""Opt-in H3 MaaS contract; normalizes assets for the existing inference worker."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import re

from maas_fl2va import InvalidInput, _read_json
from maas_media import BYTE_LIMITS, PIXEL_LIMITS, cleanup, materialize, new_directory, enable_image_codecs
from x_diffusers.adaptor.h3_refiner_config import H3_MAX_PROMPT_LENGTH, h3_prompt_length

MAX_BODY_BYTES = 64*1024*1024
RATIOS = ('21:9', '16:9', '4:3', '1:1', '3:4', '9:16')
FIELDS = {'model', 'task_type', 'resolution', 'aspect_ratio', 'duration', 'num_inference_steps',
          'prompt', 'first_frame', 'last_frame', 'ref_path', 'seed', 'enable_audio'}


def validate_startup(config, backend, warmup):
    if (backend != 'cuda' or config.get('model') != 'MiniMax-H3' or config.get('h3_refiner')
            or config.get('task_type') not in ('t2va', 'fl2va', 'ref2va')
            or config.get('h3_attention', 'legacy') != 'legacy' or warmup):
        raise ValueError('--maas requires CUDA H3 Base t2va/fl2va/ref2va without Refiner or legacy warmup')
    if (config.get('service_mode') == 'ref2va') != (config.get('task_type') == 'ref2va'):
        raise ValueError('Ref2VA service mode and model workflow must agree')
    enable_image_codecs()


def output_size(ratio):
    """768P pixel budget, 32-aligned; keep the established landscape preset."""
    if ratio == Fraction(7, 4) or ratio == Fraction(16, 9):
        return 1344, 768
    if ratio == Fraction(9, 16):
        return 768, 1344
    budget = 1344*768
    width = max(32, int(math.sqrt(budget*float(ratio))/32)*32)
    height = max(32, int(math.sqrt(budget/float(ratio))/32)*32)
    return width, height


def normalize(value, config):
    if not isinstance(value, dict) or value.keys()-FIELDS:
        raise InvalidInput('input', 'Expected an object containing only advertised MaaS fields')
    prompt = value.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or '\0' in prompt or h3_prompt_length(prompt)>H3_MAX_PROMPT_LENGTH:
        raise InvalidInput('prompt', 'Expected nonblank text of at most 7000 UTF-16 code units without NUL')
    try:
        prompt.encode('utf-8')
    except UnicodeEncodeError:
        raise InvalidInput('prompt', 'Invalid Unicode text') from None
    task = value.get('task_type', config.get('task_type', 'fl2va'))
    if task not in ('t2va', 'fl2va', 'ref2va'):
        raise InvalidInput('task_type', 'Expected t2va/fl2va/ref2va')
    resident = config.get('service_mode', config.get('task_type', 'fl2va'))
    if (resident == 'ref2va') != (task == 'ref2va'):
        raise InvalidInput('task_type', 'Use a separately deployed Ref2VA worker for reference requests')
    if value.get('resolution', '768p') not in ('768p', '768P'):
        raise InvalidInput('resolution', 'Only 768P output is supported')
    for field, expected in (('model', 'MiniMax-H3'),
                            ('num_inference_steps', config.get('num_inference_steps', 50))):
        if type(value.get(field, expected)) is not type(expected) or value.get(field, expected) != expected:
            raise InvalidInput(field, f'Expected {expected}')
    duration = value.get('duration', 5)
    seed = value.get('seed', 42)
    audio = value.get('enable_audio', True)
    if type(duration) is not int or not 5 <= duration <= 15:
        raise InvalidInput('duration', 'Expected an integer from 5 to 15')
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise InvalidInput('seed', 'Expected an integer from 0 to 2147483647')
    if type(audio) is not bool:
        raise InvalidInput('enable_audio', 'Expected a boolean')
    first, last = value.get('first_frame'), value.get('last_frame')
    refs = value.get('ref_path', [])
    if not isinstance(refs, list) or len(refs)>12 or any(not isinstance(x, str) or not x for x in refs):
        raise InvalidInput('ref_path', 'Expected at most 12 media URLs/data URIs')
    if any(key in value and (not isinstance(value[key], str) or not value[key]) for key in ('first_frame', 'last_frame')):
        raise InvalidInput('first_frame', 'Frame fields must be nonempty image URLs/data URIs')
    if ((task == 't2va' and (first or last or refs)) or
            (task == 'fl2va' and (not (first or last) or refs)) or
            (task == 'ref2va' and (first or last or not refs))):
        raise InvalidInput('input', 'T2VA, first/last-frame I2VA and reference input modes cannot be mixed')
    ratio = value.get('aspect_ratio', '16:9' if task == 't2va' else 'adaptive')
    if ratio not in RATIOS+('adaptive',) or (task == 't2va' and ratio == 'adaptive'):
        raise InvalidInput('aspect_ratio', 'Expected an advertised ratio; T2VA requires a concrete ratio')
    directory = new_directory()
    result = dict(model='MiniMax-H3', task_type=task, resolution='768p', prompt=prompt,
                  duration=duration, seed=seed, enable_audio=audio,
                  num_inference_steps=config.get('num_inference_steps', 50), _maas_dir=str(directory))
    try:
        frames = {}
        for field, source in (('first_frame', first), ('last_frame', last)):
            if source:
                frames[field] = materialize(source, directory, field, frame=True)
                result[field] = frames[field]['path']
        media = [materialize(source, directory, str(i)) for i, source in enumerate(refs)]
        counts = Counter(m['kind'] for m in media)
        if any(counts[k]>n for k,n in (('image',9),('video',3),('audio',3))):
            raise InvalidInput('ref_path', 'At most 9 images, 3 videos and 3 audio clips')
        if media and not (counts['image'] or counts['video']):
            raise InvalidInput('ref_path', 'Audio references require an image or video')
        for kind in ('video', 'audio'):
            if sum((m['seconds'] for m in media if m['kind']==kind), Fraction())>15:
                raise InvalidInput('ref_path', f'Combined {kind} duration exceeds 15 seconds')
        result['ref_path'] = [m['path'] for m in media]
        if task == 'fl2va' or ratio == 'adaptive':
            visual = next(iter(frames.values()), None) or next(m for m in media if m['kind']!='audio')
            resolved = Fraction(visual['width'], visual['height'])
            ratio = 'adaptive'
        else:
            a,b = ratio.split(':'); resolved = Fraction(int(a),int(b))
        result['aspect_ratio'] = ratio
        result['_maas_size'] = output_size(resolved)
        signature = {k:v for k,v in result.items() if k not in ('_maas_dir','first_frame','last_frame','ref_path')}
        result['_maas_fingerprint'] = hashlib.sha256(json.dumps(dict(
            request=signature, frames={k:m['sha256'] for k,m in frames.items()},
            references=[m['sha256'] for m in media]),
            sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        return result
    except Exception:
        cleanup(result)
        raise


def read_submission(request, config, *, internal=False):
    body = _read_json(request, MAX_BODY_BYTES)
    if not internal:
        return normalize(body, config)
    if not isinstance(body, dict) or body.keys()-{'id','input','created_at'}:
        raise InvalidInput('body', 'Expected id, input and optional created_at')
    identifier = body.get('id')
    if not isinstance(identifier, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', identifier) is None:
        raise InvalidInput('id', 'Expected a valid task ID of 1..128 ASCII characters')
    created = body.get('created_at')
    if 'created_at' in body and (type(created) is not int or not 0<created<4102444800000):
        raise InvalidInput('created_at', 'Expected a positive Unix millisecond timestamp before 2100')
    return identifier, created, normalize(body.get('input'), config)


def parameter_spec(config):
    return dict(profile='maas-h3-base-768p', max_body_bytes=MAX_BODY_BYTES, parameters=dict(
        model=dict(allowed_values=['MiniMax-H3']), resolution=dict(allowed_values=['768p','768P']),
        task_type=dict(allowed_values=['ref2va'] if config.get('service_mode', config.get('task_type'))=='ref2va' else ['t2va','fl2va']),
        prompt=dict(required=True,max_length=7000,length_unit='utf16_code_units'),
        duration=dict(min=5,max=15,type='integer'), aspect_ratio=dict(allowed_values=list(RATIOS)+['adaptive']),
        num_inference_steps=dict(allowed_values=[config.get('num_inference_steps',50)]),
        media=dict(byte_limits=BYTE_LIMITS,pixel_limits=PIXEL_LIMITS,min_edge=256,max_edge=5760,
                   min_aspect_ratio=.4,max_aspect_ratio=2.5,max_images=9,max_videos=3,max_audio=3,
                   max_files=12,clip_seconds=[2,15],max_seconds_per_media_type=15),
        seed=dict(min=0,max=2**31-1),enable_audio=dict(type='boolean',default=True)))
