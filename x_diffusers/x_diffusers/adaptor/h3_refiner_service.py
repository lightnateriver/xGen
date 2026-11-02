"""Small H3 additions to the existing service validation/configuration; no routes."""
from .h3_refiner_config import profile


def apply_startup_settings(config, args, argv):
    """CLI overrides only explicitly supplied new fields; never erase JSON asset paths."""
    config['h3_refiner'] = True
    for name in ('h3_refiner_weights','h3_refiner_prepared','h3_refiner_loading','h3_refiner_seconds'):
        explicit = any(value.split('=',1)[0] == '--'+name for value in argv)
        if explicit or name not in config:
            config[name] = getattr(args,name)


def parameter_spec(config):
    return dict(experimental=True,backend='h3_regenerate',parameters={
        'prompt':dict(type='string',required=True),
        'task_type':dict(type='string',allowed_values=[config.get('task_type','t2va')]),
        'duration':dict(type='integer',allowed_values=[5,10,15],default=config.get('h3_refiner_seconds',5)),
        'resolution':dict(type='string',allowed_values=['2k'],default='2k'),
        'aspect_ratio':dict(type='string',allowed_values=['16:9'],default='16:9'),
        'num_inference_steps':dict(type='integer',allowed_values=[config.get('num_inference_steps',51)]),
        'first_frame':dict(type='string',required=config.get('task_type')=='fl2va'),
        'ref_path':dict(type='array',count=5,required=config.get('task_type')=='ref2va'),
        'enable_audio':dict(type='boolean',default=True),
        'seed':dict(type='integer',min=0,max=2**63-1)},
        example=dict(prompt='A small robot walks through a sunlit garden.',duration=5,resolution='2k',seed=42))


def configure_args(config, data, args):
    # Startup settings cannot be supplied by a remote request.
    for key in ('h3_refiner', 'h3_refiner_weights', 'h3_refiner_prepared', 'h3_refiner_loading'):
        setattr(args, key, config.get(key, getattr(args, key, None)))
    args.h3_refiner_seconds = data.get('duration', config.get('h3_refiner_seconds', 5))
    spec = profile(args.h3_refiner_seconds)
    args.frames = spec['generation_frames']
    args.width, args.height, args.save_fps = 2560, 1440, 24
    args.num_inference_steps = data.get('num_inference_steps', config.get('num_inference_steps', 51))
    args.adopt_sr = False


def validate_request(service, data, errors, warnings):
    """Reuse existing prompt/media validators; add only the enabled feature's bounds."""
    prompt = service._validate_prompt(data, errors)
    task = data.get('task_type', service.default_config.get('task_type', 't2va'))
    if task != service.default_config.get('task_type', 't2va'):
        errors.append('Experimental H3 Refiner fixes task_type at startup; use a separate worker for another workflow')
    if task not in ('t2va', 'fl2va', 'ref2va'):
        errors.append('task_type must be t2va/fl2va/ref2va')
    first, last = service._validate_mmh3_frame_fields(data, task, errors)
    refs = service._validate_mmh3_ref_path(data, task, errors)
    if task == 'ref2va' and (not isinstance(refs, list) or len(refs) != 5):
        errors.append('Experimental ref2va requires exactly five images')
    if task == 'fl2va' and (not first or last):
        errors.append('Experimental fl2va accepts one first-frame image only')
    duration = data.get('duration', service.default_config.get('h3_refiner_seconds', 5))
    try:
        spec = profile(duration)
    except ValueError as exc:
        errors.append(str(exc)); spec = profile(5)
    if data.get('resolution', '2k') != '2k' or data.get('aspect_ratio', '16:9') != '16:9':
        errors.append('Experimental H3 Refiner requires resolution=2k and aspect_ratio=16:9')
    steps = data.get('num_inference_steps', service.default_config.get('num_inference_steps', 51))
    if type(steps) is not int or steps != service.default_config.get('num_inference_steps', 51):
        errors.append('num_inference_steps is fixed at startup and must match Base AdaLN assets')
    seed = data.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        errors.append('seed must be an integer in [0, 2**63)')
    if data.get('adopt_sr', False) is not False:
        errors.append('SeedVR2 adopt_sr cannot be combined with H3 Refiner')
    audio = data.get('enable_audio', True)
    if type(audio) is not bool:
        errors.append('enable_audio must be boolean')
    if any(key.startswith('h3_refiner') for key in data):
        errors.append('h3_refiner settings are startup-only; restart to change or disable the feature')
    if errors:
        return False, dict(errors=errors, warnings=warnings)
    validated = dict(model='MiniMax-H3', task_type=task, prompt=prompt, resolution='2k',aspect_ratio='16:9',
                     duration=duration,frames=spec['generation_frames'],num_inference_steps=steps,
                     seed=seed,save_fps=24,first_frame=first,last_frame=last,ref_path=refs,
                     enable_audio=audio,adopt_sr=False,attention_sparse=False)
    if data.get('save_path'):
        validated['save_path'] = data['save_path']
    return True, dict(data=validated,warnings=warnings+['experimental H3 Refiner; no automatic retry'])
