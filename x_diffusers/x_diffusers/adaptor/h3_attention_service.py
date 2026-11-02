"""Optional validation/configuration for the original HTTP service; no routes."""
from .h3_attention_config import key, validate


def apply_startup_settings(config, args, argv):
    for name in ('h3_attention','h3_veda_predictor'):
        if any(value.split('=',1)[0]=='--'+name for value in argv):
            config[name]=getattr(args,name)


def validate_warmup(config, enabled):
    if enabled and config.get('h3_attention','legacy')!='legacy':
        raise ValueError('Sparse H3 attention requires a full 50-step first request, not reduced-step warmup')


def validate_request(config, data, errors):
    for name in ('h3_attention','h3_veda_predictor'):
        if name in data:
            errors.append(name+' is a startup setting; restart to change it')
    if type(data.get('duration',5)) is not int or data.get('duration',5)!=5:
        errors.append('Initial sparse H3 attention requires duration=5')
    if data.get('task_type',config.get('task_type','t2va'))!='t2va':
        errors.append('Initial sparse H3 attention requires t2va')
    steps=data.get('num_inference_steps',config.get('num_inference_steps',51))
    if type(steps) is not int or steps!=51:
        errors.append('Initial sparse H3 attention requires 51 schedule points / 50 actual steps')
    resolution='2k' if config.get('h3_refiner',False) else '768p'
    if data.get('resolution',resolution)!=resolution or data.get('aspect_ratio','16:9')!='16:9':
        errors.append('Sparse H3 resolution and landscape aspect ratio are fixed at startup')
    if any(data.get(name) for name in ('first_frame','last_frame','ref_path','adopt_sr',
                                        'frame_interpolation','attention_sparse')):
        errors.append('Sparse H3 does not accept additional media, attention modes or super-resolution plugins')


def configure_args(config, data, args):
    errors=[];validate_request(config,data,errors)
    if errors:
        raise ValueError('; '.join(errors))
    args.h3_attention=config['h3_attention']
    args.h3_veda_predictor=config.get('h3_veda_predictor')
    args.frames=124
    args.num_inference_steps=data.get('num_inference_steps',config.get('num_inference_steps',51))
    validate(args)


def validate_manager(manager, args):
    validate(args)
    if manager.init_args is not None and key(manager.init_args)!=key(args):
        raise ValueError('Restart the service to change H3 attention backend or predictor')


def parameter_spec(config):
    return dict(experimental=True,backend=config['h3_attention'],scope='Base DiT only',parameters={
        'prompt':dict(type='string',required=True),
        'task_type':dict(type='string',allowed_values=['t2va']),
        'duration':dict(type='integer',allowed_values=[5],default=5),
        'resolution':dict(type='string',allowed_values=['2k' if config.get('h3_refiner',False) else '768p']),
        'aspect_ratio':dict(type='string',allowed_values=['16:9']),
        'num_inference_steps':dict(type='integer',allowed_values=[51]),
        'seed':dict(type='integer',min=0,max=2**31-1),
        'enable_audio':dict(type='boolean',default=True)})
