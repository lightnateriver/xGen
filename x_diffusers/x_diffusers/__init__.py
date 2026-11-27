"""Public xGen API. Importing the package does not initialize model patches."""
from importlib import import_module

_EXPORTS = {
    "parse_args": (".adaptor.config", "parse_args"),
    "init_env": (".adaptor.runtime_env", "init_env"),
    "init_cfg_env": (".adaptor.runtime_env", "init_cfg_env"),
    "InferenceManager": (".adaptor.infer_tools", "InferenceManager"),
    "ImageInferenceManager": (".adaptor.infer_tools", "ImageInferenceManager"),
    "initialize_framework": (".framework", "initialize_framework"),
    "FlowMatchEulerDiscreteSchedulerPusa": (".framework.schedulers", "FlowMatchEulerDiscreteSchedulerPusa"),
    "WanImageToVideoPipelineJoint": (".framework.pipeline.joint", "WanImageToVideoPipelineJoint"),
    "WanVideoToVideoPipeline": (".framework.pipeline.ten_second", "WanVideoToVideoPipeline"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
