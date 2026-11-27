"""Application API with lazy imports for CLI configuration and runtime entry points."""
from importlib import import_module

_EXPORTS = {
    "parse_args": ".config",
    "init_env": ".runtime_env",
    "init_cfg_env": ".runtime_env",
    "InferenceManager": ".infer_tools",
    "ImageInferenceManager": ".infer_tools",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
