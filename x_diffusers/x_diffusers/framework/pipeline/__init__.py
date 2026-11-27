"""Pipeline exports; registration is owned by initialize_framework()."""
from importlib import import_module

_EXPORTS = {
    "WanImageToVideoPipelineJoint": ".joint",
    "WanVideoToVideoPipeline": ".ten_second",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
