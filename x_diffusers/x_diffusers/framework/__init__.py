"""Explicit, ordered registration of Diffusers model adaptations."""
from importlib import import_module
from threading import RLock

# Preserve the established registration order. Modules are cached by Python;
# successful initialization is also guarded against repeated/concurrent calls.
_REGISTRATION_MODULES = (
    ".transformer.attention_backend",
    ".lora.lora",
    ".vae.hunyuan", ".vae.wan", ".vae.cogvideox",
    ".transformer.transformer_z_image",
    ".transformer.cogvideox", ".transformer.hunyuan", ".transformer.wan",
    ".transformer.wan_vace", ".transformer.minimax_h3", ".vae.minimax_h3",
    ".schedulers", ".pipeline.pipeline", ".pipeline.joint", ".pipeline.ten_second",
    ".pipeline.longcat_image", ".pipeline.z_image", ".pipeline.qwenimage",
)
_initialized = False
_initialization_lock = RLock()


def initialize_framework():
    """Register patches and pipeline mappings once, before building models."""
    global _initialized
    with _initialization_lock:
        if _initialized:
            return
        for module in _REGISTRATION_MODULES:
            import_module(module, __name__)
        _initialized = True
