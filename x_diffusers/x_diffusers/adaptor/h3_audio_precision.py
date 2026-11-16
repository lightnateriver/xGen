"""Keep H3's DAC/BigVGAN audio VAE in FP32 independently of the DiT recipe."""
from functools import wraps

import torch


def prepare_audio_vae(audio):
    """Preserve FP32 weights and disable surrounding autocast for audio calls."""
    audio.float().eval()

    def protect(method):
        @wraps(method)
        def call(*args, **kwargs):
            def fp32(value):
                return value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value

            device_type = next(audio.parameters()).device.type
            with torch.autocast(device_type=device_type, enabled=False):
                return method(*(fp32(value) for value in args),
                              **{name: fp32(value) for name, value in kwargs.items()})

        return call

    audio.encode = protect(audio.encode)
    audio.decode = protect(audio.decode)
