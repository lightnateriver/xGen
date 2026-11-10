# SPDX-License-Identifier: Apache-2.0
"""Bound decoded temporal storage while retaining the native H3 decode loop."""

from functools import wraps

import torch


def rgb8_to_pil(rgb):
    """Match VideoProcessor's non-normalizing PIL path before the D2H copy."""
    from PIL import Image

    pixels=(rgb*255).round().to(torch.uint8).permute(0,2,3,4,1).contiguous().cpu().numpy()
    return [[Image.fromarray(frame) for frame in batch] for batch in pixels]


def prepare_decoder_linears(decoder):
    """Store Linear weights as their FP16 autocast operands; retain other dtypes."""
    count=0
    for layer in decoder.modules():
        if isinstance(layer,torch.nn.Linear):
            layer.to(dtype=torch.float16)
            count+=1
    return count


def install_staged_decoder(vae,device,retain_module):
    """Load the decode half once per video, with native autocast weight rounding."""
    cpu=torch.device('cpu')
    retain_module(vae.decoder,cpu)
    retain_module(vae.post_quant_conv,cpu)
    prepare_decoder_linears(vae.decoder)
    from diffusers.hooks.group_offloading import ModuleGroup

    group=ModuleGroup(modules=[vae.decoder,vae.post_quant_conv],offload_device=cpu,
                      onload_device=torch.device(device),offload_leader=vae.decoder,
                      stream=torch.cuda.Stream(device=device),low_cpu_mem_usage=False)
    group.offload_()
    vae._h3_decoder_group=group
    native_decode=vae.decode

    @wraps(native_decode)
    def decode(*values,**kwargs):
        import time
        started=time.perf_counter()
        group.onload_()
        torch.cuda.current_stream(device).wait_stream(group.stream)
        group.stream.synchronize()
        loaded=time.perf_counter()
        try:
            return native_decode(*values,**kwargs)
        finally:
            unloading=time.perf_counter()
            group.offload_()
            vae._h3_decoder_staging={'h2d_seconds':loaded-started,
                                     'd2h_seconds':time.perf_counter()-unloading}
    vae.decode=decode


def install_cpu_decode_chunks(vae):
    native_decode=vae._decode

    @wraps(native_decode)
    def decode(z):
        native_clip,native_blend=vae._decode_clip,vae._blend
        clips=0
        copied_bytes=0

        def clip(*values,**kwargs):
            nonlocal clips,copied_bytes
            output=native_clip(*values,**kwargs)
            clips+=1
            copied_bytes+=output.numel()*output.element_size()
            return output.to('cpu')

        def blend(a,b,*values,**kwargs):
            # Spatial tile stitching still operates on CUDA tensors. Only the
            # native temporal loop now sees CPU clips; perform its blend on the
            # original device/dtype so its arithmetic and rounding stay intact.
            if a.device.type=='cpu' and b.device.type=='cpu':
                return native_blend(a.to(z.device),b.to(z.device),*values,**kwargs).to('cpu')
            return native_blend(a,b,*values,**kwargs)

        vae._decode_clip,vae._blend=clip,blend
        try:
            return native_decode(z)
        finally:
            vae._decode_clip,vae._blend=native_clip,native_blend
            vae._h3_cpu_decode_stats={'clips':clips,'copied_bytes':copied_bytes}

    vae._decode=decode
