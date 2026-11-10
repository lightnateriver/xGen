"""Optional whole-module staging around H3 reference conditioning.

Use the existing module device contract; do not rebuild or requantize weights.
On encoder failure keep DiT on CPU and preserve the original error for the caller.
"""
from functools import wraps
import gc
import logging
import time
import torch

logger=logging.getLogger(__name__)


def install_conditioning_staging(pipe,device,*,defer_until_dit=False):
    accelerator=torch.get_device_module(torch.device(device).type)
    model=pipe.transformer
    encoder=pipe.text_encoder.model
    if getattr(encoder,'_h3_conditioning_staging_installed',False):
        raise ValueError('H3 conditioning staging is already installed')
    metrics=dict(calls=0,d2h_seconds=0.,h2d_seconds=0.,last=None)
    pipe._h3_cuda_conditioning_staging=metrics
    original=encoder.forward

    def restore(event):
        accelerator.synchronize();gc.collect();accelerator.empty_cache()
        event['status']='restoring'
        started=time.perf_counter();model.to(device);accelerator.synchronize()
        event['h2d_seconds']=time.perf_counter()-started
        metrics['h2d_seconds']+=event['h2d_seconds']
        event.update(status='completed',allocated_after=accelerator.memory_allocated())
        logger.info('H3 conditioning staging (weights preserved, no reload/quantization): %s',event)

    if defer_until_dit:
        def before_dit(module,inputs):
            event=metrics['last']
            if event is not None and event['status']=='awaiting_reference_encoding':
                restore(event)
        model.register_forward_pre_hook(before_dit)

    @wraps(original)
    def forward(*values,**kwargs):
        event=dict(status='offloading',allocated_before=accelerator.memory_allocated())
        metrics['last']=event;metrics['calls']+=1
        accelerator.synchronize()
        started=time.perf_counter();model.to('cpu');accelerator.synchronize()
        event['d2h_seconds']=time.perf_counter()-started
        metrics['d2h_seconds']+=event['d2h_seconds']
        gc.collect();accelerator.empty_cache()
        event.update(status='encoding',allocated_during_encoding=accelerator.memory_allocated())
        try:
            output=original(*values,**kwargs)
        except BaseException:
            event['status']='encoder_failed_dit_on_cpu'
            logger.exception('H3 conditioning failed; DiT remains on CPU, original failure propagated')
            raise
        if defer_until_dit:
            # ModularPipeline must still infer CUDA from its existing group
            # offload hooks, so reference pixels and request noise stay on CUDA.
            if torch.device(pipe._execution_device).type!=torch.device(device).type:
                event['status']='invalid_execution_device_dit_on_cpu'
                raise RuntimeError('Deferred H3 reference staging requires the pipeline execution device to match the accelerator')
            event.update(status='awaiting_reference_encoding',restore_boundary='before_first_dit')
            logger.info('H3 reference staging: DiT remains on CPU through reference encoding')
        else:
            restore(event)
        return output

    encoder.forward=forward
    encoder._h3_conditioning_staging_installed=True
