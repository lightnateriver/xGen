"""Optional reference component parallelism; native H3 inference scheduling stays in charge."""
from contextlib import contextmanager
from functools import wraps
import logging
import math
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@contextmanager
def strict_audio_convolution(device=None):
    if device is not None and torch.device(device).type != 'cuda':
        yield
        return
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


def gather_time(local, total, group, unit=1):
    """Uneven temporal partitions: pad transport only, never add modeled frames."""
    world = dist.get_world_size(group)
    sizes = [(total * (i + 1) // world - total * i // world) * unit for i in range(world)]
    width = max(sizes)
    payload = torch.nn.functional.pad(local, (0, width - local.shape[-1])).contiguous()
    received = [torch.empty_like(payload) for _ in range(world)]
    dist.all_gather(received, payload, group=group)
    return torch.cat([tensor[..., :size] for tensor, size in zip(received, sizes)], dim=-1)


def partition(total, halo, group):
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    lo, hi = total * rank // world, total * (rank + 1) // world
    return lo, hi, max(0, lo - halo), min(total, hi + halo)


def register_encoder_stage(vae, weights, device):
    """Keep native pipeline.to()/execution-device detection aware of staging.

    Removing the final encoder leaf hooks otherwise makes pipeline.to() move
    the entire VAE, invalidating the decoder group's buffer-keyed CPU master.
    H3 calls encode/decode methods, so their wrappers own the stage lifetime.
    """
    from diffusers.hooks.group_offloading import (
        GroupOffloadingConfig, GroupOffloadingType, _apply_group_offloading_hook)

    config = GroupOffloadingConfig(onload_device=torch.device(device), offload_device=torch.device('cpu'),
        offload_type=GroupOffloadingType.BLOCK_LEVEL, non_blocking=False, record_stream=False,
        low_cpu_mem_usage=False, stream=weights.stream)
    _apply_group_offloading_hook(vae, weights, config=config)


def install_video_encoder(vae, group_getter, device, *, stage_weights=True):
    from diffusers.hooks.group_offloading import ModuleGroup
    from .h3_cuda_precision import retain_module

    accelerator = torch.get_device_module(torch.device(device).type)
    cpu = torch.device('cpu')
    weights = None
    if stage_weights:
        retain_module(vae.encoder, cpu)
        retain_module(vae.quant_conv, cpu)
        weights = ModuleGroup(modules=[vae.encoder, vae.quant_conv], offload_device=cpu,
                              onload_device=device, offload_leader=vae.encoder,
                              stream=accelerator.Stream(device=device), low_cpu_mem_usage=False)
        weights.offload_()
        register_encoder_stage(vae, weights, device)
    metrics = dict(calls=0, completed=0, clips=0, local_tiles=0, valid_tiles=0,
                   fallback_clips=0, h2d_seconds=0., return_master_seconds=0.)
    original_encode, original_clip = vae.encode, vae._encode_clip

    @wraps(original_clip)
    def encode_clip(pixels):
        group = group_getter()
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if not vae.use_tiling:
            metrics['fallback_clips'] += 1
            return original_clip(pixels)
        ys, hs, yo = vae._split_tiles(pixels.shape[-2], vae.tile_sample_min_height,
                                     vae.tile_sample_min_overlap_height)
        xs, ws, xo = vae._split_tiles(pixels.shape[-1], vae.tile_sample_min_width,
                                     vae.tile_sample_min_overlap_width)
        total = len(ys) * len(xs)
        # Native small/unequal tiles stay native. Every rank takes this same branch.
        if total < world or len(set(hs)) > 1 or len(set(ws)) > 1:
            metrics['fallback_clips'] += 1
            return original_clip(pixels)
        tasks = list(range(rank, total, world))
        local = []
        for task in tasks:
            i, j = divmod(task, len(xs))
            local.append(vae.quant_conv(vae.encoder(pixels[..., ys[i]:ys[i]+hs[i], xs[j]:xs[j]+ws[j]])))
        slots, batch = math.ceil(total / world), pixels.shape[0]
        payload = torch.cat(local + [torch.zeros_like(local[0])] * (slots-len(local)), dim=0).contiguous()
        received = [torch.empty_like(payload) for _ in range(world)]
        dist.all_gather(received, payload, group=group)
        tiles = [received[t % world][(t // world)*batch:(t // world+1)*batch] for t in range(total)]
        rows = [tiles[i:i+len(xs)] for i in range(0, total, len(xs))]
        ratio = vae.spatial_compression_ratio
        result = vae._stitch_tiles(rows, [v // ratio for v in yo], [v // ratio for v in xo])
        metrics['clips'] += 1
        metrics['local_tiles'] += len(tasks)
        metrics['valid_tiles'] += total
        return result

    @wraps(original_encode)
    def encode(*args, **kwargs):
        if torch.is_grad_enabled():
            raise ValueError('H3 parallel VAE encoder is inference-only')
        metrics['calls'] += 1
        if weights is None:
            result = original_encode(*args, **kwargs)
            metrics['completed'] += 1
            return result
        started = time.perf_counter()
        weights.onload_()
        accelerator.current_stream(device).wait_stream(weights.stream)
        weights.stream.synchronize()
        metrics['h2d_seconds'] += time.perf_counter()-started
        try:
            result = original_encode(*args, **kwargs)
            metrics['completed'] += 1
            return result
        finally:
            started = time.perf_counter()
            weights.offload_()
            metrics['return_master_seconds'] += time.perf_counter()-started

    vae._encode_clip, vae.encode = encode_clip, encode
    vae._h3_encoder_parallel_metrics = metrics
    vae._h3_encoder_group = weights


def install_audio_parallel(audio, group_getter, device, encode_enabled, decode_profile, *, stage_weights=True):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        MiniMaxH3AudioDiagonalGaussianDistribution, MiniMaxH3AudioEncoderOutput)
    from diffusers.models.autoencoders.vae import DecoderOutput
    from .h3_audio_receptive import audio_halos
    from .h3_cuda_precision import offload_audio

    halos = audio_halos(audio)
    metrics = dict(encode_calls=0, encode_parallel=0, decode_calls=0, decode_parallel=0,
                   encoder_halo=halos['encoder_used'], decoder_halo=halos['decoder_used'])
    original_encode, original_decode = audio.encode, audio.decode

    @wraps(original_encode)
    def encode(sample, return_dict=True):
        metrics['encode_calls'] += 1
        if sample.ndim != 3 or sample.shape[1] != 1:
            return original_encode(sample, return_dict=return_dict)
        hop = audio.hop_length
        sample = torch.nn.functional.pad(sample, (0, (-sample.shape[-1]) % hop))
        total, group = sample.shape[-1] // hop, group_getter()
        if total < dist.get_world_size(group):
            return original_encode(sample, return_dict=return_dict)
        if torch.is_grad_enabled():
            raise ValueError('H3 parallel audio encoder is inference-only')
        if stage_weights:
            audio.to(device)
        try:
            with strict_audio_convolution(device):
                lo, hi, a, b = partition(total, halos['encoder_used'], group)
                hidden = audio.encoder(sample[..., a*hop:b*hop].to(device, torch.float32))
                hidden = gather_time(hidden[..., lo-a:hi-a], total, group)
                hidden = audio.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
                posterior = MiniMaxH3AudioDiagonalGaussianDistribution(audio.mean_proj(hidden), audio.logs_proj(hidden))
            metrics['encode_parallel'] += 1
            return MiniMaxH3AudioEncoderOutput(latent_dist=posterior) if return_dict else (posterior,)
        finally:
            if stage_weights:
                offload_audio(audio)

    @wraps(original_decode)
    def decode(latents, return_dict=True):
        metrics['decode_calls'] += 1
        # Native video durations include alignment frames (e.g. 405, 603 audio latents).
        # Limit this opt-in to the 10/15-second operating ranges; 5s stays unchanged.
        total = latents.shape[-1]
        metrics['last_decode_latents'] = total
        if not (400 <= total <= 420 or 600 <= total <= 620):
            return original_decode(latents, return_dict=return_dict)
        group = group_getter()
        with strict_audio_convolution(device):
            lo, hi, a, b = partition(total, halos['decoder_used'], group)
            wave = original_decode(latents[..., a:b], return_dict=False)[0]
            local = wave[..., (lo-a)*audio.hop_length:(hi-a)*audio.hop_length]
            result = gather_time(local, total, group, unit=audio.hop_length)
        metrics['decode_parallel'] += 1
        return DecoderOutput(sample=result) if return_dict else (result,)

    if encode_enabled:
        audio.encode = encode
    if decode_profile == 'long':
        audio.decode = decode
    audio._h3_parallel_metrics = metrics


def install_condition_parallel(pipe, args, device, *, stage_weights=True):
    def group():
        manager = getattr(pipe.transformer, '_h3_condition_parallel_manager', getattr(pipe.transformer, 'parallel_manager', None))
        if manager is None or manager.sp_group is None or manager.sp_size not in (2, 4, 8):
            raise ValueError('H3 condition parallelism requires an initialized SP2/4/8 group')
        return manager.sp_group

    if getattr(args, 'h3_cuda_vision_sp', False):
        from .h3_vision_sp import install_vision_sp
        install_vision_sp(pipe.text_encoder.model.visual, group)
    if getattr(args, 'h3_cuda_vae_encoder_parallel', False):
        install_video_encoder(pipe.vae, group, device, stage_weights=stage_weights)
    encoder = bool(getattr(args, 'h3_cuda_audio_encoder_parallel', False))
    decoder = getattr(args, 'h3_cuda_audio_decoder_parallel', 'none')
    if encoder or decoder != 'none':
        install_audio_parallel(pipe.audio_vae, group, device, encoder, decoder, stage_weights=stage_weights)
    logger.info('Experimental H3 condition parallelism: vision=%s video_encoder=%s audio_encoder=%s audio_decoder=%s',
                getattr(args, 'h3_cuda_vision_sp', False), getattr(args, 'h3_cuda_vae_encoder_parallel', False), encoder, decoder)
