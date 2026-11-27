"""H3 video/audio VAE adaptation and decode mechanisms."""
from __future__ import annotations


# ------------------------------------------------------------------------
# vae.h3.audio_precision
# ------------------------------------------------------------------------

# Keep H3's DAC/BigVGAN audio VAE in FP32 independently of the DiT recipe.

from functools import wraps

import torch


def prepare_audio_vae(audio):
    """Preserve FP32 weights and disable surrounding autocast for audio calls."""
    audio.float().eval()

    def protect(method):
        @wraps(method)
        def call(*args, **kwargs):
            def fp32(value):
                return (
                    value.float()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )

            device_type = next(audio.parameters()).device.type
            with torch.autocast(device_type=device_type, enabled=False):
                return method(
                    *(fp32(value) for value in args),
                    **{name: fp32(value) for name, value in kwargs.items()},
                )

        return call

    audio.encode = protect(audio.encode)
    audio.decode = protect(audio.decode)


def offload_audio(audio):
    audio.to(torch.device("cpu"))
    # Legacy weight_norm caches an unregistered weight tensor. Unused encoder
    # caches can still be meta placeholders after loading; they carry no data.
    for child in audio.modules():
        if (
            hasattr(child, "weight_g")
            and hasattr(child, "weight_v")
            and child.weight.device.type in ("cuda", "npu")
        ):
            child.weight = child.weight.to("cpu")


# ------------------------------------------------------------------------
# vae.h3.audio_receptive
# ------------------------------------------------------------------------

# Native H3 audio convolution/anti-alias receptive-field accounting.

import math




def _conv_span(lo, hi, kernel, stride=1, padding=0, dilation=1, transpose=False):
    if transpose:
        return (
            -((-(lo + padding - dilation * (kernel - 1))) // stride),
            (hi - 1 + padding) // stride + 1,
        )
    return lo * stride - padding, (hi - 1) * stride - padding + dilation * (
        kernel - 1
    ) + 1


def audio_span(module, lo, hi):
    """Back-propagate exact convolution/AA receptive intervals, no audio cuts."""
    name = type(module).__name__
    if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        return _conv_span(
            lo,
            hi,
            module.kernel_size[0],
            module.stride[0],
            module.padding[0],
            module.dilation[0],
            isinstance(module, torch.nn.ConvTranspose1d),
        )
    if isinstance(module, torch.nn.Sequential):
        for child in reversed(list(module)):
            lo, hi = audio_span(child, lo, hi)
        return lo, hi
    if name in ("MiniMaxH3AudioEncoder", "MiniMaxH3AudioEncoderBlock"):
        return audio_span(module.block, lo, hi)
    if name == "MiniMaxH3AudioResidualUnit":
        a, b = audio_span(module.block, lo, hi)
        return min(a, lo), max(b, hi)
    if name == "MiniMaxH3AudioActivation1d":
        lo, hi = audio_span(module.downsample, lo, hi)
        return audio_span(module.upsample, lo, hi)
    if name == "MiniMaxH3AudioDownSample1d":
        return audio_span(module.lowpass, lo, hi)
    if name == "MiniMaxH3AudioLowPassFilter1d":
        return _conv_span(
            lo, hi, module.filter.shape[-1], module.stride, module.pad_left
        )
    if name == "MiniMaxH3AudioUpSample1d":
        lo, hi = _conv_span(
            lo + module.pad_left,
            hi + module.pad_left,
            module.filter.shape[-1],
            module.stride,
            transpose=True,
        )
        return lo - module.pad, hi - module.pad
    if name == "MiniMaxH3AudioAMPBlock":
        for i in reversed(range(len(module.convs1))):
            a, b = audio_span(module.convs2[i], lo, hi)
            a, b = audio_span(module.activations[2 * i + 1], a, b)
            a, b = audio_span(module.convs1[i], a, b)
            a, b = audio_span(module.activations[2 * i], a, b)
            lo, hi = min(lo, a), max(hi, b)
        return lo, hi
    if name == "MiniMaxH3AudioBigVGANDecoder":
        lo, hi = audio_span(module.conv_post, lo, hi)
        lo, hi = audio_span(module.activation_post, lo, hi)
        for i in reversed(range(module.num_upsamples)):
            intervals = [
                audio_span(module.resblocks[i * module.num_kernels + j], lo, hi)
                for j in range(module.num_kernels)
            ]
            lo, hi = min(a for a, b in intervals), max(b for a, b in intervals)
            lo, hi = audio_span(module.ups[i][0], lo, hi)
        return audio_span(module.conv_pre, lo, hi)
    if name in ("MiniMaxH3AudioSnake1d", "MiniMaxH3AudioSnakeBeta"):
        return lo, hi
    raise TypeError("Unhandled receptive field module: " + name)


def audio_halos(model):
    hop = model.hop_length
    frame = 1000
    a, b = audio_span(model.encoder, frame, frame + 1)
    encoder = max(
        math.ceil((frame * hop - a) / hop), math.ceil((b - (frame + 1) * hop) / hop)
    )
    a, b = audio_span(model.decoder, frame * hop, (frame + 1) * hop)
    a, b = audio_span(model.dec_in_proj, a, b)
    decoder = max(frame - a, b - frame - 1)
    return dict(
        encoder_required=encoder,
        decoder_required=decoder,
        encoder_used=math.ceil((encoder + 1) / 4) * 4,
        decoder_used=math.ceil((decoder + 1) / 4) * 4,
    )


# ------------------------------------------------------------------------
# vae.h3.parallel
# ------------------------------------------------------------------------

# Optional reference component parallelism; native H3 inference scheduling stays in charge.

from x_diffusers.framework.h3_config import H3ExecutionOptions
from contextlib import contextmanager

import logging

import time


import torch.distributed as dist
from x_base.sequence_parallelism.comm import gather_time, partition

logger = logging.getLogger(__name__)


@contextmanager
def strict_audio_convolution(device=None):
    if device is not None and torch.device(device).type != "cuda":
        yield
        return
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


def register_encoder_stage(vae, weights, device):
    """Keep native pipeline.to()/execution-device detection aware of staging.

    Removing the final encoder leaf hooks otherwise makes pipeline.to() move
    the entire VAE, invalidating the decoder group's buffer-keyed CPU master.
    H3 calls encode/decode methods, so their wrappers own the stage lifetime.
    """
    from diffusers.hooks.group_offloading import (
        GroupOffloadingConfig,
        GroupOffloadingType,
        _apply_group_offloading_hook,
    )

    config = GroupOffloadingConfig(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type=GroupOffloadingType.BLOCK_LEVEL,
        non_blocking=False,
        record_stream=False,
        low_cpu_mem_usage=False,
        stream=weights.stream,
    )
    _apply_group_offloading_hook(vae, weights, config=config)


def install_video_encoder(vae, group_getter, device, *, stage_weights=True):
    from diffusers.hooks.group_offloading import ModuleGroup
    from x_diffusers.framework.offload import retain_module

    accelerator = torch.get_device_module(torch.device(device).type)
    cpu = torch.device("cpu")
    weights = None
    if stage_weights:
        retain_module(vae.encoder, cpu)
        retain_module(vae.quant_conv, cpu)
        weights = ModuleGroup(
            modules=[vae.encoder, vae.quant_conv],
            offload_device=cpu,
            onload_device=device,
            offload_leader=vae.encoder,
            stream=accelerator.Stream(device=device),
            low_cpu_mem_usage=False,
        )
        weights.offload_()
        register_encoder_stage(vae, weights, device)
    metrics = dict(
        calls=0,
        completed=0,
        clips=0,
        local_tiles=0,
        valid_tiles=0,
        fallback_clips=0,
        h2d_seconds=0.0,
        return_master_seconds=0.0,
    )
    original_encode, original_clip = vae.encode, vae._encode_clip

    @wraps(original_clip)
    def encode_clip(pixels):
        transport = getattr(vae, "_h3_overlap_transport", None)
        group = group_getter() if transport is None else transport.group
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if not vae.use_tiling:
            metrics["fallback_clips"] += 1
            return original_clip(pixels)
        ys, hs, yo = vae._split_tiles(
            pixels.shape[-2],
            vae.tile_sample_min_height,
            vae.tile_sample_min_overlap_height,
        )
        xs, ws, xo = vae._split_tiles(
            pixels.shape[-1],
            vae.tile_sample_min_width,
            vae.tile_sample_min_overlap_width,
        )
        total = len(ys) * len(xs)
        # Native small/unequal tiles stay native. Every rank takes this same branch.
        if total < world or len(set(hs)) > 1 or len(set(ws)) > 1:
            metrics["fallback_clips"] += 1
            return original_clip(pixels)
        tasks = list(range(rank, total, world))
        local = []
        for task in tasks:
            i, j = divmod(task, len(xs))
            local.append(
                vae.quant_conv(
                    vae.encoder(
                        pixels[..., ys[i] : ys[i] + hs[i], xs[j] : xs[j] + ws[j]]
                    )
                )
            )
        slots, batch = math.ceil(total / world), pixels.shape[0]
        payload = torch.cat(
            local + [torch.zeros_like(local[0])] * (slots - len(local)), dim=0
        ).contiguous()
        received = [torch.empty_like(payload) for _ in range(world)]
        if transport is None:
            dist.all_gather(received, payload, group=group)
        else:
            transport.gather(received, payload)
        tiles = [
            received[t % world][(t // world) * batch : (t // world + 1) * batch]
            for t in range(total)
        ]
        rows = [tiles[i : i + len(xs)] for i in range(0, total, len(xs))]
        ratio = vae.spatial_compression_ratio
        result = vae._stitch_tiles(
            rows, [v // ratio for v in yo], [v // ratio for v in xo]
        )
        metrics["clips"] += 1
        metrics["local_tiles"] += len(tasks)
        metrics["valid_tiles"] += total
        return result

    @wraps(original_encode)
    def encode(*options, **kwargs):
        if torch.is_grad_enabled():
            raise ValueError("H3 parallel VAE encoder is inference-only")
        metrics["calls"] += 1
        if weights is None:
            result = original_encode(*options, **kwargs)
            metrics["completed"] += 1
            return result
        started = time.perf_counter()
        weights.onload_()
        accelerator.current_stream(device).wait_stream(weights.stream)
        weights.stream.synchronize()
        metrics["h2d_seconds"] += time.perf_counter() - started
        try:
            result = original_encode(*options, **kwargs)
            metrics["completed"] += 1
            return result
        finally:
            started = time.perf_counter()
            weights.offload_()
            metrics["return_master_seconds"] += time.perf_counter() - started

    vae._encode_clip, vae.encode = encode_clip, encode
    vae._h3_encoder_parallel_metrics = metrics
    vae._h3_encoder_group = weights


def install_audio_parallel(
    audio, group_getter, device, encode_enabled, decode_profile, *, stage_weights=True
):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        MiniMaxH3AudioDiagonalGaussianDistribution,
        MiniMaxH3AudioEncoderOutput,
    )
    from diffusers.models.autoencoders.vae import DecoderOutput

    halos = audio_halos(audio)
    metrics = dict(
        encode_calls=0,
        encode_parallel=0,
        decode_calls=0,
        decode_parallel=0,
        encoder_halo=halos["encoder_used"],
        decoder_halo=halos["decoder_used"],
    )
    original_encode, original_decode = audio.encode, audio.decode

    @wraps(original_encode)
    def encode(sample, return_dict=True):
        metrics["encode_calls"] += 1
        if sample.ndim != 3 or sample.shape[1] != 1:
            return original_encode(sample, return_dict=return_dict)
        hop = audio.hop_length
        sample = torch.nn.functional.pad(sample, (0, (-sample.shape[-1]) % hop))
        total, group = sample.shape[-1] // hop, group_getter()
        if total < dist.get_world_size(group):
            return original_encode(sample, return_dict=return_dict)
        if torch.is_grad_enabled():
            raise ValueError("H3 parallel audio encoder is inference-only")
        if stage_weights:
            audio.to(device)
        try:
            with (
                torch.autocast(device_type=torch.device(device).type, enabled=False),
                strict_audio_convolution(device),
            ):
                lo, hi, a, b = partition(total, halos["encoder_used"], group)
                hidden = audio.encoder(
                    sample[..., a * hop : b * hop].to(device, torch.float32)
                )
                hidden = gather_time(hidden[..., lo - a : hi - a], total, group)
                hidden = audio.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
                posterior = MiniMaxH3AudioDiagonalGaussianDistribution(
                    audio.mean_proj(hidden), audio.logs_proj(hidden)
                )
            metrics["encode_parallel"] += 1
            return (
                MiniMaxH3AudioEncoderOutput(latent_dist=posterior)
                if return_dict
                else (posterior,)
            )
        finally:
            if stage_weights:
                offload_audio(audio)

    @wraps(original_decode)
    def decode(latents, return_dict=True):
        metrics["decode_calls"] += 1
        # Native video durations include alignment frames (e.g. 405, 603 audio latents).
        # Limit this opt-in to the 10/15-second operating ranges; 5s stays unchanged.
        total = latents.shape[-1]
        metrics["last_decode_latents"] = total
        if not (400 <= total <= 420 or 600 <= total <= 620):
            return original_decode(latents, return_dict=return_dict)
        group = group_getter()
        with strict_audio_convolution(device):
            lo, hi, a, b = partition(total, halos["decoder_used"], group)
            wave = original_decode(latents[..., a:b], return_dict=False)[0]
            local = wave[..., (lo - a) * audio.hop_length : (hi - a) * audio.hop_length]
            result = gather_time(local, total, group, unit=audio.hop_length)
        metrics["decode_parallel"] += 1
        return DecoderOutput(sample=result) if return_dict else (result,)

    if encode_enabled:
        audio.encode = encode
    if decode_profile == "long":
        audio.decode = decode
    audio._h3_parallel_metrics = metrics


def install_condition_parallel(
    pipe, options: H3ExecutionOptions, device, *, stage_weights=True
):
    def group():
        manager = getattr(
            pipe.transformer,
            "_h3_condition_parallel_manager",
            getattr(pipe.transformer, "parallel_manager", None),
        )
        if (
            manager is None
            or manager.sp_group is None
            or manager.sp_size not in (2, 4, 8)
        ):
            raise ValueError(
                "H3 condition parallelism requires an initialized SP2/4/8 group"
            )
        return manager.sp_group

    if getattr(options, "h3_cuda_vision_sp", False):
        from x_diffusers.framework.h3_encoders import install_vision_sp

        install_vision_sp(pipe.text_encoder.model.visual, group)
    if getattr(options, "h3_cuda_vae_encoder_parallel", False):
        install_video_encoder(pipe.vae, group, device, stage_weights=stage_weights)
    encoder = bool(getattr(options, "h3_cuda_audio_encoder_parallel", False))
    decoder = getattr(options, "h3_cuda_audio_decoder_parallel", "none")
    if encoder or decoder != "none":
        install_audio_parallel(
            pipe.audio_vae, group, device, encoder, decoder, stage_weights=stage_weights
        )
    logger.info(
        "Experimental H3 condition parallelism: vision=%s video_encoder=%s audio_encoder=%s audio_decoder=%s",
        getattr(options, "h3_cuda_vision_sp", False),
        getattr(options, "h3_cuda_vae_encoder_parallel", False),
        encoder,
        decoder,
    )


# ------------------------------------------------------------------------
# vae.h3_compile
# ------------------------------------------------------------------------

# Optional Torch Compiler integration for the CUDA native H3 VAE decoder.









def compile_decoder_tile_subgraph(decoder: torch.nn.Module) -> torch.nn.Module:
    """Compile only the native decoder tile module and preserve its module API.

    The native-stream wrapper performs distributed tile ownership and gather
    outside this module.  Keeping those collectives in Python avoids compiling
    the request-level stream choreography.  ``max-autotune-no-cudagraphs`` is
    deliberate: CUDA Graph capture can retain stale all-gather output buffers
    across native tile calls.
    """

    if getattr(decoder, "_xgen_torch_compile_enabled", False):
        return decoder
    original = decoder.forward
    try:
        compiled = torch.compile(
            original,
            dynamic=True,
            fullgraph=False,
            mode="max-autotune-no-cudagraphs",
        )
    except TypeError as exc:
        raise RuntimeError(
            "Torch Compiler for H3 requires a PyTorch build supporting "
            "mode='max-autotune-no-cudagraphs'"
        ) from exc
    decoder._xgen_torch_compile_original_forward = original
    decoder.forward = compiled
    decoder._xgen_torch_compile_enabled = True
    logger.info(
        "H3 native-stream VAE decoder tile subgraph compiled with Torch Compiler"
    )
    return decoder


# ------------------------------------------------------------------------
# vae.h3_cuda_chunks
# ------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0
# Bound decoded temporal storage while retaining the native H3 decode loop.






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


# ------------------------------------------------------------------------
# vae.h3_cuda_tiles
# ------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0
# Distribute existing H3 spatial decode tiles across the CUDA SP group.

from types import MethodType





def install_vae_parallel(vae, manager, *, tile_parallel=False):
    """Attach the explicit SP manager and optionally parallelize spatial tiles."""
    heads = vae.decoder.transformer_blocks[0].attn.heads
    if heads % manager.sp_size:
        raise ValueError("CUDA H3 VAE attention heads must divide evenly across SP ranks")
    vae._h3_cuda_vae_parallel = manager
    if tile_parallel:
        vae._h3_original_decode_clip = vae._decode_clip
        vae._decode_clip = MethodType(decode_clip_tiles, vae)
        vae._h3_cuda_vae_tiles = True


def decode_clip_tiles(vae, z):
    if not vae._h3_cuda_vae_tiles or not vae.use_tiling:
        return vae._h3_original_decode_clip(z)

    ratio = vae.spatial_compression_ratio
    y, heights, y_overlap = vae._split_tiles(
        z.shape[-2] * ratio, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height,
    )
    x, widths, x_overlap = vae._split_tiles(
        z.shape[-1] * ratio, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width,
    )
    group = vae._h3_cuda_vae_parallel.sp_group
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    num_tiles = len(y) * len(x)
    tiles = []
    for start in range(0, num_tiles, size):
        tile_id = min(start + rank, num_tiles - 1)
        i, j = divmod(tile_id, len(x))
        top, left = y[i] // ratio, x[j] // ratio
        tile = z[..., top:top + heights[i] // ratio, left:left + widths[j] // ratio]
        local = vae.decoder(vae.post_quant_conv(tile)).contiguous()
        gathered = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(gathered, local, group=group)
        tiles.extend(gathered[:num_tiles - start])

    rows = [tiles[start:start + len(x)] for start in range(0, num_tiles, len(x))]
    return vae._stitch_tiles(rows, y_overlap, x_overlap)


# ------------------------------------------------------------------------
# vae.h3_refiner_output
# ------------------------------------------------------------------------

# Refiner pixel/crop adapter over the existing SaveAVStream codec, not a new encoder.
from pathlib import Path


from x_base.vae_parallelism.audio_utils import save_audio_to_wav
from x_base.vae_parallelism.save_av_stream import SaveAVStream


def refiner_rgb_frames(chunk):
    """Convert normalized [1,C,T,H,W] Refiner pixels into RGB8 [T,H,W,C]."""
    if not torch.isfinite(chunk).all():
        raise ValueError("Nonfinite pixels before video export")
    mean = chunk.new_tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = chunk.new_tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
    return ((chunk.float() * std + mean).clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 3, 0)


# ------------------------------------------------------------------------
# vae.omni_decoder
# ------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0
# Opt-in native Omni H3 decoder, behind the existing H3 decode block.
from dataclasses import asdict




import numpy as np






class OmniH3Decoder:
    def __init__(self, path, device, *, trust_remote_code=False, fp16_linears=False,
                 torch_compile=False):
        from diffusers.hooks.group_offloading import ModuleGroup
        from x_diffusers.framework._vendor.omni_h3 import MiniMaxH3VideoVAE

        self.device = torch.device(device)
        self.device_module = torch.get_device_module(self.device.type)
        self.path = path
        self.vae = MiniMaxH3VideoVAE(path, device=self.device, load_device=torch.device("cpu"),
                                   trust_remote_code=trust_remote_code)
        if fp16_linears:
            count = prepare_decoder_linears(self.vae.model.decoder)
            logger.info("H3 native decoder: %d Linear modules stored as FP16 autocast operands", count)
        self.torch_compile_enabled = bool(torch_compile)
        if self.torch_compile_enabled:
            compile_decoder_tile_subgraph(self.vae.model.decoder)
        # Reuse xGen/Diffusers' CPU-master staging contract. No re-read,
        # requantization or GPU-to-host weight copy between requests.
        self.group = ModuleGroup(modules=[self.vae.model], offload_device=torch.device("cpu"),
                                 onload_device=self.device, offload_leader=self.vae.model,
                                 stream=self.device_module.Stream(device=self.device), low_cpu_mem_usage=False)
        self.group.offload_()
        if dist.is_initialized():
            self.vae.set_parallel_size(dist.get_world_size(), process_group=dist.group.WORLD)
        self.audio = None
        self.sample_rate = None
        self.metrics = {}

    def decode_step(self, block, pipeline, state):
        pipeline, state = block.sub_blocks["audio"](pipeline, state)
        self.audio, self.sample_rate = state.get("audio"), state.get("sampling_rate")
        try:
            return block.sub_blocks["video"](pipeline, state)
        finally:
            self.audio = self.sample_rate = None

    @torch.no_grad()
    def decode_video(self, block, pipeline, state):
        from x_base.utils.infer_info import infer_info
        from x_diffusers.framework._vendor.omni_h3.chunked_video import ChunkedVideoMP4Session
        from x_diffusers.framework._vendor.omni_h3.chunked_video import decode_to_mp4

        values = block.get_block_state(state)
        infer_info._route_b_saved = False
        if values.output_type == "latent":
            values.videos = values.latents
            block.set_block_state(state, values)
            return pipeline, state
        started = time.perf_counter()
        weights_started = None
        loaded = None
        session = None
        succeeded = False
        try:
            prepare = getattr(pipeline, "_h3_prepare_video_decode", None)
            if prepare is not None:
                prepare()
            weights_started = time.perf_counter()
            self.group.onload_()
            self.group.stream.synchronize()
            loaded = time.perf_counter()
            with torch.autocast(self.device.type, dtype=self.vae.decode_dtype):
                if values.output_type != "pil":
                    frames = self.vae.decode_latent(values.latents)
                    if frames.dtype == torch.uint8:
                        frames = frames.float().div_(255)
                    values.videos = pipeline.video_processor.postprocess_video(frames, output_type=values.output_type)
                else:
                    if values.latents.shape[0] != 1:
                        raise ValueError("Native-stream H3 file output currently supports one video per request")
                    audio = self.audio
                    if audio is not None:
                        if isinstance(audio, torch.Tensor):
                            audio = audio.detach().float().cpu().numpy()
                        audio = np.asarray(audio, dtype=np.float32)
                        if audio.ndim == 3:
                            audio = audio[0]
                        if audio.ndim == 1:
                            audio = audio[None, :]
                        if audio.shape[0] > 2 and audio.shape[-1] <= 2:
                            audio = audio.T
                        audio = np.ascontiguousarray(audio)
                    frame_limit = getattr(infer_info, '_h3_native_frame_limit', None)
                    if frame_limit is not None and audio is not None:
                        audio = audio[..., :round(frame_limit / infer_info.fps * self.sample_rate)]
                    height = int(getattr(infer_info, "save_height", 0) or infer_info.height)
                    width = int(getattr(infer_info, "save_width", 0) or infer_info.width)
                    session = ChunkedVideoMP4Session(value_range=self.vae.chunk_value_range,
                        fps=infer_info.fps, audio_waveforms=[audio] if audio is not None else None,
                        audio_sample_rate=self.sample_rate, batch_frames=17,
                        video_codec_options={"preset": "fast", "crf": "10", "threads": "0"}, crop=(height, width),
                        frame_limit=frame_limit)
                    # Agree on encoder startup before any rank enters tile collectives.
                    error = None
                    owner = not dist.is_initialized() or dist.get_rank() == 0
                    if owner:
                        try:
                            session.start(width=width, height=height, batch_size=1)
                        except BaseException as exc:
                            error = exc
                    failed = torch.tensor([int(error is not None)], device=self.device, dtype=torch.int32)
                    if dist.is_initialized():
                        dist.broadcast(failed, src=0)
                    if failed.item():
                        raise RuntimeError("Native-stream H3 MP4 encoder startup failed") from error
                    error = None
                    try:
                        videos = decode_to_mp4(self.vae, values.latents, session=session)
                        if owner:
                            Path(infer_info.save_path).write_bytes(videos[0])
                    except BaseException as exc:
                        error = exc
                    failed.fill_(int(error is not None))
                    if dist.is_initialized():
                        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                    if failed.item():
                        raise RuntimeError("Native-stream H3 decode or MP4 publication failed") from error
                    values.videos = [None]
                    infer_info._route_b_saved = True
            succeeded = True
        except BaseException:
            logger.exception("H3 native-stream decode failed before cleanup")
            raise
        finally:
            # Always attempt master restoration, even if onload, encoder abort
            # or device synchronization failed. Keep the original failure.
            cleanup_error = None
            for name, cleanup in (
                ("encoder", session.abort if session is not None else lambda: None),
                # A failed rank may have peer collectives pending. A global
                # device sync here can hide its original OOM indefinitely.
                ("device", lambda: self.device_module.synchronize(self.device) if succeeded else None),
            ):
                try:
                    cleanup()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
                    logger.exception("H3 VAE %s cleanup failed", name)
            decoded = time.perf_counter()
            try:
                self.group.offload_()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
                logger.exception("H3 VAE CPU master restoration failed")
            if not succeeded or cleanup_error is not None:
                infer_info._route_b_saved = False
            transfer = session.transfer_stats() if session is not None else None
            self.metrics = dict(prepare_seconds=(weights_started or decoded)-started,
                                h2d_seconds=(loaded or decoded)-(weights_started or decoded),
                                decode_output_seconds=decoded-loaded if loaded is not None else 0.0,
                                offload_seconds=time.perf_counter()-decoded,
                                success=succeeded and cleanup_error is None,
                                transfer=asdict(transfer) if transfer else None)
            logger.info("H3 VAE backend=native-stream metrics=%s", self.metrics)
            if succeeded and cleanup_error is not None:
                raise cleanup_error
        block.set_block_state(state, values)
        return pipeline, state


# ------------------------------------------------------------------------
# vae.minimax_h3
# ------------------------------------------------------------------------

# NPU adaptation for the MiniMax-H3 video and audio VAEs.
#
# The MiniMax-H3 video VAE (``AutoencoderKLMiniMaxH3``) uses causal 3-D
# convolutions with reflect spatial padding and per-frame GroupNorm.  On NPU
# the upstream encode / decode paths work correctly but cannot take advantage
# of VAE parallelism or frame-chunked decoding.  This module patches the VAE
# with NPU-friendly encode / decode entry points that mirror the Wan VAE
# adaptation: tiled / chunked decode with optional VAE-lightning support.
#
# The audio VAE (``AutoencoderKLMiniMaxH3Audio``) is a DAC-lineage 1-D
# convolutional stack that runs efficiently as-is; no adaptation is needed
# beyond ensuring the module is moved to the NPU device.
#
# Sequence-parallel (SP) mode
# ---------------------------
# When ``vae_lightning`` is enabled and ``torch.distributed`` is initialised,
# the ViT decoder runs with Ulysses-style sequence parallelism.  Each temporal
# chunk's token sequence is split across ranks; ``all_to_all`` converts between
# sequence-shard and head-shard before / after every attention layer so that the
# attention result is mathematically identical to single-card decode.  This
# replaces the approximate spatial-tiling approach (``VAEManager``) with an
# exact parallel decode — no seams, no blend artifacts.


import os
from concurrent.futures import ThreadPoolExecutor

# Switch: use a separate NPU stream for D2H copy (async) or block on compute stream (sync).
# Default OFF — sync D2H is stable; async copy stream can cause performance fluctuation.
# Set VAE_COPY_STREAM=1 to enable async D2H on a separate stream.
_USE_COPY_STREAM = os.environ.get("VAE_COPY_STREAM", "0") == "1"


from x_base.device import device_type

if device_type() == "npu":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


from diffusers.models.autoencoders import autoencoder_kl_minimax_h3
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
    MiniMaxH3VideoViTDecoder3d,
    MiniMaxH3VideoTransformerBlock,
    DecoderOutput,
    AutoencoderKLOutput,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from x_base.vae_parallelism.vae_mgr import VAEManager
from x_base.vae_parallelism.utils import enable_lightning, add_lightning_init


from x_base.utils.infer_info import infer_info
from x_base.sequence_parallelism import (
    all_to_all_before_attn,
    all_to_all_after_attn,
    gather_sequence,
    split_sequence,
    get_pad_manager,
)

# ---------------------------------------------------------------------------
# Video VAE — NPU encode / decode
# ---------------------------------------------------------------------------

# ===================================================================
# SP INFRASTRUCTURE
# ===================================================================

_VAE_SP_MANAGER = None  # cached ParallelManager for VAE SP
_CURRENT_SP_MANAGER = None  # injected into _sp_decoder_forward during upstream _decode

# SP decode strategy toggle (module-level so Route B can read it):
#   "full":   upstream _decode (temporal chunking + spatial tiling + blend) with SP injected
#   "chunks": temporal chunking + SP inside ViT, streaming output via SaveAVStream
_SP_MODE = "vae_mgr"


def _get_vae_sp_manager():
    """Create or reuse a ParallelManager sized to the current world.

    VAE decode runs *after* the transformer forward, so the same process
    group can be reused safely.
    """
    global _VAE_SP_MANAGER
    if _VAE_SP_MANAGER is not None:
        return _VAE_SP_MANAGER

    from x_base import ParallelManager
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    _VAE_SP_MANAGER = ParallelManager(
        dp_size=1, cp_size=1, sp_size=world_size, tp_size=1,
    )
    return _VAE_SP_MANAGER


# -------------------------------------------------------------------
# SP-aware TransformerBlock forward
# -------------------------------------------------------------------
_original_block_forward = MiniMaxH3VideoTransformerBlock.forward


def _sp_block_forward(
        self: MiniMaxH3VideoTransformerBlock,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        parallel_manager=None,
) -> torch.Tensor:
    """Transformer block with optional Ulysses SP.

    When *parallel_manager* is None or ``sp_size <= 1`` the call is
    identical to the upstream forward (zero communication overhead).

    SP data-flow (per layer)::

        hidden_states: (B, S/P, dim)           ← local sequence shard
        Q/K/V:          (B, S/P, H, D)         ← local projections
        RoPE:           (B, S/P, H, D)         ← on split seq, full heads
        all_to_all:     (B, S, H/P, D)         ← full seq, partial heads
        attention:      (B, S, H/P, D)         ← mathematically exact
        all_to_all:     (B, S/P, H, D)         ← back to seq shard
        FFN:            (B, S/P, dim)          ← local
    """
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    if sp_size <= 1:
        return _original_block_forward(self, hidden_states, rotary_emb)

    sp_group = parallel_manager.sp_group

    # --- Attention ---
    # NPU npu_rms_norm accepts BF16 input + FP32 weight → BF16 output (verified)
    norm_hidden = self.norm1(hidden_states)
    attn = self.attn

    # Q/K/V projections → (B, S/P, H, D)
    query = attn.to_q(norm_hidden).unflatten(-1, (attn.heads, attn.dim_head))
    key = attn.to_k(norm_hidden).unflatten(-1, (attn.heads, attn.dim_head))
    value = attn.to_v(norm_hidden).unflatten(-1, (attn.heads, attn.dim_head))

    # RMSNorm Q/K — BF16 input, affine=False (no weight, falls back to F.rms_norm)
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # RoPE — applied on split sequence with full heads (BEFORE all_to_all)
    # cos/sin shape: (B, S/P, 1, rotary_dim) — broadcasts with (B, S/P, H, D)
    if rotary_emb is not None:
        cos, sin = rotary_emb
        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        rotary_dim = cos.shape[-1]
        query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        query_first, query_second = query_rot.chunk(2, dim=-1)
        key_first, key_second = key_rot.chunk(2, dim=-1)
        query_rotated = torch.cat([-query_second, query_first], dim=-1)
        key_rotated = torch.cat([-key_second, key_first], dim=-1)
        query = torch.cat([query_rot * cos + query_rotated * sin, query_pass], dim=-1)
        key = torch.cat([key_rot * cos + key_rotated * sin, key_pass], dim=-1)

    # Ulysses: (B, S/P, H, D) → (B, S, H/P, D)
    query = all_to_all_before_attn(query, sp_group, scatter_dim=2, gather_dim=1, pad_key="vae_sp")
    key = all_to_all_before_attn(key, sp_group, scatter_dim=2, gather_dim=1, pad_key="vae_sp")
    value = all_to_all_before_attn(value, sp_group, scatter_dim=2, gather_dim=1, pad_key="vae_sp")

    # Attention: (B, S, H/P, D) → reshape (B, H/P, S, D) for SDPA
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    hidden_attn = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    # Back to (B, S, H/P * D)
    hidden_attn = hidden_attn.transpose(1, 2).flatten(2, 3)

    # Ulysses: (B, S, H/P, D) → (B, S/P, H, D) → flatten → (B, S/P, dim)
    hidden_attn = all_to_all_after_attn(
        hidden_attn, sp_group, scatter_dim=1, gather_dim=2, pad_key="vae_sp"
    )

    hidden_attn = attn.to_out[0](hidden_attn)
    hidden_states = hidden_states + hidden_attn * self.scale1

    # --- FFN (local, no communication) ---
    norm_hidden = self.norm2(hidden_states)
    hidden_states = hidden_states + self.ff(norm_hidden) * self.scale2

    return hidden_states


# -------------------------------------------------------------------
# SP-aware ViT Decoder forward
# -------------------------------------------------------------------
_original_decoder_forward = MiniMaxH3VideoViTDecoder3d.forward


def _sp_decoder_forward(
        self: MiniMaxH3VideoViTDecoder3d,
        hidden_states: torch.Tensor,
        parallel_manager=None,
) -> torch.Tensor:
    """ViT decoder forward with optional Ulysses SP.

    *hidden_states*: ``(B, C, T, H, W)`` latent input.

    SP data-flow::

        reshape → tokens (B, T*H*W, C)
        proj_in → (B, N, dim)
        append register + cls → (B, N+5, dim)
        split_sequence → (B, (N+5)/P, dim)          ← each rank
        36× block (SP attention)
        gather_sequence → (B, N+5, dim)
        norm_out → proj_out → unpatchify             ← full sequence
    """
    sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    # Fall back to module-level injection (used when upstream _decode calls
    # self.decoder() without parallel_manager kwarg).
    if parallel_manager is None:
        parallel_manager = _CURRENT_SP_MANAGER
        sp_size = parallel_manager.sp_size if parallel_manager is not None else 1

    if sp_size <= 1:
        return _original_decoder_forward(self, hidden_states)

    sp_group = parallel_manager.sp_group
    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    # --- 1. Reshape: (B,C,T,H,W) → (B, T*H*W, C) ---
    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
        batch_size, num_frames * height * width, num_channels
    )
    num_patches = hidden_states.shape[1]

    # --- 2. proj_in ---
    hidden_states = self.proj_in(hidden_states)

    # --- 3. Append register + cls tokens ---
    register_tokens = self.register_tokens.expand(batch_size, -1, -1)
    cls_token = torch.zeros_like(hidden_states[:, :1, :])
    hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

    # --- 4. RoPE position encoding (full, before split) ---
    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
    suffix_ids = position_ids.new_zeros(
        (batch_size, self.num_register_tokens + 1, 3)
    )
    position_ids = torch.cat([position_ids, suffix_ids], dim=1)
    rotary_emb = self.rope(position_ids)

    # --- 5. Split sequence across SP group ---
    # split_sequence handles padding internally based on the pad amount
    # calculated by PadManager.  No manual padding needed.
    full_seq_len = hidden_states.shape[1]
    pad_mgr = get_pad_manager()
    _pad = pad_mgr.set_from_group("vae_sp", full_seq_len, sp_group)

    hidden_states = split_sequence(hidden_states, sp_group, dim=1, pad=_pad)
    cos = split_sequence(rotary_emb[0], sp_group, dim=1, pad=_pad)
    sin = split_sequence(rotary_emb[1], sp_group, dim=1, pad=_pad)
    rotary_emb = (cos, sin)

    # --- 6. Transformer blocks with SP ---
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, rotary_emb, parallel_manager
            )
        else:
            hidden_states = block(hidden_states, rotary_emb, parallel_manager)

    # --- 7. Gather sequence back (gather_sequence removes padding internally) ---
    hidden_states = gather_sequence(hidden_states, sp_group, dim=1, pad=_pad)

    # --- 8. Output projection + unpatchify (on full sequence) ---
    hidden_states = self.norm_out(hidden_states)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states[:, :num_patches, :]

    patch_size, patch_size_t = self.patch_size, self.patch_size_t
    hidden_states = hidden_states.view(
        batch_size, num_frames, height, width,
        self.out_channels, patch_size_t, patch_size, patch_size,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(
        batch_size, self.out_channels,
        num_frames * patch_size_t, height * patch_size, width * patch_size,
    )


# -------------------------------------------------------------------
# Apply monkey-patches for SP (blocks + decoder)
# -------------------------------------------------------------------
MiniMaxH3VideoTransformerBlock.forward = _sp_block_forward
MiniMaxH3VideoViTDecoder3d.forward = _sp_decoder_forward


# ===================================================================
# Encode / Decode entry points
# ===================================================================

@torch.no_grad()
def _encode_ascend(self, x: torch.Tensor):
    """NPU encode for the MiniMax-H3 video VAE.

    Delegates to the upstream ``_encode``.
    """
    return self._encode(x)


@torch.no_grad()
def encode_ascend(
        self, x: torch.Tensor, return_dict: bool = True
):
    h = self._encode_ascend(x)
    posterior = DiagonalGaussianDistribution(h)
    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)


# ImageNet denormalization constants (module-level to avoid per-call allocation)
_PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)
_PIXEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)
_pixel_stats_cache: dict = {}


def _get_pixel_stats(device):
    """Cache device-specific ImageNet normalization constants."""
    if device not in _pixel_stats_cache:
        _pixel_stats_cache[device] = (_PIXEL_MEAN.to(device), _PIXEL_STD.to(device))
    return _pixel_stats_cache[device]


# --- Async D2H infrastructure (only used when VAE_COPY_STREAM=1) ---
_copy_stream = None
_write_executor = ThreadPoolExecutor(max_workers=1)


def _get_copy_stream():
    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.npu.Stream()
    return _copy_stream


def _flush_writes():
    """Block until all pending async frame-write tasks complete.

    No-op when VAE_COPY_STREAM is off (sync path writes inline).
    """
    if not _USE_COPY_STREAM:
        return
    _write_executor.submit(lambda: None).result()


def _stream_frames_to_av(av_stream, chunk):
    """Convert decoded chunk to uint8 HWC frames and stream to SaveAVStream.

    Default (VAE_COPY_STREAM=0): D2H on compute stream, frames written inline.
    Async  (VAE_COPY_STREAM=1): D2H on a separate stream, frames written
    in a background thread so compute is not blocked by CPU-side work.
    """
    if chunk.dim() == 4:
        chunk = chunk.unsqueeze(0)
    mean, std = _get_pixel_stats(chunk.device)
    frames = (chunk.float() * std + mean).clamp(0, 1)
    frames = frames.squeeze(0).permute(1, 2, 3, 0)  # (F, H, W, C)
    frames = (frames * 255).to(torch.uint8)

    if _USE_COPY_STREAM:
        _stream_frames_async(av_stream, frames)
    else:
        """D2H on compute stream, write frames inline."""
        frames_np = frames.cpu().numpy()
        for f in range(frames_np.shape[0]):
            av_stream.put_frame(frames_np[f])


def _stream_frames_async(av_stream, frames):
    """D2H on a separate stream, background thread writes frames."""
    event = torch.npu.Event()
    event.record()
    copy_stream = _get_copy_stream()
    with torch.npu.stream(copy_stream):
        event.wait()
        frames_cpu = frames.cpu()

    def _write():
        copy_stream.synchronize()
        frames_np = frames_cpu.numpy()
        for f in range(frames_np.shape[0]):
            av_stream.put_frame(frames_np[f])

    _write_executor.submit(_write)


def _decode_chunks_streaming(vae, z, vae_manager, av_stream, frame_callback=None):
    """Replicate upstream ``_decode`` chunk loop with streaming to SaveAVStream.

    Decodes the latent ``z`` chunk-by-chunk (each chunk = ``tokens_chunk_size``
    latents → ``chunk_num_frames`` pixel frames). After each chunk is blended,
    if ``vae_manager`` is provided, spatial tiles are gathered across ranks.
    Then frames are streamed to ``av_stream`` (rank 0 only).

    Returns the full decoded tensor (same as upstream ``_decode``).
    """
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    decoded_chunks = []
    overlap = None
    for i in range(num_chunks):
        start = i * tokens_chunk_size
        clip = vae._decode_clip(z[:, :, start: start + tokens_chunk_size + vae.token_overlap])
        for j in range(int(token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip[:, :, frame_start: frame_start + chunk_num_frames]
            chunk = chunk[:, :, vae.frame_pre_padding:]
            if j == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                # Spatial gather for lightning mode (collective — all ranks must call)
                if vae_manager is not None:
                    chunk = vae_manager.align_out(chunk)
                    chunk = vae_manager.collect_out(chunk)
                decoded_chunks.append(chunk)
                # Stream frames (rank 0 only, av_stream may be None before integration)
                if rank == 0 and (av_stream is not None or frame_callback is not None):
                    if frame_callback is None:
                        _stream_frames_to_av(av_stream, chunk)
                    else:
                        frame_callback(chunk)
            else:
                overlap = chunk

    if overlap is not None:
        if vae_manager is not None:
            overlap = vae_manager.align_out(overlap)
            overlap = vae_manager.collect_out(overlap)
        decoded_chunks.append(overlap)
        if rank == 0 and (av_stream is not None or frame_callback is not None):
            if frame_callback is None:
                _stream_frames_to_av(av_stream, overlap)
            else:
                frame_callback(overlap)

    dec = torch.cat(decoded_chunks, dim=2)

    if pad_tokens > 0:
        intra_tail = vae.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        dec = dec[:, :, :-pad_frames]
    return dec


# -------------------------------------------------------------------
# SP temporal-chunk decode (mathematically exact parallel decode)
# -------------------------------------------------------------------

def _decode_chunks_sp(vae, z, parallel_manager, av_stream=None):
    """Temporal chunked decode with Ulysses SP + streaming output.

    Same temporal chunking + spatial tiling + blend as upstream
    ``_decode``, but SP is injected into each ViT forward via the
    module-level ``_CURRENT_SP_MANAGER``.

    Each chunk's decoded frames are streamed to ``av_stream`` (rank 0
    only) before the next chunk starts, keeping memory bounded.
    """
    global _CURRENT_SP_MANAGER
    _CURRENT_SP_MANAGER = parallel_manager
    try:
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        temporal_ratio = vae.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        decoded_chunks = []
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            # Upstream _decode_clip does spatial tiling + blend internally.
            # SP is picked up via _CURRENT_SP_MANAGER inside _sp_decoder_forward.
            z_clip = z[:, :, start: start + tokens_chunk_size + vae.token_overlap]
            clip = vae._decode_clip(z_clip)
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start: frame_start + chunk_num_frames]
                chunk = chunk[:, :, vae.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                    if rank == 0 and av_stream is not None:
                        _stream_frames_to_av(av_stream, chunk)
                else:
                    overlap = chunk

        if overlap is not None:
            decoded_chunks.append(overlap)
            if rank == 0 and av_stream is not None:
                _stream_frames_to_av(av_stream, overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        if pad_tokens > 0:
            intra_tail = vae.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec
    finally:
        _CURRENT_SP_MANAGER = None


def _decode_sp_full(vae, z, parallel_manager):
    """Upstream ``_decode`` with SP injected.

    Calls upstream ``_decode`` as-is — temporal chunking + spatial tiling
    + blend are all handled by upstream.  SP is injected transparently:
    the module-level ``_CURRENT_SP_MANAGER`` is picked up by
    ``_sp_decoder_forward`` inside each ``self.decoder(...)`` call.

    Output is identical to single-card upstream ``_decode``, but each
    ViT forward is parallelised across ranks.
    """
    global _CURRENT_SP_MANAGER
    _CURRENT_SP_MANAGER = parallel_manager
    try:
        return vae._decode(z)
    finally:
        _CURRENT_SP_MANAGER = None


def _create_av_stream(vae, z):
    """Create SaveAVStream on rank 0, return None on other ranks."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return None
    audio_path = getattr(infer_info, "audio_path", None)
    vae_spatial_ratio = getattr(vae, "spatial_compression_ratio", 8)
    height = z.shape[-2] * vae_spatial_ratio
    width = z.shape[-1] * vae_spatial_ratio
    fps = getattr(infer_info, "fps", None) or 24
    return SaveAVStream(infer_info.save_path, audio_path, fps, width, height)


def _decode_sp(vae, z, parallel_manager):
    """SP multi-card dispatch — selects strategy based on _SP_MODE."""
    if _SP_MODE == "full":
        decoded = _decode_sp_full(vae, z, parallel_manager)
        infer_info._route_b_saved = False
        return decoded

    if _SP_MODE == "vae_mgr":
        vae_manager = VAEManager(
            z.shape,
            factor=vae.spatial_compression_ratio,
            is_encode=False,
            pad_mode=getattr(infer_info, "vae_pad_mode", "all_sides_valid"),
            pad_content=getattr(infer_info, "vae_pad_content", "constant"),
            use_blend=getattr(infer_info, "vae_use_blend", False),
        )
        tile = vae_manager.get_tile_from_x(z)
        av_stream = _create_av_stream(vae, z)
        decoded = _decode_chunks_streaming(vae, tile, vae_manager, av_stream)
        _flush_writes()
        if av_stream is not None:
            av_stream.close()
        infer_info._route_b_saved = True
        return decoded

    # "chunks"
    av_stream = _create_av_stream(vae, z)
    decoded = _decode_chunks_sp(vae, z, parallel_manager, av_stream=av_stream)
    _flush_writes()
    if av_stream is not None:
        av_stream.close()
    infer_info._route_b_saved = True
    return decoded


def _decode_lightning_single(vae, z):
    """Single-card lightning: VAEManager spatial tiling + streaming."""
    vae_manager = VAEManager(
        z.shape,
        factor=vae.spatial_compression_ratio,
        is_encode=False,
        pad_mode=getattr(infer_info, "vae_pad_mode", "all_sides_valid"),
        pad_content=getattr(infer_info, "vae_pad_content", "constant"),
        use_blend=getattr(infer_info, "vae_use_blend", False),
    )
    tile = vae_manager.get_tile_from_x(z)
    av_stream = _create_av_stream(vae, z)
    decoded = _decode_chunks_streaming(vae, tile, vae_manager, av_stream)
    _flush_writes()
    if av_stream is not None:
        av_stream.close()
    infer_info._route_b_saved = True
    return decoded


@torch.no_grad()
def _decode_ascend(self, z: torch.Tensor, return_dict: bool = True):
    """NPU decode dispatch.

    ``vae_lightning`` is the single switch.  Three paths:
    1. Multi-card (lightning + distributed): SP via ``_decode_sp``
    2. Single-card (lightning, no distributed): ``_decode_lightning_single``
    3. Off: upstream ``_decode``
    """
    is_lightning_mode = (
            getattr(infer_info, "vae_lightning", "none")
            in ["decoder", "encoder_and_decoder"]
    )
    is_distributed = torch.distributed.is_initialized()

    cuda_parallel = getattr(self, "_h3_cuda_vae_parallel", None)
    if cuda_parallel is not None:
        if getattr(self, "_h3_cuda_vae_tiles", False) and self.use_tiling:
            decoded = self._decode(z)
        else:
            decoded = _decode_sp_full(self, z, cuda_parallel)
        infer_info._route_b_saved = False
    elif is_lightning_mode and is_distributed:
        decoded = _decode_sp(self, z, _get_vae_sp_manager())
    elif is_lightning_mode:
        decoded = _decode_lightning_single(self, z)
    else:
        decoded = self._decode(z)
        infer_info._route_b_saved = False

    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)


@torch.no_grad()
def decode_ascend(self, z: torch.Tensor, return_dict: bool = True):
    decoded = self._decode_ascend(z).sample
    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)


# -----------------------------------------------------------------------
# Patch VideoProcessor.postprocess_video for MiniMax-H3.
#
# Wan's postprocess_video_ascend (globally monkey-patched onto
# VideoProcessor) returns [None] when lightning=True and
# return_output=False (streaming mode).  That is correct for Wan's
# streaming decode but wrong for MiniMax-H3 SP mode, where the full
# decoded tensor is returned and must be postprocessed normally.
#
# We restore the original upstream postprocess_video logic, bypassing
# Wan's [None] short-circuit.
# -----------------------------------------------------------------------
from diffusers.video_processor import VideoProcessor as _VP


def _mmh3_postprocess_video(self, video, output_type="np", **kwargs):
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = self.postprocess(batch_vid, output_type, **kwargs)
        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


_VP.postprocess_video = _mmh3_postprocess_video

# Monkey-patch the video VAE.
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._encode_ascend = _encode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.encode = encode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._decode_ascend = _decode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.decode = decode_ascend
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.enable_lightning = enable_lightning
autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3 = add_lightning_init(
    AutoencoderKLMiniMaxH3
)

# -----------------------------------------------------------------------
# Patch _decode_video_file: compatibility shim for av >= 12.0.0
#
# PyAV 12.0.0 removed VideoFrame.rotation. diffusers' references.py uses
# frame.rotation in _decode_video_file when loading a reference video
# (ref2va only). Since av.VideoFrame is a Cython extension type, we cannot
# add the property back. Instead, we replace _decode_video_file with a
# version that reads the DISPLAYMATRIX side data directly.
#
# Triggered: ref2va → MiniMaxH3VideoReference.from_file → _decode_video_file
# Unaffected: t2va/fl2va (no reference video, av never called)
# -----------------------------------------------------------------------
import av as _av

if not hasattr(_av.VideoFrame, "rotation"):
    import ctypes as _ctypes
    import math as _math
    import struct as _struct

    import av.sidedata.sidedata as _av_sd
    import numpy as _np

    from diffusers.modular_pipelines.minimax_h3 import references as _refs

    def _read_frame_rotation(frame):
        """Read rotation from DISPLAYMATRIX side data (av >= 12.0.0)."""
        for sd in frame.side_data.keys():
            if sd.type == _av_sd.Type.DISPLAYMATRIX:
                # 3×3 int32 matrix, 16.16 fixed-point, row-major (36 bytes)
                if sd.buffer_size >= 36:
                    raw = _ctypes.string_at(sd.buffer_ptr, 36)
                    matrix = _struct.unpack("<9i", raw)
                    cos_val = matrix[0] / (1 << 16)
                    sin_val = matrix[3] / (1 << 16)
                    angle = -_math.degrees(_math.atan2(sin_val, cos_val))
                    return round(angle / 90.0) * 90.0
        return 0.0

    def _decode_video_file_av12(media):
        """Replacement for references._decode_video_file compatible with av >= 12.0.0."""
        av = _refs._import_av()
        with _refs._local_media_file(media) as path, av.open(path) as container:
            stream = container.streams.video[0]
            frames, rotation = [], 0.0
            for frame in container.decode(stream):
                rotation = _read_frame_rotation(frame)
                frames.append(frame.to_ndarray(format="rgb24"))
            frame_rate = float(stream.average_rate or stream.guessed_rate)
            soundtrack = None
            if container.streams.audio:
                container.seek(0)
                soundtrack = _refs._decode_reference_soundtrack(av, container, container.streams.audio[0])

        if not frames:
            raise ValueError(f"No video frames to decode in {media}.")
        frames = _np.stack(frames)
        turns = round(rotation / 90.0) % 4
        if turns:
            frames = _np.ascontiguousarray(_np.rot90(frames, k=-turns, axes=(1, 2)))
        waveform, sample_rate = soundtrack if soundtrack is not None else (None, None)
        return frames, frame_rate, waveform, sample_rate

    _refs._decode_video_file = _decode_video_file_av12

# -----------------------------------------------------------------------
# Patch MiniMaxH3DecodeStep: Route B swaps VAE decode order (audio first)
# -----------------------------------------------------------------------
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3Blocks,
    MiniMaxH3DecodeStep,
)
from diffusers.modular_pipelines.minimax_h3.decoders import (
    MiniMaxH3VideoDecodeStep,
    MiniMaxH3AudioDecodeStep,
)

_original_decode_step_call = MiniMaxH3DecodeStep.__call__
_original_video_decode_step_call = MiniMaxH3VideoDecodeStep.__call__


@torch.no_grad()
def _decode_long_cuda_video(self, pipeline, state):
    backend = getattr(pipeline, "_h3_vae_decode_backend", None)
    if backend is not None:
        return backend.decode_video(self, pipeline, state)
    cpu_chunks = getattr(pipeline, "_h3_cuda_cpu_decode_chunks", False)
    if not cpu_chunks and not getattr(pipeline, "_h3_base_long_video", False):
        return _original_video_decode_step_call(self, pipeline, state)
    device = pipeline._execution_device
    block_state = self.get_block_state(state)
    if device.type != "cuda" or block_state.output_type != "pil":
        if cpu_chunks:
            raise ValueError("H3 CUDA CPU decode chunks require CUDA PIL video output")
        return _original_video_decode_step_call(self, pipeline, state)

    vae = pipeline.vae
    latent_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latent_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    latents = block_state.latents * latent_std + latent_mean
    with torch.autocast("cuda", dtype=torch.float16):
        video = vae.decode(latents, return_dict=False)[0]

    pixel_mean = torch.tensor(pipeline.pixel_mean, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(pipeline.pixel_std, device=device).view(1, -1, 1, 1, 1)
    videos = [[] for _ in range(video.shape[0])]
    # Keep the VAE and pixel math unchanged, but limit FP32 temporaries to eight frames.
    for frames in video.split(8, dim=2):
        if cpu_chunks:
            frames = frames.to(device)
        rgb = (frames.float() * pixel_std + pixel_mean).clamp(0, 1)
        if cpu_chunks and not getattr(pipeline.video_processor.config,"do_normalize",True):

            processed = rgb8_to_pil(rgb)
        else:
            processed = pipeline.video_processor.postprocess_video(rgb.cpu(), output_type="pil")
        for output, images in zip(videos, processed):
            output.extend(images)
    block_state.videos = videos
    self.set_block_state(state, block_state)
    return pipeline, state


MiniMaxH3VideoDecodeStep.__call__ = _decode_long_cuda_video


@torch.no_grad()
def _decode_step_call_route_b(self, pipeline, state):
    backend = getattr(pipeline, "_h3_vae_decode_backend", None)
    if backend is not None:
        return backend.decode_step(self, pipeline, state)
    is_lightning = (
            getattr(infer_info, "vae_lightning", "none")
            in ["decoder", "encoder_and_decoder"]
    )
    is_distributed = torch.distributed.is_initialized()

    # Non-lightning: original decode order
    # SP "full" mode: no streaming, original decode order
    # SP "chunks" / "vae_mgr" mode + single-card lightning: Route B (audio first → streaming video)
    if not is_lightning:
        return _original_decode_step_call(self, pipeline, state)
    if is_distributed and _SP_MODE == "full":
        return _original_decode_step_call(self, pipeline, state)

    # Route B：先音频后视频
    try:
        # 1. 音频 VAE decode
        audio_block = self.sub_blocks["audio"]
        pipeline, state = audio_block(pipeline, state)

        # 2. 存 WAV → 设 infer_info.audio_path
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        audio = state.get("audio")
        sampling_rate = state.get("sampling_rate")
        if rank == 0 and audio is not None and sampling_rate is not None:
            audio_path = save_audio_to_wav(
                audio, sampling_rate,
                save_path=infer_info.save_path,
            )
            infer_info.audio_path = audio_path
        else:
            infer_info.audio_path = None

        # 3. 视频 VAE decode
        #    _decode_ascend 会读 infer_info.audio_path 创建 SaveAVStream
        #    chunk 流式写入 → close() 自动清理 WAV
        video_block = self.sub_blocks["video"]
        pipeline, state = video_block(pipeline, state)

    finally:
        # 4. 清理（无论成功失败都执行）
        infer_info.audio_path = None

    return pipeline, state


MiniMaxH3DecodeStep.__call__ = _decode_step_call_route_b


# -----------------------------------------------------------------------
# Patch MiniMaxH3Blocks.get_workflow: re-wrap flattened decode blocks
#
# get_execution_blocks() recursively flattens SequentialPipelineBlocks
# containers. MiniMaxH3DecodeStep is dissolved into "decode.video" and
# "decode.audio" leaf blocks, bypassing Route B (which patches
# MiniMaxH3DecodeStep.__call__). This patch intercepts get_workflow's
# return value and re-wraps the leaf blocks back into a
# MiniMaxH3DecodeStep container, restoring Route B in the execution path.
#
# Affected: ref2va (workflow="ref2va" triggers get_workflow)
# Unaffected: t2va/fl2va (workflow=None, get_workflow not called)
# -----------------------------------------------------------------------
_original_get_workflow = MiniMaxH3Blocks.get_workflow


def _get_workflow_rewrap(self, workflow_name):
    blocks = _original_get_workflow(self, workflow_name)

    # Find flattened video/audio leaf blocks
    video_key = audio_key = None
    video_block = audio_block = None
    for key, block in list(blocks.sub_blocks.items()):
        if isinstance(block, MiniMaxH3VideoDecodeStep):
            video_key, video_block = key, block
        elif isinstance(block, MiniMaxH3AudioDecodeStep):
            audio_key, audio_block = key, block

    if video_key and audio_key:
        # Remove flattened leaves
        blocks.sub_blocks.pop(video_key)
        blocks.sub_blocks.pop(audio_key)

        # Re-wrap in MiniMaxH3DecodeStep container (Route B lives here)
        decode_step = MiniMaxH3DecodeStep()
        decode_step.sub_blocks["video"] = video_block
        decode_step.sub_blocks["audio"] = audio_block

        blocks.sub_blocks["decode"] = decode_step

    return blocks


MiniMaxH3Blocks.get_workflow = _get_workflow_rewrap
