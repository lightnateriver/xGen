"""H3 video model pipeline adaptation and internal Refiner workflow."""


# ------------------------------------------------------------------------
# pipeline.h3_regenerate.assets
# ------------------------------------------------------------------------

# Prepared-model integrity and offline refresh validation (CPU only).
import hashlib
import json
from pathlib import Path


def validate_base_schedule(steps, metadata=None, *, turbo8=False):
    allowed = (8,) if turbo8 else (20, 50)
    if type(steps) is not int or steps not in allowed:
        raise ValueError("Resident Base schedule does not match the selected inference profile")
    if metadata is not None and (int(metadata.get("num_steps", "-1")) != steps + 1
                                 or int(metadata.get("num_eval_steps", steps)) != steps):
        raise ValueError("Base AdaLN cache evaluation count differs from the configured schedule")
    return steps


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_digests(folder):
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file()
                   and p.suffix in (".json", ".safetensors", ".bin"))
    if not (folder / "config.json").is_file() or not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError(f"Missing model config or weights: {folder}")
    return {p.relative_to(folder).as_posix(): digest(p) for p in files}


def prepared_cache(folder):
    folder = Path(folder)
    record = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    name = record.get("cache_file", "adaln_cache.safetensors")
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Invalid prepared cache filename")
    cache = folder / name
    if digest(cache) != record["cache_sha256"]:
        raise ValueError("Prepared AdaLN cache checksum mismatch")
    return cache


def validate_refresh(output, fingerprint, base_digests):
    output = Path(output)
    prepared_cache(output)
    record = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if record.get("source_base_sha256") != base_digests:
        raise ValueError("Base weights changed or legacy manifest; prepare into a new directory")
    if record.get("adapters_sha256") != fingerprint:
        raise ValueError("Adapter weights changed; prepare into a new directory")
    if record.get("transformer_sha256") != model_digests(output / "transformer"):
        raise ValueError("Prepared transformer changed; prepare into a new directory")
    return record


# ------------------------------------------------------------------------
# pipeline.h3_regenerate.state
# ------------------------------------------------------------------------

# State persistence and instance-scoped temporal padding mechanisms.

from contextlib import contextmanager




@contextmanager
def base_temporal_padding(pipe, spec):
    """Apply the explicit model profile for one Base request."""
    pipe._h3_base_long_video = True
    try:
        with native_padding(pipe, spec):
            yield
    finally:
        del pipe._h3_base_long_video


@contextmanager
def native_padding(pipe, spec):
    """Permit native temporal padding on this instance, never patch a global class."""
    original = type(pipe)
    if spec['generation_frames'] / pipe.fps > pipe.max_duration:
        padded = type('H3TemporallyPaddedWorkflow', (original,), {
            'max_duration': property(lambda self: spec['generation_frames'] / self.fps)})
        pipe.__class__ = padded
    try:
        yield
    finally:
        pipe.__class__ = original


# ------------------------------------------------------------------------
# pipeline.h3.references
# ------------------------------------------------------------------------

# Ref2VA media setup with request-local reference image geometry.

from collections import Counter

import torch
from PIL import Image
from diffusers.modular_pipelines.minimax_h3.before_encoder import (
    MiniMaxH3Ref2VASetupStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames,
    resolve_canvas_size,
)
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

from x_diffusers.framework.h3_config import REFERENCE_RESIZE_MODES, reference_image_size


class H3ReferenceSetupStep(MiniMaxH3Ref2VASetupStep):
    @property
    def inputs(self):
        return super().inputs + [
            InputParam(
                name="h3_reference_image_resize_mode",
                type_hint=str,
                default=None,
                description="Ref2VA reference image sizing: diffusers, max, or match.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block = self.get_block_state(state)
        mode = block.h3_reference_image_resize_mode
        if mode is None:
            return super().__call__(components, state)
        if mode not in REFERENCE_RESIZE_MODES:
            raise ValueError(f"Invalid h3_reference_image_resize_mode: {mode!r}")

        refs = block.references
        if not refs or any(not isinstance(ref, MiniMaxH3Reference) for ref in refs):
            raise ValueError("Ref2VA requires decoded MiniMax-H3 reference objects")
        counts = Counter(ref.kind for ref in refs)
        for kind, limit in (
            ("image", self.max_images),
            ("video", self.max_videos),
            ("audio", self.max_audios),
        ):
            if counts[kind] > limit:
                raise ValueError(f"Ref2VA accepts at most {limit} {kind} references")
        if len(refs) > self.max_references or not (counts["image"] or counts["video"]):
            raise ValueError(
                f"Ref2VA requires visual media and accepts at most {self.max_references} references"
            )

        multiple = components.canvas_multiple
        height, width = block.height, block.width
        if (height is None) != (width is None):
            raise ValueError("height and width must be supplied together")
        if height is None:
            height, width = resolve_canvas_size(
                16,
                9,
                multiple,
                components.config.canvas_short_edge,
                components.config.canvas_max_pixels,
            )
        if min(height, width) <= 0 or height % multiple or width % multiple:
            raise ValueError(
                f"Output dimensions must be positive multiples of {multiple}"
            )

        frames = align_num_frames(
            block.num_frames,
            components.vae_frames_per_chunk,
            components.vae_latents_per_chunk,
        )
        duration = frames / components.fps
        if not components.min_duration <= duration <= components.max_duration:
            raise ValueError(
                f"Aligned Ref2VA duration {duration:g}s is outside the model's supported range"
            )

        block.height, block.width, block.num_frames = height, width, frames
        block.normalized_references = [
            self._normalize_reference(
                ref, components, width, height, frames, duration, mode
            )
            for ref in refs
        ]
        self.set_block_state(state, block)
        return components, state

    def _normalize_reference(
        self, ref, components, width, height, frames, duration, mode
    ):
        if ref.kind == "image":
            image = ref.image
            if not isinstance(image, Image.Image):
                raise ValueError(
                    "Reference image resize modes require a decoded PIL image"
                )
            short, long = sorted(image.size)
            if short <= 0 or long > 4 * short:
                raise ValueError(f"Invalid reference image size {image.size}")
            tw, th = reference_image_size(
                *image.size, width, height, components.canvas_multiple, mode
            )
            if image.size != (tw, th):
                image = components.image_processor.resize(image, height=th, width=tw)
            return MiniMaxH3ImageReference(image=image)

        rate = components.audio_sampling_rate
        audio = None
        if ref.has_audio:
            audio = self._normalize_audio_condition(
                ref.audio,
                ref.sample_rate if ref.sample_rate is not None else rate,
                rate,
                max_duration=duration,
            )
        if ref.kind == "video":
            pixels = self._normalize_video_condition(
                ref.frames,
                float(ref.fps),
                frames,
                components.canvas_multiple,
                components.config.canvas_short_edge,
                components.config.canvas_max_pixels,
                float(components.fps),
            )
            return MiniMaxH3VideoReference(
                frames=pixels,
                fps=float(components.fps),
                audio=audio,
                sample_rate=rate if audio is not None else None,
            )
        return MiniMaxH3AudioReference(audio=audio, sample_rate=rate)


def install_reference_setup(pipe):
    # A named Diffusers workflow exposes its flattened setup step directly.
    blocks = pipe.blocks
    for name, block in blocks.sub_blocks.items():
        if isinstance(block, MiniMaxH3Ref2VASetupStep):
            blocks.sub_blocks[name] = H3ReferenceSetupStep()
            pipe._blocks = blocks
            return
    raise RuntimeError(
        "The Diffusers Ref2VA workflow does not expose its reference setup step"
    )


# ------------------------------------------------------------------------
# pipeline.h3.staging
# ------------------------------------------------------------------------

# Optional whole-module staging around H3 reference conditioning.
#
# Use the existing module device contract; do not rebuild or requantize weights.
# On encoder failure keep DiT on CPU and preserve the original error for the caller.

from functools import wraps
import gc
import logging
import time


logger = logging.getLogger(__name__)


def install_conditioning_staging(pipe, device, *, defer_until_dit=False):
    accelerator = torch.get_device_module(torch.device(device).type)
    model = pipe.transformer
    encoder = pipe.text_encoder.model
    if getattr(encoder, "_h3_conditioning_staging_installed", False):
        raise ValueError("H3 conditioning staging is already installed")
    metrics = dict(calls=0, d2h_seconds=0.0, h2d_seconds=0.0, last=None)
    pipe._h3_cuda_conditioning_staging = metrics
    original = encoder.forward

    def restore(event):
        accelerator.synchronize()
        gc.collect()
        accelerator.empty_cache()
        event["status"] = "restoring"
        started = time.perf_counter()
        model.to(device)
        accelerator.synchronize()
        event["h2d_seconds"] = time.perf_counter() - started
        metrics["h2d_seconds"] += event["h2d_seconds"]
        event.update(status="completed", allocated_after=accelerator.memory_allocated())
        logger.info(
            "H3 conditioning staging (weights preserved, no reload/quantization): %s",
            event,
        )

    if defer_until_dit:

        def before_dit(module, inputs):
            event = metrics["last"]
            if event is not None and event["status"] == "awaiting_reference_encoding":
                restore(event)

        model.register_forward_pre_hook(before_dit)

    @wraps(original)
    def forward(*values, **kwargs):
        event = dict(
            status="offloading", allocated_before=accelerator.memory_allocated()
        )
        metrics["last"] = event
        metrics["calls"] += 1
        accelerator.synchronize()
        started = time.perf_counter()
        model.to("cpu")
        accelerator.synchronize()
        event["d2h_seconds"] = time.perf_counter() - started
        metrics["d2h_seconds"] += event["d2h_seconds"]
        gc.collect()
        accelerator.empty_cache()
        event.update(
            status="encoding", allocated_during_encoding=accelerator.memory_allocated()
        )
        try:
            output = original(*values, **kwargs)
        except BaseException:
            event["status"] = "encoder_failed_dit_on_cpu"
            logger.exception(
                "H3 conditioning failed; DiT remains on CPU, original failure propagated"
            )
            raise
        if defer_until_dit:
            # ModularPipeline must still infer CUDA from its existing group
            # offload hooks, so reference pixels and request noise stay on CUDA.
            if torch.device(pipe._execution_device).type != torch.device(device).type:
                event["status"] = "invalid_execution_device_dit_on_cpu"
                raise RuntimeError(
                    "Deferred H3 reference staging requires the pipeline execution device to match the accelerator"
                )
            event.update(
                status="awaiting_reference_encoding",
                restore_boundary="before_first_dit",
            )
            logger.info(
                "H3 reference staging: DiT remains on CPU through reference encoding"
            )
        else:
            restore(event)
        return output

    encoder.forward = forward
    encoder._h3_conditioning_staging_installed = True


# ------------------------------------------------------------------------
# pipeline.h3.overlap
# ------------------------------------------------------------------------

# Opt-in Ref2VA encoder overlap on the native modular pipeline, CUDA or NPU.

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta

import math
import os
import threading

import weakref




def _destroy_groups(groups):
    import torch.distributed as dist

    if dist.is_initialized():
        for group in reversed(groups):
            dist.destroy_process_group(group)


class _CPUTransport:
    """Transport native VAE tile bytes without concurrent accelerator collectives."""

    def __init__(self, owner):
        self.owner = owner

    @property
    def group(self):
        return self.owner.vae_group

    def gather(self, received, payload):
        import torch
        import torch.distributed as dist

        host = payload.detach().cpu()
        buffers = [torch.empty_like(host) for _ in received]
        dist.all_gather(buffers, host, group=self.group)
        for target, source in zip(received, buffers):
            target.copy_(source)
        self.owner.metrics["vae_cpu_gathers"] += 1
        self.owner.metrics["vae_payload_bytes"] += host.numel() * host.element_size()


class ConditionOverlap:
    def __init__(self, pipe, *, sp, timeout):
        import torch
        import torch.distributed as dist

        self.pipe = weakref.proxy(pipe)
        self.timeout = timeout
        self.device = torch.device(
            device_type(), torch.get_device_module(device_type()).current_device()
        )
        self.accelerator = torch.get_device_module(self.device.type)
        self.stream = self.accelerator.Stream(device=self.device)
        self.lock = threading.Lock()
        self.text_done = threading.Event()
        self.cancelled = threading.Event()
        self.failed = self.active = False
        self.future = self.pool = None
        self.metrics = {}
        self.vae_group = self.audio_gate = None
        groups = []
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world != sp:
            raise ValueError(
                "h3_condition_overlap requires one model SP group, without extra DP ranks"
            )
        if world > 1:
            if not dist.is_gloo_available():
                raise ValueError(
                    "h3_condition_overlap requires CPU Gloo for multi-rank execution"
                )
            try:
                for _ in range(2):
                    groups.append(
                        dist.new_group(
                            ranks=list(range(world)),
                            backend="gloo",
                            timeout=timedelta(seconds=self.timeout),
                        )
                    )
                self.vae_group, self.audio_gate = groups
            except BaseException:
                _destroy_groups(groups)
                raise
        weakref.finalize(pipe, _destroy_groups, groups)
        self.transport = _CPUTransport(self)
        self._install_blocks(pipe)
        self._guard_audio(pipe)

    def _install_blocks(self, pipe):
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VATextEncoderStep as Text,
            MiniMaxH3Ref2VAReferenceEncoderStep as Reference,
        )

        owner = self

        class OverlapText(Text):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def __call__(self, components, state):
                return owner.text(self.original, components, state)

        class JoinReference(Reference):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def __call__(self, components, state):
                return owner.reference(self.original, components, state)

        found = []

        def visit(block):
            for name, child in getattr(block, "sub_blocks", {}).items():
                if isinstance(child, (Text, Reference)):
                    found.append((block, name, child))
                else:
                    visit(child)

        visit(pipe._blocks)
        if (
            sum(isinstance(child, Text) for _, _, child in found) != 1
            or sum(isinstance(child, Reference) for _, _, child in found) != 1
        ):
            raise ValueError(
                "Unsupported Ref2VA encoder block layout for condition overlap"
            )
        self.replacements = []
        for parent, name, original in found:
            wrapped = (
                OverlapText(original)
                if isinstance(original, Text)
                else JoinReference(original)
            )
            self.replacements.append((parent, name, original, wrapped))
            if isinstance(original, Reference):
                self.original_reference = original

    def _guard_audio(self, pipe):
        from functools import wraps

        original = pipe.audio_vae.encode

        @wraps(original)
        def encode(*values, **kwargs):
            if self.active:
                if not self.text_done.wait(self.timeout):
                    raise TimeoutError(
                        "Text completion gate for reference audio timed out"
                    )
                if self.cancelled.is_set():
                    raise RuntimeError(
                        "Reference encoding cancelled after text preparation failed"
                    )
            return original(*values, **kwargs)

        self.original_audio_encode, self.audio_encode = original, encode

    def _run_reference(self, components, shadow, inference_mode):
        import torch

        vae = components.vae
        previous = getattr(vae, "_h3_overlap_transport", None)
        vae._h3_overlap_transport = self.transport
        started = time.perf_counter()
        try:
            with (
                torch.inference_mode(inference_mode),
                torch.no_grad(),
                self.accelerator.device(self.device),
                self.accelerator.stream(self.stream),
            ):
                _, shadow = self.original_reference(components, shadow)
                self.stream.synchronize()
            return shadow
        finally:
            if previous is None:
                del vae._h3_overlap_transport
            else:
                vae._h3_overlap_transport = previous
            self.metrics["reference_wall_s"] = time.perf_counter() - started

    def text(self, original, components, state):
        import torch
        import torch.distributed as dist

        refs = state.get("normalized_references")
        if not self.active or not refs:
            return original(components, state)
        if self.future is not None:
            raise RuntimeError(
                "Reference encoding was already submitted for this request"
            )
        shadow = type(state)(
            values=dict(state.values), kwargs_mapping=deepcopy(state.kwargs_mapping)
        )
        self.stream.wait_stream(self.accelerator.current_stream(self.device))
        self.future = self.pool.submit(
            self._run_reference, components, shadow, torch.is_inference_mode_enabled()
        )
        self.metrics["submitted"] += 1
        started = time.perf_counter()
        try:
            result = original(components, state)
            if any(ref.has_audio for ref in refs):
                # Audio keeps its native collective order, after all text ranks finish.
                self.accelerator.current_stream(self.device).synchronize()
                if self.audio_gate is not None:
                    dist.barrier(group=self.audio_gate)
            self.text_done.set()
            return result
        except BaseException:
            self.cancelled.set()
            self.text_done.set()
            raise
        finally:
            self.metrics["text_wall_s"] = time.perf_counter() - started

    def reference(self, original, components, state):
        if not self.active or self.future is None:
            return original(components, state)
        started = time.perf_counter()
        shadow = self.future.result(timeout=self.timeout)
        self.future = None
        block = original.get_block_state(state)
        block.condition_latents = shadow.get("condition_latents")
        block.audio_condition_latents = shadow.get("audio_condition_latents")
        original.set_block_state(state, block)
        self.metrics["joined"] += 1
        self.metrics["join_wait_s"] = time.perf_counter() - started
        return components, state

    @contextmanager
    def request(self):
        if self.failed:
            raise RuntimeError(
                "Condition-overlap worker failed; restart the manager/service"
            )
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("One Ref2VA pipeline cannot prepare concurrent requests")
        self.metrics = dict(
            backend=self.device.type,
            submitted=0,
            joined=0,
            text_wall_s=0.0,
            reference_wall_s=0.0,
            join_wait_s=0.0,
            vae_cpu_gathers=0,
            vae_payload_bytes=0,
        )
        self.cancelled.clear()
        self.text_done.clear()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-reference")
        for parent, name, original, wrapped in self.replacements:
            parent.sub_blocks[name] = wrapped
        self.pipe.audio_vae.encode = self.audio_encode
        self.active = True
        error = None
        try:
            yield self
        except BaseException as exc:
            error = exc
            self.failed = True
            raise
        finally:
            self.cancelled.set()
            self.text_done.set()
            drain_error = None
            try:
                if isinstance(error, FutureTimeout):
                    drain_error = error
                elif self.future is not None:
                    self.future.result(timeout=self.timeout)
            except BaseException as exc:
                drain_error = exc
                self.failed = True
            finally:
                # A timed-out worker must be retired, never reused or retried here.
                self.pool.shutdown(
                    wait=not isinstance(drain_error, FutureTimeout), cancel_futures=True
                )
                self.pool = self.future = None
                self.active = False
                for parent, name, original, wrapped in self.replacements:
                    parent.sub_blocks[name] = original
                self.pipe.audio_vae.encode = self.original_audio_encode
                self.metrics["failed"] = self.failed
                self.pipe._h3_condition_overlap_metrics = dict(self.metrics)
                self.lock.release()
                logger.info("H3 condition overlap: %s", self.metrics)
            if drain_error is not None and error is None:
                raise drain_error


# ------------------------------------------------------------------------
# pipeline.h3.execution
# ------------------------------------------------------------------------

# Apply H3 component wiring from an explicit resolved execution configuration.

from x_diffusers.framework.h3_config import H3ExecutionOptions



from x_diffusers.framework.offload import retain_module
from x_diffusers.framework.vae.minimax_h3 import offload_audio
from x_diffusers.framework.h3_config import condition_parallel_key




def configure_model(pipe, options: H3ExecutionOptions, device):
    """Install scheduling independently of quantization and attention kernels."""
    import torch
    from x_diffusers.framework.transformer.minimax_h3 import initialize
    from x_diffusers.framework.transformer.minimax_h3 import prepare_block
    from x_diffusers.framework.transformer.minimax_h3 import prepare_shared_block
    from x_diffusers.framework.offload import retain_module

    model = pipe.transformer
    config = None
    if getattr(options, "h3_parallel_mode", "sp") != "sp":
        config = initialize(model, options.parallel)
        from x_base import attention_manager

        def attention(q, k, v, *, attn_mask=None, **kwargs):
            return attention_manager.attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        for block in model.transformer_blocks:
            block.attn._h3_attention_backend = attention
            if config.mode == "tp_cp_shared":
                prepare_shared_block(
                    block,
                    model._h3_shared_tp_manager,
                    model._h3_shared_sequence_exchange,
                    fp8=False,
                    chunks=options.phaa_num,
                )
            elif config.mode != "tp_sp" and config.tp > 1:
                prepare_block(block, model.parallel_manager, config, fp8=False)
            elif config.mode == "tp_sp":
                # The pair adapter expects fused QKV before slicing.
                block.attn.fuse_projections()
        if config.mode == "tp_sp":
            from x_diffusers.framework.transformer.minimax_h3 import install_pair_parallel

            install_pair_parallel(model)
    for name in ("local_input", "local_output"):
        setattr(
            model, "_h3_cuda_" + name, bool(getattr(options, "h3_cuda_" + name, False))
        )
    if model._h3_cuda_local_input:
        for block in model.transformer_blocks:
            block.attn._h3_tp_cp = True
    model._h3_precompute_conditioning = bool(
        getattr(options, "h3_cuda_precompute_conditioning", False)
    )
    if getattr(options, "h3_cuda_input_refiner", "none") != "none":
        from x_diffusers.framework.transformer.minimax_h3 import install_input_refiner

        install_input_refiner(
            model,
            options.h3_cuda_input_refiner,
            sp=options.h3_cuda_input_refiner_sp,
            attention=options.h3_cuda_input_refiner_attention,
            ffn=options.h3_cuda_input_refiner_ffn,
        )
    if getattr(options, "h3_cuda_refiner_ff_chunk_size", 0):
        from x_diffusers.framework.transformer.minimax_h3 import install_ff_chunking

        install_ff_chunking(model, options.h3_cuda_refiner_ff_chunk_size)
    # Keep original modules/quantization; remove competing offload hooks only
    # after existing quantization has run. Subsequent requests reuse weights.
    if not getattr(options, "h3_refiner", False):
        retain_module(model, torch.device(device))
    model._h3_portable_scheduling = True
    logger.info(
        "H3 portable scheduling installed: backend=%s mode=%s conditioning=%s",
        torch.device(device).type,
        getattr(options, "h3_parallel_mode", "sp"),
        model._h3_precompute_conditioning,
    )


def configure_components(pipe, options: H3ExecutionOptions, device):
    import torch
    from functools import wraps
    from x_diffusers.framework.vae.minimax_h3 import install_condition_parallel

    coordinator_owned = bool(getattr(options, "h3_refiner", False))
    if not coordinator_owned and getattr(
        options, "h3_cuda_vae_encoder_parallel", False
    ):
        # Quantization may create CPU decoder buffers after the loader's VAE
        # transfer. The encoder's root offload hook makes pipeline.to() skip
        # this complete VAE, so finish the native decoder transfer first.
        pipe.vae.decoder.to(device)
        pipe.vae.post_quant_conv.to(device)
    install_condition_parallel(
        pipe, options, device, stage_weights=not coordinator_owned
    )
    if not coordinator_owned and (
        getattr(options, "h3_cuda_audio_encoder_parallel", False)
        or getattr(options, "h3_cuda_audio_decoder_parallel", "none") != "none"
    ):
        # Reference encoding returns audio weights to CPU. The native decoder
        # has no onload hook, unlike the CUDA FP8 recipe's decoder wrapper.
        from x_diffusers.framework.vae.minimax_h3 import offload_audio

        audio = pipe.audio_vae
        original_audio_decode = audio.decode

        @wraps(original_audio_decode)
        def decode_audio(*values, **kwargs):
            audio.to(torch.device(device))
            try:
                return original_audio_decode(*values, **kwargs)
            finally:
                offload_audio(audio)

        audio.decode = decode_audio
    if getattr(options, "h3_cuda_conditioning_offload", False):
        install_conditioning_staging(
            pipe,
            device,
            defer_until_dit=bool(getattr(options, "h3_cuda_reference_offload", False)),
        )
    original = pipe.vae.decode

    def release():
        from x_diffusers.framework.transformer.minimax_h3 import release_conditioning

        release_conditioning(pipe.transformer)

    previous = getattr(pipe, "_h3_prepare_video_decode", None)

    def prepare():
        release()
        if previous is not None:
            previous()

    pipe._h3_prepare_video_decode = prepare

    @wraps(original)
    def decode(*values, **kwargs):
        release()
        return original(*values, **kwargs)

    pipe.vae.decode = decode


def prepare_vaes(pipe, options, device):
    from diffusers.hooks import apply_group_offloading
    from x_diffusers.framework.vae.minimax_h3 import install_cpu_decode_chunks
    from x_diffusers.framework.vae.minimax_h3 import install_staged_decoder

    omni_decode = getattr(options, "h3_vae_backend", "legacy") == "native-stream"
    if not omni_decode:
        install_cpu_decode_chunks(pipe.vae)
        pipe._h3_cuda_cpu_decode_chunks = True
    apply_group_offloading(
        pipe.vae,
        offload_type="leaf_level",
        onload_device=device,
        offload_device=torch.device("cpu"),
        use_stream=False,
        low_cpu_mem_usage=True,
    )
    if not omni_decode:
        install_staged_decoder(pipe.vae, device, retain_module)
    # Audio weight_norm pre-hooks run before leaf onload hooks. Move the complete
    # module before decode instead, as required by its upstream offload contract.
    audio = pipe.audio_vae
    audio.to(torch.device("cpu"))
    original_audio_decode = audio.decode

    @wraps(original_audio_decode)
    def decode_audio(*values, **kwargs):
        audio.to(device)
        try:
            return original_audio_decode(*values, **kwargs)
        finally:
            offload_audio(audio)

    audio.decode = decode_audio
    original = pipe.vae.decode

    def prepare_video_decode():
        import gc
        import time

        if getattr(
            getattr(pipe, "transformer", None), "_h3_precompute_conditioning", False
        ):
            from x_diffusers.framework.transformer.minimax_h3 import release_conditioning

            release_conditioning(pipe.transformer)
        started = time.perf_counter()
        torch.cuda.synchronize()
        before = torch.cuda.memory_reserved()
        gc.collect()
        torch.cuda.empty_cache()
        event = dict(
            seconds=time.perf_counter() - started,
            reserved_before=before,
            reserved_after=torch.cuda.memory_reserved(),
            allocated=torch.cuda.memory_allocated(),
        )
        pipe._h3_cuda_decode_trim = event
        logger.info(
            "H3 CUDA before video VAE decode: unused memory reclaimed, DiT retained: %s",
            event,
        )

    # Both old and native decoders must release request-scoped conditioning
    # and reclaim unused allocations before staging decoder weights.
    pipe._h3_prepare_video_decode = prepare_video_decode

    @wraps(original)
    def decode(*values, **kwargs):
        pipe._h3_prepare_video_decode()
        return original(*values, **kwargs)

    pipe.vae.decode = decode
    if (
        getattr(options, "h3_parallel_mode", "sp") in ("tp_cp", "tp_cp_shared", "tp_sp")
        or getattr(options, "task_type", None) == "ref2va"
    ):
        from x_diffusers.framework.transformer.minimax_h3 import stage_reference_audio

        stage_reference_audio(pipe, device)
    if getattr(options, "h3_cuda_conditioning_offload", False):

        install_conditioning_staging(
            pipe,
            device,
            defer_until_dit=bool(getattr(options, "h3_cuda_reference_offload", False)),
        )
    if (
        any(condition_parallel_key(options)[:3])
        or condition_parallel_key(options)[-1] != "none"
    ):
        from x_diffusers.framework.vae.minimax_h3 import install_condition_parallel

        install_condition_parallel(pipe, options, device)
    logger.info(
        "H3 CUDA VAE backend=%s; DiT resident blocks remain on device through decode",
        getattr(options, "h3_vae_backend", "legacy"),
    )


# ------------------------------------------------------------------------
# pipeline.h3_regenerate.pipeline
# ------------------------------------------------------------------------

# Experimental latent SR backend for XGEN's existing managers (no request loop).
import functools


from types import SimpleNamespace



import torch.distributed as dist

from x_base.device import device_type
from x_base.infinite_vram.immutable_weights import CpuWeights
from x_base.utils.stage_metrics import Telemetry
from x_diffusers.framework.h3_config import schedule_points
from x_diffusers.framework.h3_config import (
    refiner_shape, H3RefinerConfig, H3RefinerRequest, H3RefinerResult, H3RefinerOutput,
)




def prepare_vae_precision(vae, backend="npu"):
    """Override from_pretrained's keep-in-FP32 Conv policy before BF16 tile inputs."""
    vae.to(dtype=torch.bfloat16)
    if backend == 'cuda':
        return vae
    for module in vae.decoder.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.RMSNorm)):
            module.float()
    return vae


class H3RefinerPipeline:
    """Borrow the manager's Base; own only Refiner components and stage residency."""
    def __init__(self, base, config: H3RefinerConfig):
        import x_diffusers.framework._vendor.h3_regenerate.upscaler as upscaler
        import x_diffusers.framework._vendor.h3_regenerate.vae_fast as vae_fast
        from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel
        from diffusers.models.attention_dispatch import AttentionBackendName
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
        from x_base.adaln_cache import install_adaln_cache, release_adaln_weights, validate_stripped_checkpoint
        from x_base.phaa.utils import inject_phaa_for_minimax_h3
        self.base, self.config = base, config
        self.backend = config.backend
        self.accelerator = torch.get_device_module(self.backend)
        self.rank = dist.get_rank()
        self.failed = False
        self.completed = 0
        self.startup_seconds = None
        self.metrics = Telemetry()
        turbo8 = config.turbo8
        self.base_steps = validate_base_schedule(config.base_steps, turbo8=turbo8)
        if config.base_cache_file:
            from safetensors import safe_open
            with safe_open(config.base_cache_file, framework='pt') as cache:
                validate_base_schedule(config.base_steps, cache.metadata() or {}, turbo8=turbo8)
        weights, prepared = Path(config.weights), Path(config.prepared)
        self.base.audio_vae.float()
        self.base_weights = CpuWeights(base.transformer, 'h3.weights', self.metrics)
        self.base_vae = CpuWeights(base.vae, 'h3.reference_vae', self.metrics)
        self.base_audio = CpuWeights(base.audio_vae, 'h3.audio_vae', self.metrics)
        self.calls = dict(h3=0, refiner=0)
        self.original_forwards = []
        self._instrument(base.transformer, 'h3')
        self.refiner = MiniMaxH3Transformer3DModel.from_pretrained(
            str(prepared/'transformer'), torch_dtype=torch.bfloat16, local_files_only=True).eval()
        validate_stripped_checkpoint(self.refiner, str(prepared/'transformer'))
        install_adaln_cache(SimpleNamespace(transformer=self.refiner), str(prepared_cache(prepared)), 'none')
        release_adaln_weights(self.refiner)
        self.refiner.enable_parallel(dp_size=1, sp_size=8)
        inject_phaa_for_minimax_h3(self.refiner, 2)
        self.ref_weights = CpuWeights(self.refiner, 'refiner.weights', self.metrics)
        self.ref_weights.stream_blocks(config.loading)
        self._instrument(self.refiner, 'refiner')
        self.text = MiniMaxH3TextEncoderStep().init_pipeline(str(weights/'base_model'))
        self.text.update_components(text_encoder=base.text_encoder, tokenizer=base.tokenizer, processor=base.processor)
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(str(weights/'base_model'), subfolder='vae',
            torch_dtype=torch.bfloat16, local_files_only=True).eval()
        prepare_vae_precision(self.vae, self.backend)
        if self.backend == 'npu':
            self.vae.set_attention_backend(AttentionBackendName._NATIVE_NPU)
        self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                              tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
        vae_fast.install(self.vae, batch=1, distributed=True)
        for field in ('latents_mean', 'latents_std'):
            if list(getattr(base.vae.config, field)) != list(getattr(self.vae.config, field)):
                raise ValueError('H3/Refiner VAE normalization differs: '+field)
        self.vae_weights = CpuWeights(self.vae, 'refiner.vae_weights', self.metrics)
        self.node = upscaler.load(weights/'latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
        if self.backend == 'npu':
            resolve = self.node._resolve_device
            self.node._resolve_device = lambda backend: torch.device('npu') if backend == 'cuda' else resolve(backend)
        self.node.load_model('upscaler', torch.device('cpu'), 'bf16')
        self.upscale_model = next(iter(self.node.MODEL_CACHE.values()))
        self.upscale_weights = CpuWeights(self.upscale_model, 'upscaler.weights', self.metrics)
        self.node.load_model = lambda *unused: self.upscale_model
        self.base_weights.onload()
        self.identity = self.identities()
        logger.info('H3_REFINER_ASSETS_READY cpu_master_bytes=%s refiner_dtype=bf16 refiner_steps=3',
                    sum(h.nbytes for h in self.holders()))

    def holders(self):
        return (self.base_weights, self.base_vae, self.base_audio, self.ref_weights,
                self.vae_weights, self.upscale_weights)

    def identities(self):
        return {h.name: h.fingerprint() for h in self.holders()}

    def _instrument(self, model, name):
        forward = model.forward
        self.original_forwards.append((model, forward))
        @functools.wraps(forward)
        def measured(*args, **kwargs):
            self.calls[name] += 1
            with self.metrics.span(name+'.dit', sync=False):
                return forward(*args, **kwargs)
        model.forward = measured

    def close(self):
        """Release only this backend's holders/hooks; changing backend requires restart."""
        for holder in self.holders():
            holder.close()
        for model, forward in self.original_forwards:
            model.forward = forward
        self.failed = True

    @torch.inference_mode()
    def _draft(self, request: H3RefinerRequest, spec):
        self.base_weights.onload()
        self.base_vae.onload()
        params = dict(prompt=request.prompt, width=1344, height=768, num_frames=spec['generation_frames'],
                      num_inference_steps=schedule_points(self.config.base_steps),
                      generator=torch.Generator().manual_seed(request.seed), output=['latents', 'audio_latents'])
        reference_audio = request.reference_audio
        params.update(request.media)
        original = self.base._blocks
        blocks = self.base.blocks
        blocks.sub_blocks.pop('decode')
        self.base._blocks = blocks
        try:
            if reference_audio:
                self.base_audio.onload()
            from contextlib import nullcontext
            scope = nullcontext()
            if getattr(self.base.transformer, '_h3_precompute_conditioning', False) is True:
                from x_diffusers.framework.transformer.minimax_h3 import request_conditioning
                scope = request_conditioning(self.base.transformer)
            with scope, native_padding(self.base, spec), self.metrics.span('h3.pipeline'):
                result = self.base(**params)
        finally:
            self.base._blocks = original
            if reference_audio:
                self.base_audio.offload()
        video, audio = result['latents'].float().cpu(), result['audio_latents'].float().cpu()
        if tuple(video.shape) != (1,24,spec['latent_frames'],48,84) or not torch.isfinite(video).all():
            raise ValueError('Unexpected or nonfinite normalized H3 video latent')
        if tuple(audio.shape) != (2,32,spec['audio_frames']) or not torch.isfinite(audio).all():
            raise ValueError('Unexpected or nonfinite H3 audio latent')
        self.base_vae.offload()
        self.base_weights.offload()
        return video[0], audio

    @torch.inference_mode()
    def _conditions(self, request: H3RefinerRequest, spec, native):
        import x_diffusers.framework._vendor.h3_regenerate.upscaler as upscaler
        self.base_audio.onload()
        av = self.base.audio_vae
        with self.metrics.span('refiner.silence_encode'):
            silence = av.encode(torch.zeros(2,1,round(spec['generation_frames']/24*32000),device=self.backend),
                                return_dict=False)[0].mode()[...,:spec['audio_frames']]
            if silence.shape[-1] != spec['audio_frames']:
                raise ValueError('Short silent conditioning')
            mean = silence.new_tensor(av.config.latents_mean).view(1,-1,1)
            std = silence.new_tensor(av.config.latents_std).view(1,-1,1)
            audio = ((silence-mean)/std).bfloat16().cpu()
        self.base_audio.offload()
        with self.metrics.span('refiner.text_encode'):
            text = self.text(prompt=request.prompt, output=['prompt_embeds','text_token_tags'])
            text = dict(prompt_embeds=text['prompt_embeds'][0].cpu(),
                        text_token_tags=text['text_token_tags'].flatten().cpu())
        self.upscale_weights.onload()
        with self.metrics.span('upscaler.compute'):
            low = upscaler.upscale(self.node, native, (24,spec['latent_frames'],90,160)).cpu()
        self.upscale_weights.offload()
        return dict(upscaled=low, audio=audio, **text)

    @torch.inference_mode()
    def _refine(self, request: H3RefinerRequest, spec, conditions):
        import x_diffusers.framework._vendor.h3_regenerate.geometry as geometry
        from x_diffusers.framework._vendor.h3_regenerate.sigmas import SCHEDULES
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        self.ref_weights.onload(self.ref_weights.nonblocks, 'refiner.initial_h2d_wait')
        audio = geometry.pack_audio_latents(conditions['audio'][None]).to(self.backend)
        shape = geometry.MiniMaxH3Geometry(spec['latent_frames'],90,160,conditions['audio'].shape[-1])
        layout = geometry.build_t2va_layout(conditions['text_token_tags'],shape).to(self.backend)
        prompt = conditions['prompt_embeds'][None].to(self.backend)
        low = patchify_video_latents(conditions['upscaled'][None].to(self.backend),shape.patch_size).reshape(
            1,shape.num_video_rows,shape.video_row_dim).float()
        state = (1-SCHEDULES[3][0])*low+SCHEDULES[3][0]*torch.randn(low.shape,device=self.backend,
            generator=torch.Generator(device=self.backend).manual_seed(request.seed),dtype=torch.float32)
        for index,(sigma,following) in enumerate(zip(SCHEDULES[3][:-1],SCHEDULES[3][1:])):
            times,indices = geometry.build_row_timestep_plan(layout,video_sigma=sigma,audio_sigma=0.,keyframe_noise_aug=1.)
            with self.metrics.span('refiner.step',step=index):
                if self.config.loading == 'prefetch':
                    self.ref_weights.prefetch.begin_step()
                velocity,_ = self.refiner(hidden_states=state.bfloat16(),audio_hidden_states=audio,
                    encoder_hidden_states=prompt,timestep=times,timestep_indices=indices,
                    token_tags=layout.token_tags,position_ids=layout.position_ids,
                    video_indices=layout.video_indices,audio_indices=layout.audio_indices,
                    text_indices=layout.text_indices,attention_kwargs={},return_dict=False)
                state = state+(sigma-following)*velocity.float()
                if self.config.loading == 'prefetch':
                    self.ref_weights.prefetch.end_step()
        final = geometry.unpatchify_video_rows(state,shape)[0].cpu()
        if not torch.isfinite(final).all():
            raise ValueError('Nonfinite Refiner output')
        if self.config.loading == 'prefetch':
            self.ref_weights.prefetch.release()
        self.ref_weights.offload()
        return final

    @torch.inference_mode()
    def _decode(self, request: H3RefinerRequest, spec, final, audio, output):
        import x_diffusers.framework.vae.minimax_h3 as native_vae
        waveform = None
        if request.enable_audio:
            self.base_audio.onload()
            with self.metrics.span('audio.decode'):
                latent = audio.to(self.backend)
                av = self.base.audio_vae
                mean = latent.new_tensor(av.config.latents_mean).view(1,-1,1)
                std = latent.new_tensor(av.config.latents_std).view(1,-1,1)
                waveform = av.decode(latent*std+mean,return_dict=False)[0].float().permute(1,0,2).cpu()
                if not torch.isfinite(waveform).all():
                    raise ValueError('Nonfinite decoded H3 audio')
            self.base_audio.offload()
        sink = output if self.rank == 0 else None
        if sink is not None:
            sink.start(waveform, 32000, spec['output_frames'])
        self.vae_weights.onload()
        try:
            with self.metrics.span('vae.decode'):
                latent = final[None].to(self.backend).float()
                mean = latent.new_tensor(self.vae.config.latents_mean).view(1,24,1,1,1)
                std = latent.new_tensor(self.vae.config.latents_std).view(1,24,1,1,1)
                pixels = native_vae._decode_chunks_streaming(self.vae,(latent*std+mean).bfloat16(),None,None,
                                                            frame_callback=sink.enqueue if sink else None)
                if pixels.shape[2] < spec['output_frames']:
                    raise ValueError('Decoded video is shorter than requested')
                videos = pixels[:, :, :spec['output_frames']].cpu() if sink is None and self.rank == 0 else None
                del pixels, latent
            if sink:
                sink.finish()
        finally:
            self.vae_weights.offload()
        return videos, waveform

    def __call__(self, request: H3RefinerRequest, *, output: H3RefinerOutput | None = None):
        from x_base import rope_manager
        from x_base.phaa import get_phaa_split_num, set_phaa_split_num
        if self.failed:
            raise RuntimeError('H3 Refiner worker is failed; restart required')
        spec = refiner_shape(request.seconds)
        old_rope, old_phaa = rope_manager.is_rope_fused, get_phaa_split_num()
        started = time.perf_counter()
        self.metrics.reset()
        self.calls = dict(h3=0,refiner=0)
        self.accelerator.reset_peak_memory_stats()
        try:
            torch.manual_seed(request.seed)
            self.accelerator.manual_seed(request.seed)
            set_phaa_split_num(request.phaa_num)
            native, audio = self._draft(request,spec)
            rope_manager.is_rope_fused = False
            conditions = self._conditions(request,spec,native)
            set_phaa_split_num(2)
            final = self._refine(request,spec,conditions)
            videos, waveform = self._decode(request,spec,final,audio,output)
            self.base_weights.onload(label='h3.restore_for_next_request_h2d_wait')
            if self.calls != dict(h3=self.base_steps,refiner=3):
                raise RuntimeError('Unexpected actual DiT counts: '+str(self.calls))
            if self.identities() != self.identity:
                raise RuntimeError('Resident model/CPU master identity changed')
            record = dict(experimental=True, startup_seconds=self.startup_seconds,
                request_seconds=time.perf_counter()-started, actual_dit=dict(self.calls),
                profile=spec, phases=self.metrics.resolve(), completed_requests=self.completed+1,
                peak_allocated_bytes=self.accelerator.max_memory_allocated(),peak_reserved_bytes=self.accelerator.max_memory_reserved(),
                allocated_after=self.accelerator.memory_allocated(),reserved_after=self.accelerator.memory_reserved(),
                model_and_cpu_master_reused=True)
            self.completed += 1
            logger.info('H3_REFINER_COMPLETE rank=%s seconds=%.3f base_dit=%s refiner_dit=3',
                        self.rank,record['request_seconds'],self.calls['h3'])
            return H3RefinerResult(metrics=record, videos=videos, audio=waveform)
        except Exception:
            self.failed = True
            logger.exception('H3_REFINER_FAILED rank=%s retry=false restart_required=true',self.rank)
            raise
        finally:
            self.metrics.active = False
            rope_manager.is_rope_fused = old_rope
            set_phaa_split_num(old_phaa)
