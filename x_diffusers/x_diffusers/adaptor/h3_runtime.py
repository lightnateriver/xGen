"""Shared H3 request preparation and execution for CLI and HTTP adapters."""
from pathlib import Path


from contextlib import nullcontext
import gc
import logging
import os
import time

from .validation import h3_schedule_points as schedule_points

logger = logging.getLogger(__name__)


def release_request_memory():
    """Keep resident weights; callers must first release their request outputs."""
    import torch
    from x_base.device import device_type

    started = time.perf_counter()
    accelerator = torch.get_device_module(device_type())
    gc.collect()
    accelerator.synchronize()
    allocated = accelerator.memory_allocated()
    reserved = accelerator.memory_reserved()
    accelerator.empty_cache()
    accelerator.synchronize()
    logger.info(
        "H3_REQUEST_MEMORY_RELEASE device=%s seconds=%.3f allocated_before=%d "
        "allocated_after=%d reserved_before=%d reserved_after=%d",
        device_type(), time.perf_counter() - started, allocated,
        accelerator.memory_allocated(), reserved, accelerator.memory_reserved(),
    )


def load_keyframes(args, parameters):
    """Load ordered first/last frames without changing the request configuration."""
    first = getattr(args, "keyframe_image", "")
    last = getattr(args, "keyframe_last_image", "")
    if first or last:
        from diffusers.utils import load_image

        if first:
            parameters["image"] = load_image(first)
        if last:
            parameters["last_image"] = load_image(last)


def validate_turbo_sla_schedule(pipe, args, points):
    """Verify the resident Turbo-SLA adapter and four-step request schedule."""
    from .validation import validate_h3_turbo_sla

    validate_h3_turbo_sla(args)
    if not getattr(args, "h3_turbo_sla_lora", None):
        return
    if getattr(pipe, "_h3_turbo_sla_path", None) != os.path.abspath(args.h3_turbo_sla_lora):
        raise RuntimeError("Turbo-SLA adapter changed; rebuild the pipeline before inference.")
    # Five sigma points -> four forwards. Verify upstream semantics on every request.
    for name in ("scheduler", "audio_scheduler"):
        scheduler = getattr(pipe, name)
        scheduler.set_timesteps(points, device="cpu")
        if len(scheduler.timesteps) != args.num_inference_steps or len(scheduler.sigmas) != points:
            raise RuntimeError("Diffusers H3 scheduler step-count contract changed; expected 5 sigmas / 4 forwards.")


def build_video_parameters(args, *, include_keyframes=True):
    """Translate public actual steps and seed into the model request once."""
    import torch

    parameters = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": schedule_points(
            args.num_inference_steps, getattr(args, "adaln_cache_file", None)
        ),
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if include_keyframes:
        load_keyframes(args, parameters)
    return parameters


def fusion_context(transformer):
    """Use the model's request context when it exposes fusion lifecycle hooks."""
    context = getattr(transformer, "inference_context", None)
    return context() if context is not None else nullcontext()


def condition_overlap_context(pipe, args):
    """Reuse one overlap controller for enabled Ref2VA requests."""
    from .validation import h3_overlap_key

    if not h3_overlap_key(args)[0]:
        return nullcontext()
    owner = getattr(pipe, '_h3_condition_overlap', None)
    if owner is None:
        from x_diffusers.framework.pipeline.minimax_h3 import ConditionOverlap

        owner = ConditionOverlap(
            pipe, sp=getattr(args, 'sp', 1), timeout=h3_overlap_key(args)[1]
        )
        pipe._h3_condition_overlap = owner
    return owner.request()


def run_video_request(pipe, args, parameters, *, output):
    """Execute one request with the same cache/overlap/fusion lifetime on every entry."""
    transformer = getattr(pipe, "transformer", None)
    from x_diffusers.adaptor.h3_runtime import begin_h3_attention_request as begin_request

    begin_request(pipe, args)
    condition_context = nullcontext()
    if getattr(args, "h3_condition_overlap", False):
        condition_context = condition_overlap_context(pipe, args)
    rope_context = nullcontext()
    if getattr(transformer, "_h3_rope_table_enabled", False) is True:
        from x_diffusers.framework.transformer.minimax_h3 import request_rope_table

        rope_context = request_rope_table(transformer)
    with (
        base_temporal_padding(pipe, parameters["num_frames"]),
        fusion_context(transformer),
        condition_context,
        rope_context,
    ):
        if getattr(args, "h3_cuda_precompute_conditioning", False):
            from x_diffusers.framework.transformer.minimax_h3 import request_conditioning

            with request_conditioning(transformer):
                return pipe(output=output, **parameters)
        return pipe(output=output, **parameters)


def base_temporal_padding(pipe, frames):
    """Select the supported long-video profile at the application boundary."""
    from .validation import h3_refiner_profile, H3_BASE_MAX_DURATION_S
    from contextlib import nullcontext
    spec = h3_refiner_profile(H3_BASE_MAX_DURATION_S)
    if frames not in (spec['output_frames'], spec['generation_frames']):
        return nullcontext()
    from x_diffusers.framework.pipeline.minimax_h3 import base_temporal_padding as padding
    return padding(pipe, spec)


def refiner_worker_scope(manager, args):
    """Cover caller-side work as well as model loading and request execution."""
    if not getattr(args, "h3_refiner", False):
        return nullcontext()
    from .load_pipe import prepare_h3_refiner

    prepare_h3_refiner(manager, args)
    return manager._h3_refiner_failure_guard


def infer_h3_refiner(manager, args):
    """Translate an application request and own output delivery for CLI and HTTP."""
    from x_diffusers.adaptor.load_pipe import prepare_h3_refiner
    from x_diffusers.adaptor.validation import validate_h3_refiner_inputs, validate_h3_refiner_settings
    from x_diffusers.framework.h3_config import H3RefinerRequest
    from x_base import infer_info
    from x_base.utils.file import atomic_json
    import torch.distributed as dist

    prepare_h3_refiner(manager, args)
    guard = manager._h3_refiner_failure_guard
    with guard:
        writer = None
        old_lightning = infer_info.vae_lightning
        try:
            validate_h3_refiner_settings(args)
            validate_h3_refiner_inputs(args)
            media, reference_audio = prepare_h3_draft_media(manager.pipe, args)
            request = H3RefinerRequest(
                prompt=args.prompt, seed=args.seed, seconds=args.h3_refiner_seconds,
                enable_audio=getattr(args, "enable_audio", True), phaa_num=args.phaa_num,
                media=media, reference_audio=reference_audio,
            )
            rank = dist.get_rank()
            output_path = Path(args.save_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = RefinerVideoWriter(output_path, manager.sr_pipe.metrics) if rank == 0 else None
            infer_info.update_info(args)
            infer_info.vae_lightning = "none"
            begin_h3_attention_request(manager.pipe, args)
            with fusion_context(manager.pipe.transformer):
                result = manager.sr_pipe(request, output=writer)
            atomic_json(output_path.with_suffix(f".metrics.rank{rank}.json"), result.metrics)
            # Successful guard exit waits for every rank, including writer cleanup.
            # Never enter a device barrier after rank-local file operations.
            return dict(videos=None, audio=None, sampling_rate=result.sampling_rate,
                        saved_path=str(output_path), h3_refiner_metrics=result.metrics)
        except BaseException as error:
            manager.sr_pipe.failed = True
            guard.fail(error)
            raise
        finally:
            infer_info.vae_lightning = old_lightning
            if writer is not None:
                try:
                    writer.close()
                except BaseException as error:
                    manager.sr_pipe.failed = True
                    guard.fail(error)
                    raise


class RefinerVideoWriter:
    """Application delivery of normalized Refiner chunks through SaveAVStream."""

    def __init__(self, output, metrics):
        import tempfile

        self.output, self.metrics = Path(output), metrics
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.work = tempfile.TemporaryDirectory(prefix=".h3-refiner-", dir=self.output.parent)
        self.staged = Path(self.work.name) / "output.mp4"
        self.stream = None
        self.closed = False
        self.frames = self.sent = 0

    def start(self, audio, sampling_rate, frames):
        from x_base.vae_parallelism.audio_utils import save_audio_to_wav
        from x_base.vae_parallelism.save_av_stream import SaveAVStream

        self.frames = frames
        wav = str(Path(self.work.name) / "audio.wav") if audio is not None else None
        if wav:
            save_audio_to_wav(audio[..., :frames * sampling_rate // 24], sampling_rate, audio_path=wav)
        self.stream = SaveAVStream(str(self.staged), wav, 24, 2560, 1440, strict=True)

    def enqueue(self, chunk):
        from x_diffusers.framework.vae.minimax_h3 import refiner_rgb_frames

        chunk = chunk[:, :, :max(0, self.frames - self.sent)]
        if chunk.shape[2] == 0:
            return
        with self.metrics.span("video.rgb8", sync=False):
            pixels = refiner_rgb_frames(chunk)
        with self.metrics.host_span("video.d2h_and_enqueue"):
            for frame in pixels.cpu().numpy():
                self.stream.put_frame(frame)
                self.sent += 1

    def finish(self):
        if self.sent != self.frames:
            raise ValueError(f"Expected {self.frames} exported frames, got {self.sent}")
        with self.metrics.host_span("video.encode_and_audio_mux_wait"):
            self.stream.close()
            self.closed = True
        if not self.staged.is_file() or self.staged.stat().st_size == 0:
            raise RuntimeError("Encoder produced no video")
        self.staged.replace(self.output)

    def close(self):
        try:
            if self.stream is not None and not self.closed:
                self.stream.abort()
        finally:
            self.work.cleanup()


def prepare_h3_draft_media(pipe, args):
    """Prepare loaded model inputs before entering the Refiner compute workflow."""
    parameters = {}
    load_keyframes(args, parameters)
    conditioning = prepare_h3_media(pipe, args, parameters)
    return parameters, conditioning in ("audio", "both")


def begin_h3_attention_request(pipe, args):
    from x_diffusers.adaptor.validation import h3_attention_config
    from x_diffusers.adaptor.validation import validate_h3_attention
    if getattr(args, 'h3_attention', 'legacy') == 'legacy':
        return
    from x_diffusers.framework.h3_config import H3AttentionRequest

    validate_h3_attention(args)
    pipe.transformer._h3_attention_session.begin(H3AttentionRequest(
        config=h3_attention_config(args), task_type=args.task_type,
        num_inference_steps=args.num_inference_steps, phaa_num=args.phaa_num,
        save_path=args.save_path,
    ))


def select_h3_cuda_adaln_table(pipe, args, conditioning):
    """Switch only the table; retain the same stripped model and CUDA operators."""
    from x_diffusers.adaptor.validation import validate_h3_cuda_adaln
    from x_base.adaln_cache import install_adaln_cache
    from safetensors import safe_open

    validate_h3_cuda_adaln(args)
    model = pipe.transformer
    if str(Path(args.transformer_path).resolve()) != model._h3_cuda_adaln_checkpoint:
        raise ValueError('Restart the manager to change the stripped AdaLN checkpoint')
    with safe_open(args.adaln_cache_file, framework='pt') as source:
        metadata = source.metadata()
    if (float(pipe.scheduler.config.shift) != float(metadata['video_shift'])
            or float(pipe.audio_scheduler.config.shift) != float(metadata['audio_shift'])):
        raise ValueError('AdaLN table scheduler shifts do not match the loaded pipeline')
    cache = model._adaln_cache
    if (cache._source_file != str(Path(args.adaln_cache_file).resolve())
            or cache._conditioning_type != conditioning):
        install_adaln_cache(pipe, args.adaln_cache_file, conditioning_type=conditioning)


def load_h3_references(paths):
    from x_diffusers.adaptor.validation import h3_reference_media_kind
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference,
    )

    loaders = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference,
               "audio": MiniMaxH3AudioReference}
    return [loaders[h3_reference_media_kind(path)].from_file(path) for path in paths]


def prepare_h3_media(pipe, args, parameters):
    """Load ordered references and select the same AdaLN table on every entry."""
    from .validation import h3_reference_conditioning, h3_reference_image_resize_mode

    paths = getattr(args, "reference_paths", None)
    if paths:
        parameters["references"] = load_h3_references(paths)
        mode = h3_reference_image_resize_mode(args)
        if mode is not None:
            parameters["h3_reference_image_resize_mode"] = mode
        conditioning = h3_reference_conditioning(args)
    else:
        conditioning = "video" if "image" in parameters or "last_image" in parameters else "none"
    select_h3_adaln(pipe, args, conditioning)
    return conditioning


def select_h3_adaln(pipe, args, conditioning):
    transformer = getattr(pipe, 'transformer', None)
    if getattr(transformer, '_h3_cuda_adaln_table', False):
        select_h3_cuda_adaln_table(pipe, args, conditioning)
        return
    cache = getattr(transformer, "_adaln_cache", None)
    if cache is None:
        return
    cache_file = getattr(args, "adaln_cache_file", None)
    if not cache_file:
        raise ValueError("The loaded AdaLN cache requires adaln_cache_file on the request")
    if (cache._conditioning_type != conditioning
            or getattr(cache, "_source_file", None) != str(Path(cache_file).resolve())):
        from x_base.adaln_cache import install_adaln_cache

        install_adaln_cache(pipe, args.adaln_cache_file, conditioning_type=conditioning)


def generate_h3_images(
    pipe, *, prompt, width, height, min_generation_area, generator, references=()
):
    """Run an in-memory image request with condition caches scoped to this call."""
    import torch
    with torch.inference_mode():
        from x_diffusers.framework.transformer.minimax_h3_image import clear_image_condition_cache

        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            min_generation_area=min_generation_area,
            output="images",
            generator=generator,
        )
        if references:
            kwargs["image"] = references
        clear_image_condition_cache(pipe.transformer)
        try:
            return pipe(**kwargs)
        finally:
            clear_image_condition_cache(pipe.transformer)
