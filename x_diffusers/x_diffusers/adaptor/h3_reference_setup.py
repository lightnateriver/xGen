"""Ref2VA media setup with request-local reference image geometry."""
from collections import Counter

import torch
from PIL import Image
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames, resolve_canvas_size
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

from .h3_references import REFERENCE_RESIZE_MODES, reference_image_size


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
        for kind, limit in (("image", self.max_images), ("video", self.max_videos), ("audio", self.max_audios)):
            if counts[kind] > limit:
                raise ValueError(f"Ref2VA accepts at most {limit} {kind} references")
        if len(refs) > self.max_references or not (counts["image"] or counts["video"]):
            raise ValueError(f"Ref2VA requires visual media and accepts at most {self.max_references} references")

        multiple = components.canvas_multiple
        height, width = block.height, block.width
        if (height is None) != (width is None):
            raise ValueError("height and width must be supplied together")
        if height is None:
            height, width = resolve_canvas_size(
                16, 9, multiple, components.config.canvas_short_edge, components.config.canvas_max_pixels
            )
        if min(height, width) <= 0 or height % multiple or width % multiple:
            raise ValueError(f"Output dimensions must be positive multiples of {multiple}")

        frames = align_num_frames(
            block.num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk
        )
        duration = frames / components.fps
        if not components.min_duration <= duration <= components.max_duration:
            raise ValueError(f"Aligned Ref2VA duration {duration:g}s is outside the model's supported range")

        block.height, block.width, block.num_frames = height, width, frames
        block.normalized_references = [
            self._normalize_reference(ref, components, width, height, frames, duration, mode)
            for ref in refs
        ]
        self.set_block_state(state, block)
        return components, state

    def _normalize_reference(self, ref, components, width, height, frames, duration, mode):
        if ref.kind == "image":
            image = ref.image
            if not isinstance(image, Image.Image):
                raise ValueError("Reference image resize modes require a decoded PIL image")
            short, long = sorted(image.size)
            if short <= 0 or long > 4 * short:
                raise ValueError(f"Invalid reference image size {image.size}")
            tw, th = reference_image_size(*image.size, width, height, components.canvas_multiple, mode)
            if image.size != (tw, th):
                image = components.image_processor.resize(image, height=th, width=tw)
            return MiniMaxH3ImageReference(image=image)

        rate = components.audio_sampling_rate
        audio = None
        if ref.has_audio:
            audio = self._normalize_audio_condition(
                ref.audio, ref.sample_rate if ref.sample_rate is not None else rate, rate, max_duration=duration
            )
        if ref.kind == "video":
            pixels = self._normalize_video_condition(
                ref.frames, float(ref.fps), frames, components.canvas_multiple,
                components.config.canvas_short_edge, components.config.canvas_max_pixels, float(components.fps)
            )
            return MiniMaxH3VideoReference(
                frames=pixels, fps=float(components.fps), audio=audio, sample_rate=rate if audio is not None else None
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
    raise RuntimeError("The Diffusers Ref2VA workflow does not expose its reference setup step")
