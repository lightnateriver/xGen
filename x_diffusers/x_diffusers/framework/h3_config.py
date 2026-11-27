"""Model settings and H3 geometry; independent of application policy and I/O."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


# LightX2V 768P eight-step weights shared by inference and asset preparation.
H3_TURBO8_LORA_SHA256 = "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4"


@dataclass(frozen=True)
class H3RefinerConfig:
    weights: str
    prepared: str
    backend: str
    loading: str
    base_steps: int
    turbo8: bool = False
    base_cache_file: str | None = None


@dataclass(frozen=True)
class H3RefinerRequest:
    prompt: str
    seed: int
    seconds: int
    enable_audio: bool = True
    phaa_num: int = 2
    media: Mapping[str, Any] = field(default_factory=dict)
    reference_audio: bool = False


@dataclass(frozen=True)
class H3RefinerResult:
    metrics: dict
    videos: Any = None
    audio: Any = None
    sampling_rate: int = 32000


class H3RefinerOutput(Protocol):
    """Consumer of decoded model outputs; implementations own delivery and cleanup."""

    def start(self, audio, sampling_rate: int, frames: int) -> None: ...
    def enqueue(self, normalized_frames) -> None: ...
    def finish(self) -> None: ...


def schedule_points(steps):
    """Include the terminal zero sigma expected by MiniMaxH3Scheduler."""
    if type(steps) is not int or steps < 1:
        raise ValueError("H3 num_inference_steps must be a positive integer")
    return steps + 1


def refiner_shape(seconds):
    if type(seconds) is not int or seconds < 1:
        raise ValueError("H3 duration must be a positive integer")
    output = seconds * 24
    generation = ((output - 5 + 16) // 17) * 17 + 5
    return dict(
        seconds=seconds,
        output_frames=output,
        generation_frames=generation,
        latent_frames=(generation - 5) // 17 * 5 + 2,
        audio_frames=round(generation / 24 * 40),
    )


@dataclass(frozen=True)
class H3AttentionConfig:
    backend: str
    predictor: str | None = None
    turbo8: bool = False
    sparsity: float = 0.9

    @property
    def key(self):
        return self.backend, self.predictor, self.turbo8, self.sparsity


@dataclass(frozen=True)
class H3AttentionRequest:
    config: H3AttentionConfig
    task_type: str
    num_inference_steps: int
    phaa_num: int
    save_path: str


@dataclass(frozen=True)
class H3ParallelSettings:
    mode: str = "sp"
    tp: int = 1
    cp: int = 1
    tp_comm: str = "bf16"
    cp_comm: str = "auto"
    tp_chunk_elements: int = 262144
    warmup_mib: int = 0


@dataclass(frozen=True)
class H3ExecutionOptions:
    h3_parallel_mode: str = "sp"
    h3_cuda_precision: str = "none"
    phaa_num: int = 0
    sp: int = 1
    h3_cuda_vae_sp: bool = False
    h3_vae_backend: str = "legacy"
    h3_refiner: bool = False
    h3_cuda_input_refiner: str = "none"
    h3_cuda_input_refiner_sp: bool = False
    h3_cuda_input_refiner_attention: str = "legacy"
    h3_cuda_input_refiner_ffn: str = "bf16"
    h3_cuda_refiner_ff_chunk_size: int = 0
    h3_cuda_precompute_conditioning: bool = False
    h3_cuda_local_input: bool | None = None
    h3_cuda_local_output: bool | None = None
    h3_cuda_conditioning_offload: bool = False
    h3_cuda_reference_offload: bool = False
    h3_cuda_vision_sp: bool = False
    h3_cuda_vae_encoder_parallel: bool = False
    h3_cuda_audio_encoder_parallel: bool = False
    h3_cuda_audio_decoder_parallel: str = "none"
    task_type: str = "t2va"
    adaln_cache_file: str | None = None
    transformer_path: str | None = None
    parallel: H3ParallelSettings = H3ParallelSettings()
    dit_fusions: frozenset[str] = frozenset()
    local_io_default: bool = True
    h3_cuda_dit_fusions: bool = False


REFERENCE_RESIZE_MODES = ("diffusers", "max", "match")


def reference_image_size(width, height, canvas_width, canvas_height, multiple, mode):
    import math

    if mode == "match":
        scale = min(1.0, math.sqrt(canvas_width * canvas_height / (width * height)))
    else:
        scale = 2048 / min(width, height)
        if mode == "max":
            scale = min(1.0, scale)
    return tuple(
        max(multiple, round(side * scale / multiple) * multiple)
        for side in (width, height)
    )


def condition_parallel_key(args):
    return tuple(
        getattr(args, name, "none" if name.endswith("decoder_parallel") else False)
        for name in (
            "h3_cuda_vision_sp",
            "h3_cuda_vae_encoder_parallel",
            "h3_cuda_audio_encoder_parallel",
            "h3_cuda_audio_decoder_parallel",
        )
    )
