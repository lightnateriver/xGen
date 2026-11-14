"""Optional Torch Compiler integration for the CUDA native H3 VAE decoder."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def validate_options(args, backend: str) -> None:
    """Validate the opt-in compiler before any model/device work starts."""

    if not bool(getattr(args, "h3_vae_torch_compile", False)):
        return
    if backend != "cuda":
        raise ValueError("--h3_vae_torch_compile is CUDA-only; keep it disabled on NPU")
    if getattr(args, "model", None) != "MiniMax-H3":
        raise ValueError("--h3_vae_torch_compile requires MiniMax-H3")
    if getattr(args, "h3_vae_backend", None) != "native-stream":
        raise ValueError("--h3_vae_torch_compile requires --h3_vae_backend native-stream")
    if not hasattr(torch, "compile"):
        raise RuntimeError("Torch Compiler requires a PyTorch build exposing torch.compile")


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
    logger.info("H3 native-stream VAE decoder tile subgraph compiled with Torch Compiler")
    return decoder
