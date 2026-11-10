# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Modified for xGen: runtime/import adaptation; see UPSTREAM.json.
"""H3 native decode adapter without the vLLM service/runtime dependency."""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
import torch.distributed as dist
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .chunked_decode import decode_h3_chunks
from .ops import install_h3_vae_optimizations
from .vae_temporal import install_temporal_stream_patches

logger = logging.getLogger(__name__)
DecodedChunkConsumer = Callable[[torch.Tensor], None]

class MiniMaxH3VideoVAE(torch.nn.Module):
    decode_dtype = torch.float16
    chunk_value_range = (0.0, 1.0)
    _pixel_transform = None

    def __init__(self, component_path, *, device, load_device=None, trust_remote_code=False):
        super().__init__()
        component_path = Path(component_path)
        self.config_dict = json.loads((component_path / "config.json").read_text(encoding="utf-8"))
        reference = self.config_dict.get("auto_map", {}).get("AutoModel")
        if not trust_remote_code:
            raise ValueError("Omni VAE requires explicit trust for the selected checkpoint's Python code")
        if reference != "minimax_h3_video_vae.MiniMaxH3VideoVAE":
            raise ValueError("Expected a native MiniMax-H3 video_vae checkpoint directory")
        self._device_target = torch.device(device)
        if self._device_target.type not in ("cuda", "npu"):
            raise ValueError("Native H3 decode requires CUDA or Ascend NPU")
        model_class = get_class_from_dynamic_module(reference, str(component_path))
        with torch.device("cpu"):
            self.remote = model_class.from_pretrained(str(component_path))
        for name in ("encoder", "quant_conv"):
            if hasattr(self.remote.model, name):
                delattr(self.remote.model, name)
        self.remote.eval().to(device=load_device or self._device_target, dtype=torch.float32)
        self.optimizations_installed = install_h3_vae_optimizations(
            self.remote.model.decoder, device=self._device_target)
        install_temporal_stream_patches(self.remote.model)
        self.model = self.remote.model
        self.encode_only = False
        self.parallel_size = 1

    def load_to_device(self):
        self.remote.to(self._device_target)

    def offload_to_cpu(self):
        self.remote.to("cpu")

    # Upstream numerical methods appended below.

    def set_parallel_size(
        self,
        parallel_size: int,
        mode: str = "tile",
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        if mode != "tile":
            raise ValueError(f"MiniMax H3 VAE supports its native tile parallel mode only, got {mode!r}")
        group = process_group if process_group is not None else dist.group.WORLD
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        parallel_size = int(parallel_size)
        if parallel_size not in (1, world_size):
            raise ValueError(
                "MiniMax H3 native VAE patch parallelism currently requires "
                "vae_patch_parallel_size=1 or the full DiT group size "
                f"({world_size}), got {parallel_size}"
            )
        self.parallel_size = parallel_size
        enabled = parallel_size > 1

        state = self._native_parallel_state()
        state.clear()
        state.update(
            group_size=parallel_size,
            group_rank=rank if enabled else 0,
            local_process_group=group if enabled else None,
            sp_size=parallel_size,
            sp_rank=rank if enabled else 0,
            sp_enabled=enabled,
            sp_process_group=group if enabled else None,
            tp_size=1,
            tp_rank=0,
        )
        self.model.parallel_tiling = enabled

    def _native_parallel_state(self) -> dict[str, Any]:
        """Return the checkpoint's own mutable parallel-state dict."""

        package = self.remote.__class__.__module__.rsplit(".", 1)[0]
        parallel_module = importlib.import_module(f"{package}.parallel")
        return parallel_module.get_parallel_state()

    def _decoder_tile_count(self, latent: torch.Tensor) -> int:
        """Number of decoder tiles the checkpoint will split ``latent`` into.

        Mirrors the checkpoint's ``decode_tiled``: the grid is computed from the
        pixel-space dimensions, so it is a pure function of the latent shape and
        resolves identically on every rank.
        """

        ratio = int(self.model.vae_ratio)
        rows, _, _ = self.model.split_tiles(int(latent.shape[-2]) * ratio, True)
        cols, _, _ = self.model.split_tiles(int(latent.shape[-1]) * ratio, True)
        return len(rows) * len(cols)

    @contextmanager
    def _rank_local_tiling(self) -> Iterator[None]:
        state = self._native_parallel_state()
        saved_state = dict(state)
        saved_tiling = self.model.parallel_tiling
        state.update(
            group_size=1,
            group_rank=0,
            local_process_group=None,
            sp_size=1,
            sp_rank=0,
            sp_enabled=False,
            sp_process_group=None,
            tp_size=1,
            tp_rank=0,
        )
        self.model.parallel_tiling = False
        try:
            yield
        finally:
            state.clear()
            state.update(saved_state)
            self.model.parallel_tiling = saved_tiling

    def is_distributed_enabled(self) -> bool:
        return self.parallel_size > 1 and dist.is_initialized()

    def _decode_tiling_context(self, latent: torch.Tensor) -> AbstractContextManager:
        """Pick the tiling mode a decode of ``latent`` can safely use.

        The checkpoint hands rank r the tiles ``range(r, num_tiles, sp_size)``
        and then rejects an empty share inside the gather. A rank with no
        tiles raises and leaves the collective while the others block in it
        forever, so too few tiles hangs the whole stage rather than failing
        it. Tile count depends only on the latent shape, so every rank takes
        this branch together.
        """
        num_tiles = self._decoder_tile_count(latent)
        if self.parallel_size > 1 and num_tiles < self.parallel_size:
            logger.warning(
                "MiniMax-H3 VAE decode splits into %d tile(s) but the tile group has "
                "%d ranks; decoding rank-locally for this shape instead, which is "
                "slower but avoids ranks without tiles hanging the collective.",
                num_tiles,
                self.parallel_size,
            )
            return self._rank_local_tiling()
        return nullcontext()

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self, "encode_only", False):
            raise RuntimeError("MiniMax H3 encode-only video VAE cannot decode latents")
        with self._decode_tiling_context(latent):
            decoded = self.model.decode_base(self._denormalize_latent(latent))
        if decoded.dtype == torch.uint8:
            # The streaming uint8 write-back already ran the revert and the
            # output quantizer inside write_part; nothing remains but the
            # shape contract.
            return self._normalize_decoded_frames(decoded)
        return self._normalize_decoded_frames(self._revert_decoded_inplace(decoded))

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(self.config_dict["latents_mean"], device=latent.device, dtype=latent.dtype)
        std = torch.tensor(self.config_dict["latents_std"], device=latent.device, dtype=latent.dtype)
        shape = (1, channels, 1, 1, 1)
        return latent * std.view(shape) + mean.view(shape)

    @staticmethod
    def _normalize_decoded_frames(decoded: torch.Tensor) -> torch.Tensor:
        """Canonicalize remote H3 decoder output to [B,C,T,H,W]."""
        frames = decoded
        if frames.ndim == 4:
            frames = frames.unsqueeze(0).transpose(1, 2)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        return frames if frames.dtype == torch.uint8 else frames.float()

    @property
    def owns_chunk_output(self) -> bool:
        """Whether this rank publishes frames to the streaming MP4 consumer."""
        if not self.is_distributed_enabled():
            return True
        group = self._native_parallel_state().get("sp_process_group")
        if group is None:
            raise RuntimeError("MiniMax-H3 VAE chunk decode has no spatial-parallel group")
        return dist.get_rank(group) == 0

    @torch.inference_mode()
    def decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Decode temporal clips and synchronously publish frames-only chunks.

        Implements :class:`SupportsChunkedVAEDecode`. Every rank participating
        in distributed VAE execution must invoke this method with a callback so
        the temporal collectives stay in lockstep; ``on_chunk`` is called only
        on the rank that owns output. Chunks arrive as ``[B, C, T, H, W]``
        float frames, normalized through the checkpoint's processor to match
        the complete decode path. After a callback failure, the remaining
        chunks are decoded and discarded before the exception is re-raised.
        """
        if getattr(self, "encode_only", False):
            raise RuntimeError("MiniMax H3 encode-only video VAE cannot decode latents")
        if not callable(on_chunk):
            raise TypeError("on_chunk must be callable")
        if not callable(getattr(self.model, "_adaptive_decode", None)):
            raise RuntimeError("Loaded MiniMax-H3 VAE does not expose temporal decode primitives")
        group = None
        if self.is_distributed_enabled():
            group = self._native_parallel_state().get("sp_process_group")
            if group is None or dist.get_world_size(group) != self.parallel_size:
                raise RuntimeError("MiniMax-H3 VAE chunk decode has an invalid spatial-parallel group")
        # Native H3 tiling performs its own collectives for every temporal clip,
        # so this path needs the same too-few-tiles fallback as the complete
        # decode: without it a shape that leaves some ranks tileless hangs the
        # gather instead of decoding rank-locally.
        with self._decode_tiling_context(z):
            decode_h3_chunks(self, z, on_chunk, group=group)

    def _revert_decoded(self, decoded: torch.Tensor) -> torch.Tensor:
        """Preserve chunk storage that the temporal decoder may still use."""
        if self._pixel_transform is not None:
            return self._pixel_transform(decoded)
        processor = getattr(self.model, "processor", None)
        return decoded if processor is None else processor.revert_tensor(decoded)

    def _revert_decoded_inplace(self, decoded: torch.Tensor) -> torch.Tensor:
        """In-place counterpart of the processor's ``revert_tensor``.

        The checkpoint version materializes a denormalized copy, a clamped
        copy, and a contiguous copy of the whole decoded video -- three
        pixel-scale tensors resident at the decode peak. Decoding owns the
        tensor here, so the same op order (torchvision ``Normalize`` is
        ``(x - mean) / std``, then ``clamp(0, 1)``) runs in place on the
        original ``(B, C, T, H, W)`` layout, which is bit-identical
        elementwise to normalizing the ``(b t) c h w`` rearrangement and
        returns with zero whole-video copies.

        Falls back to ``processor.revert_tensor`` when the checkpoint's
        denormalization constants are not discoverable.
        """
        if self._pixel_transform is not None:
            return self._pixel_transform(decoded)
        processor = getattr(self.model, "processor", None)
        transform_rev = getattr(processor, "transform_rev", None)
        mean = getattr(transform_rev, "mean", None)
        std = getattr(transform_rev, "std", None)
        if mean is None or std is None or len(mean) != 3 or len(std) != 3:
            return processor.revert_tensor(decoded)
        if bool(getattr(processor, "use_3d_conv", True)) and decoded.ndim == 4:
            decoded = decoded.unsqueeze(2)
        mean_t = torch.as_tensor(mean, dtype=decoded.dtype, device=decoded.device).view(1, 3, 1, 1, 1)
        std_t = torch.as_tensor(std, dtype=decoded.dtype, device=decoded.device).view(1, 3, 1, 1, 1)
        return decoded.sub_(mean_t).div_(std_t).clamp_(0.0, 1.0)
