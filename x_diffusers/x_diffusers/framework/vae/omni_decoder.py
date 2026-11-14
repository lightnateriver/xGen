# SPDX-License-Identifier: Apache-2.0
"""Opt-in native Omni H3 decoder, behind the existing H3 decode block."""
from dataclasses import asdict
import logging
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def validate_options(args, backend):
    from ...adaptor.h3_vae_config import resolve_vae_backend
    selected = resolve_vae_backend(args)
    from ...adaptor.h3_vae_compile import validate_options as validate_torch_compile
    validate_torch_compile(args, backend)
    if selected != "native-stream":
        if getattr(args, "h3_vae_model_path", None) or getattr(args, "h3_vae_trust_remote_code", False):
            raise ValueError("Native H3 VAE options require --h3_vae_backend native-stream")
        return
    if selected != "native-stream" or backend not in ("cuda", "npu") or args.model != "MiniMax-H3":
        raise ValueError("Native-stream VAE decode requires CUDA or Ascend MiniMax-H3")
    if not getattr(args, "h3_vae_model_path", None) or not getattr(args, "h3_vae_trust_remote_code", False):
        raise ValueError("Native-stream VAE requires --h3_vae_model_path and --h3_vae_trust_remote_code")
    if getattr(args, "vae_matmul_quant", "none") != "none":
        raise ValueError("Native-stream uses the native VAE weights; set --vae_matmul_quant none")
    if getattr(args, "h3_refiner", False):
        raise ValueError("H3 Refiner is not yet supported by native-stream decode")
    if dist.is_initialized() and dist.get_world_size() != getattr(args, "sp", 1):
        raise ValueError("Native-stream H3 decode currently requires one request across the complete SP world; DP groups are not supported")


def configure(pipe, args, device):
    """Do nothing for the default path; load one CPU master for an Omni worker."""
    from ...adaptor.h3_vae_config import resolve_vae_backend
    selected = resolve_vae_backend(args)
    if selected == "legacy":
        return
    pipe._h3_vae_backend_name = selected
    if selected == "lightning":
        return
    installed = getattr(pipe, "_h3_vae_decode_backend", None)
    if installed is not None:
        if str(installed.path) != str(args.h3_vae_model_path):
            raise ValueError("Restart the pipeline to change the native H3 VAE checkpoint")
        if bool(getattr(installed, "torch_compile_enabled", False)) != bool(getattr(args, "h3_vae_torch_compile", False)):
            raise ValueError("Restart the pipeline to change H3 VAE Torch Compiler")
        return
    pipe._h3_vae_decode_backend = OmniH3Decoder(args.h3_vae_model_path, device,
        trust_remote_code=args.h3_vae_trust_remote_code,
        fp16_linears=True,
        torch_compile=bool(getattr(args, "h3_vae_torch_compile", False)))
    pipe._h3_vae_torch_compile = bool(getattr(args, "h3_vae_torch_compile", False))
    logger.info("Experimental H3 VAE backend=native-stream; native tile parallelism, asynchronous RGB8 transfer and MP4 encoding")


class OmniH3Decoder:
    def __init__(self, path, device, *, trust_remote_code=False, fp16_linears=False,
                 torch_compile=False):
        from diffusers.hooks.group_offloading import ModuleGroup
        from .omni_h3 import MiniMaxH3VideoVAE

        self.device = torch.device(device)
        self.device_module = torch.get_device_module(self.device.type)
        self.path = path
        self.vae = MiniMaxH3VideoVAE(path, device=self.device, load_device=torch.device("cpu"),
                                   trust_remote_code=trust_remote_code)
        if fp16_linears:
            from .h3_cuda_chunks import prepare_decoder_linears
            count = prepare_decoder_linears(self.vae.model.decoder)
            logger.info("H3 native decoder: %d Linear modules stored as FP16 autocast operands", count)
        self.torch_compile_enabled = bool(torch_compile)
        if self.torch_compile_enabled:
            from ...adaptor.h3_vae_compile import compile_decoder_tile_subgraph
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
        from .omni_h3.chunked_video import ChunkedVideoMP4Session, decode_to_mp4

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
