import os
import io
import json
import hashlib
import logging
import threading
import queue
import time
import uuid
import gc
import signal
import atexit
import subprocess
import base64
import re
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

import sys
from file_util import file_util
from maas_fl2va import InvalidInput, parameter_spec as maas_parameter_spec, read_submission, validate_startup as validate_maas_startup
from maas_health import MaaSHealth
from maas_artifacts import StorageError, publish_obs, publish_video, validate_media_tools

from x_diffusers import init_env, parse_args
from x_base.device import device_type
from x_diffusers.adaptor.h3_refiner_config import H3_BASE_MAX_DURATION_S, H3_MAX_PROMPT_LENGTH, base_temporal_padding, h3_prompt_length
from x_diffusers.adaptor.h3_schedule import cache_steps, schedule_points
from x_diffusers.adaptor.h3_references import LIMITS, MAX_FILES, reference_conditioning
from x_diffusers.adaptor.load_pipe import load_pipe, load_v2v_pipe, update_pipe, pipe_to_device, load_seedvr2_pipe, update_lora
from x_diffusers.adaptor.infer_tools import InferenceManager, GUIDANCE_SCALE_MAP
from x_diffusers.framework.vae.wan import vfi
from x_diffusers.framework.vae.IFRNet_S_arch import IRFNet_S
from x_base import infer_info
from diffusers.utils import export_to_video, load_image
from PIL import Image
import math



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_base64_image(image_data: str) -> str:
    """Decode base64 image data and save to temporary file.
    
    Args:
        image_data: Base64 encoded image string, can be:
                   - data:image/jpeg;base64,<data>
                   - data:image/png;base64,<data>
                   - plain base64 string
    
    Returns:
        Path to temporary file containing decoded image
    """
    if image_data.startswith('data:image/'):
        parts = image_data.split(',', 1)
        if len(parts) != 2:
            raise ValueError("Invalid data URI format")
        
        header, data = parts
        if 'image/jpeg' in header or 'image/jpg' in header:
            ext = '.jpg'
        elif 'image/png' in header:
            ext = '.png'
        elif 'image/webp' in header:
            ext = '.webp'
        else:
            ext = '.jpg'
    else:
        data = image_data
        ext = '.jpg'
    
    try:
        decoded = base64.b64decode(data)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(decoded)
            logger.info(f"Decoded base64 image to temporary file: {f.name}")
            return f.name
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

_temp_files = set()

_BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')

def _is_plain_base64(s: str) -> bool:
    """Check if a string looks like plain base64 (not a file path or URL)."""
    if not s or len(s) < 50:
        return False
    if s.startswith(('data:', 'http://', 'https://', '/')):
        return False
    return bool(_BASE64_PATTERN.match(s))

def _cleanup_temp_files():
    """Clean up temporary files created by _resolve_input_file."""
    for f in _temp_files:
        try:
            os.unlink(f)
            logger.info(f"Cleaned up temp file: {f}")
        except OSError:
            pass
    _temp_files.clear()

def _resolve_input_file(input_data: str) -> str:
    """Resolve input file from base64 data URI, HTTP URL, or local file path.

    Args:
        input_data: One of:
                   - data:image/...;base64,<data>  → decoded to temp file
                   - http://... or https://...     → downloaded to temp file
                   - /local./file.jpg        → returned as-is

    Returns:
        Local file path. Empty string if input_data is empty.
    """
    if not input_data or not isinstance(input_data, str):
        return ""

    # Base64 data URI
    if input_data.startswith('data:image/'):
        path = decode_base64_image(input_data)
        _temp_files.add(path)
        return path

    # HTTP/HTTPS URL
    if input_data.startswith(('http://', 'https://')):
        import urllib.request
        url_path = input_data.split('?')[0]
        ext = os.path.splitext(url_path)[1] or '.jpg'
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        urllib.request.urlretrieve(input_data, tmp.name)
        logger.info(f"Downloaded HTTP input to: {tmp.name}")
        _temp_files.add(tmp.name)
        return tmp.name

    # Plain base64 (no data URI prefix)
    if _is_plain_base64(input_data):
        path = decode_base64_image(input_data)
        _temp_files.add(path)
        return path

    # Local file path — return as-is
    return input_data

def get_device_for_comm():
    if not dist.is_initialized():
        return 'cpu'
    backend = dist.get_backend()
    if backend == 'gloo':
        return 'cpu'
    if backend in ['nccl', 'hccl']:
        if device_type() == "cuda":
            return f"cuda:{torch.cuda.current_device()}"
        if hasattr(torch, 'npu') and torch.npu.is_available():
            rank = dist.get_rank()
            return f'npu:{rank}'
        elif torch.cuda.is_available():
            return 'cuda:0'
    return 'cpu'


CMD_HEARTBEAT = 0
CMD_EXECUTE = 1
CMD_SHUTDOWN = 2


def broadcast_dict(data: Dict, src: int = 0) -> Dict:
    if not dist.is_initialized():
        return data
    rank = dist.get_rank()
    device = get_device_for_comm()
    if rank == src:
        data_str = json.dumps(data)
        length = torch.tensor([len(data_str)], dtype=torch.long, device=device)
    else:
        length = torch.tensor([0], dtype=torch.long, device=device)
    
    dist.broadcast(length, src=src)
    
    if rank == src:
        data_tensor = torch.tensor(list(data_str.encode('utf-8')), dtype=torch.uint8, device=device)
    else:
        data_tensor = torch.zeros(length.item(), dtype=torch.uint8, device=device)
    
    dist.broadcast(data_tensor, src=src)
    
    if rank != src:
        data_str = bytes(data_tensor.cpu().tolist()).decode('utf-8')
        data = json.loads(data_str)
    return data


def broadcast_int(value: int, src: int = 0) -> int:
    if not dist.is_initialized():
        return value
    device = get_device_for_comm()
    tensor = torch.tensor([value], dtype=torch.int, device=device)
    dist.broadcast(tensor, src=src)
    return tensor.item()


MODEL_REGISTRY = {
    "wan2.2-i2v-14b": {
        "model": "Wan2.2-I2V-A14B",
        "path": "./weights/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "task_type": "i2v",
        "default_guidance": (3.5, 3.5),
    },
    "wan2.2-t2v-14b": {
        "model": "Wan2.2-T2V-A14B",
        "path": "./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "task_type": "t2v",
        "default_guidance": (3.0, 4.0),
    },
    "minimax-h3": {
        "model": "MiniMax-H3",
        "path": "./weights/MiniMax-H3",
        "task_type": "fl2va",
        "default_guidance": (None, None),
    },
}

DEFAULT_MINIMAX_H3_CONFIG = {
    "model": "MiniMax-H3",
    "pretrained_model_name_or_path": "./weights/MiniMax-H3",
    "task_type": "fl2va",
    "dtype": "bf16",
    "sp": 8,
    "fsdp": None,
    "h3_vae_backend": "native-stream",
    "h3_vae_model_path": None,
    "h3_vae_trust_remote_code": False,
    "h3_features": "full-parallel",
    "h3_cuda_refiner_ff_chunk_size": 0,
    "vae_lightning": "none",  # Internal placeholder; H3 selection is resolved by the worker.
    "turbo_mode": "default",
    "inf_vram_blocks_num": 0,
    "atten": "laser",
    "matmul_quant": "int8",
    "vae_matmul_quant": "none",
    "rope_fused": True,
    "text_encoder_tp": 8,
    "phaa_num": 2,
    "adaln_cache_file": "model-cache/t2va-21step.safetensors",
    "transformer_path": "model-cache/base-t2va",
    "num_inference_steps": 20,
    "save_fps": 24,
    "ten_second": False,
    "adopt_sr": False,
    "seedvr2_model_dir": "./weights/SeedVR2",
    "seedvr2_model_name": "seedvr2_ema_7b_fp16.safetensors",
    "frame_interpolation": False,
    "external_base_url": None,
    "lora_path_list": None,
    "lora_transformer_list": None,
    "lora_scale_weight_list": [1.0],
    "guidance_scale": None,
    "guidance_scale_2": None,
    "keyframe_image": "",
    "keyframe_last_image": "",
    "reference_paths": None,
    "audio_save_path": "",
    "attention_sparse": False,
    "resolution": "768p",
    "enable_audio": True,
    "first_frame": "",
    "last_frame": "",
    "ref_path": None,
    "duration": 5,
}

# CUDA Base defaults are independent of the existing NPU defaults above.
if device_type() == "cuda":
    DEFAULT_MINIMAX_H3_CONFIG.update({
        "h3_cuda_precision": "auto", "phaa_num": 2, "h3_cuda_vae_sp": True,
        "atten": "none", "matmul_quant": "none", "vae_matmul_quant": "none",
        "h3_vae_backend": None, "vae_lightning": "none", "rope_fused": False,
        "h3_features": None, "h3_cuda_refiner_ff_chunk_size": None,
        "num_inference_steps": 50,
        "transformer_path": None, "adaln_cache_file": None,
    })

# MiniMax-H3 resolution presets (resolution + aspect_ratio -> width x height)
MMH3_RESOLUTION_MAP = {
    "768p": {
        "16:9": (1344, 768),    # 42*32 x 24*32
        "9:16": (768, 1344),    # 24*32 x 42*32
    },
    "1080p": {
        "16:9": (1920, 1088),   # 60*32 x 34*32
        "9:16": (1088, 1920),   # 34*32 x 60*32
    },
}

DEFAULT_TEN_SECOND_CONFIG = {
    "ten_second_model_id_t2v": "./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "ten_second_model_path": "./weights/ttv_14B_4steps/transformer_14B/model.safetensors",
    "ten_second_model_path_2": "./weights/ttv_14B_4steps/transformer_14B_2/model.safetensors",
    "pusa_lora": "./weights/pusa_lora_converted/pytorch_lora_weights_high.safetensors",
    "pusa_lora2": "./weights/pusa_lora_converted/pytorch_lora_weights_low.safetensors",
    "joint_model_path": "./weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "seedvr2_model_dir": "./weights/SeedVR2/",
    "seedvr2_model_name": "seedvr2_ema_7b_fp16.safetensors",
    "x_model_path": "./weights/6steps/transformer_i2v_14B_1/model.safetensors",
    "x_model_path_2": "./weights/6steps/transformer_i2v_14B_2/model.safetensors",
    "frame_model_path": "./weights/IFRNet_S_Vimeo90K.pth",
}

TEN_SECOND_CONFIG = DEFAULT_TEN_SECOND_CONFIG.copy()


class AsyncTask:
    STATUS_MAP = {
        "pending": "pending",
        "queued": "pending",
        "running": "running",
        "completed": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }

    def __init__(self, task_id: str, task_type: str, params: Dict[str, Any], created_at_ms: Optional[int] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.status = "pending"
        self.progress = 0.0
        self.result = None
        self.output = None
        self.error = None
        if created_at_ms:
            self.created_at = datetime.fromtimestamp(created_at_ms / 1000.0)
        else:
            self.created_at = datetime.now()
        self.created_at_ms = created_at_ms or int(self.created_at.timestamp() * 1000)
        self.started_at = None
        self.completed_at = None
        self.infer_time = None

    def get_platform_status(self) -> str:
        return self.STATUS_MAP.get(self.status, self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": f"{self.progress:.1f}%",
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "infer_time": self.infer_time,
        }

    def to_platform_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "status": self.get_platform_status(),
            "output": self.output or "",
        }


MAAS_QUEUE_CAPACITY = 10
MAAS_QUEUE_TIMEOUT_S = 600


class MaaSTask(AsyncTask):
    def __init__(self, task_id, input_key, created_at_ms=None):
        super().__init__(task_id, 'video_generation', None, created_at_ms)
        self.input_key = input_key
        self.queued_at = time.monotonic()
        self.error_code = None
        self.failure_stage = None
        self.timings_s = {}

    def finish(self, status, *, code=None, stage=None, result=None):
        if self.status not in ('pending', 'running'):
            return False
        self.error_code = self.error = code
        self.failure_stage = stage
        self.completed_at = datetime.now()
        self.timings_s['total'] = round(time.monotonic() - self.queued_at, 3)
        if result is not None:
            self.result = result
            self.output = json.dumps(result)
            self.progress = 100.0
            self.infer_time = result['infer_time']
        self.status = status
        if getattr(self, 'on_change', None):
            self.on_change(self)
        return True

    def _details(self):
        return dict(error_code=self.error_code, failure_stage=self.failure_stage, timings_s=dict(self.timings_s))

    def to_dict(self):
        return {**super().to_dict(), **self._details()}

    def to_platform_dict(self):
        return {**super().to_platform_dict(), **self._details()}


class VideoInferenceManager:
    def __init__(self):
        self.pipe = None
        self.v2v_pipe = None
        self.interpolation_model = None
        self.sr_pipe = None
        self.init_args = None
        self.current_model = None

    def initialize_pipe(self, args):
        from x_diffusers.adaptor.h3_cuda_options import needs_resolution
        if (needs_resolution(args)
                or getattr(self, "_h3_cuda_precision_key", None) in ("fp8-bf16", "fp8-sage")):
            from x_diffusers.adaptor.h3_cuda_precision import validate_manager

            validate_manager(self, args, device_type())
        if (getattr(args, 'h3_attention', 'legacy') != 'legacy' or
                getattr(getattr(self, 'init_args', None), 'h3_attention', 'legacy') != 'legacy'):
            from x_diffusers.adaptor.h3_attention_service import validate_manager
            validate_manager(self, args)
        if getattr(args, 'h3_refiner', False):
            from x_diffusers.adaptor.h3_refiner import prepare
            return prepare(self, args)
        if getattr(self, '_h3_refiner_key', None) is not None:
            raise ValueError('Restart the service to disable H3 Refiner and restore the legacy backend')
        if self.pipe is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(f"[Rank {rank}] Using cached pipe for {self.current_model}")
            # Lazy-load sr_pipe if adopt_sr=True and not yet loaded
            if getattr(args, 'adopt_sr', False) and self.sr_pipe is None and getattr(args, 'seedvr2_model_dir', None):
                try:
                    logger.info(f"[Rank {rank}] Lazy-loading sr_pipe for super resolution...")
                    self.sr_pipe = load_seedvr2_pipe(args)
                    logger.info(f"[Rank {rank}] sr_pipe loaded successfully")
                except Exception as e:
                    logger.warning(f"[Rank {rank}] sr_pipe load failed, will generate at native resolution: {e}")
                    self.sr_pipe = None
            return
        
        model_name = args.model
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[Rank {rank}] Loading pipe for {model_name}...")
        
        self.pipe = load_pipe(args)

        if args.lora_path_list:
            args.fsdp = "text_encoder"
            update_pipe(self.pipe, args)
            update_lora(
                self.pipe,
                args.model,
                None,
                args.lora_path_list,
                fuse_lora=False,
                weights_list=args.lora_scale_weight_list,
                lora_transformer_list=args.lora_transformer_list
            )
        else:
            update_pipe(self.pipe, args)
        self.pipe = pipe_to_device(self.pipe, args)
        
        logger.info(f"[Rank {rank}] Setting return_output=True for VAE")
        self.pipe.vae.return_output = True
        if hasattr(self.pipe, 'video_processor'):
            self.pipe.video_processor.return_output = True
        
        if args.ten_second:
            logger.info(f"[Rank {rank}] Loading v2v_pipe for 10s video generation...")
            v2v_args = deepcopy(args)
            self.v2v_pipe = load_v2v_pipe(v2v_args)
            v2v_args.matmul_quant = 'none'
            v2v_args.joint = False
            update_pipe(self.v2v_pipe, v2v_args)
            self.v2v_pipe = pipe_to_device(self.v2v_pipe, args)
            logger.info(f"[Rank {rank}] v2v_pipe loaded successfully")
            if dist.is_initialized():
                 dist.barrier()
                 logger.info(f"[Rank {rank}] v2v_pipe loading synchronized across all ranks")
        
        if getattr(args, 'adopt_sr', False):
            try:
                logger.info(f"[Rank {rank}] Loading sr_pipe for super resolution...")
                self.sr_pipe = load_seedvr2_pipe(args)
                logger.info(f"[Rank {rank}] sr_pipe loaded successfully")
            except Exception as e:
                logger.warning(f"[Rank {rank}] sr_pipe initial load failed, will use lazy-load fallback: {e}")
                self.sr_pipe = None
        
        self.init_args = args
        self.current_model = model_name
        logger.info(f"[Rank {rank}] Pipe for {model_name} loaded and cached")

    # ------------------------------------------------------------------
    # MiniMax-H3 inference (ModularPipeline — joint video + audio)
    # ------------------------------------------------------------------

    _MMH3_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    _MMH3_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    _MMH3_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".ogg"}

    @staticmethod
    def _mmh3_videos_to_numpy(videos):
        """Convert MiniMax-H3 pipeline video output to numpy array (T, H, W, C) float32 [0,1].

        Handles PIL Images, torch Tensors, and numpy arrays (uint8 or float).
        """

        def _flatten(lst):
            for item in lst:
                if isinstance(item, (list, tuple)):
                    yield from _flatten(item)
                else:
                    yield item

        def _frame_to_float32_hwc(frame):
            if isinstance(frame, Image.Image):
                return np.array(frame, dtype=np.float32) / 255.0
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    frame = frame.transpose(1, 2, 0)
                if frame.dtype == np.uint8:
                    return frame.astype(np.float32) / 255.0
                return frame.astype(np.float32)
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        frames = [_frame_to_float32_hwc(f) for f in _flatten(videos)]
        return np.stack(frames)

    def _mmh3_setup_super_resolution(self, args, rank):
        """Setup SeedVR2 super-resolution: generate at 768p then upscale."""
        _1080p_dims = {MMH3_RESOLUTION_MAP["1080p"]["16:9"], MMH3_RESOLUTION_MAP["1080p"]["9:16"]}
        need_sr = (
            getattr(args, "adopt_sr", False)
            and (args.width, args.height) in _1080p_dims
            and self.sr_pipe is not None
        )
        if need_sr:
            logger.info(f"[Rank {rank}] adopt_sr=True with 1080p target — "
                        f"generating at 768p first, then super-resolving")
            orig_w, orig_h = args.width, args.height
            if args.width > args.height:  # landscape
                args.width, args.height = MMH3_RESOLUTION_MAP["768p"]["16:9"]
            else:  # portrait
                args.width, args.height = MMH3_RESOLUTION_MAP["768p"]["9:16"]
            return need_sr, orig_w, orig_h
        return need_sr, None, None

    def _mmh3_build_infer_params(self, args):
        """Build inference parameters dict for MiniMax-H3."""
        infer_params = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "num_inference_steps": schedule_points(args.num_inference_steps, getattr(args, "adaln_cache_file", None)),
            "generator": torch.Generator().manual_seed(args.seed),
        }
        if getattr(args, "keyframe_image", ""):
            infer_params["image"] = load_image(args.keyframe_image)
        if getattr(args, "keyframe_last_image", ""):
            infer_params["last_image"] = load_image(args.keyframe_last_image)
        return infer_params

    def _mmh3_load_references(self, ref_paths):
        """Load reference media for MiniMax-H3 ref2va."""
        if not ref_paths:
            return None
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
            MiniMaxH3AudioReference,
        )
        references = []
        for path in ref_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in self._MMH3_IMAGE_EXTS:
                references.append(MiniMaxH3ImageReference.from_file(path))
            elif ext in self._MMH3_VIDEO_EXTS:
                references.append(MiniMaxH3VideoReference.from_file(path))
            elif ext in self._MMH3_AUDIO_EXTS:
                references.append(MiniMaxH3AudioReference.from_file(path))
            else:
                logger.warning(f"Unknown reference media type for: {path}, skipping")
        return references if references else None

    def _mmh3_setup_attention(self, args):
        """Setup attention optimization for MiniMax-H3."""
        if args.atten == 'rainfusion':
            num_latent_frames = (args.frames - 1) // 4 + 1
            latent_height = args.height // 8
            latent_width = args.width // 8
            from x_base import attention_manager
            attention_manager.set_rainfusion_attention(
                [num_latent_frames, latent_height, latent_width],
                skip_timesteps=int(args.rainfusion_ratio * schedule_points(args.num_inference_steps)),
                sparsity=getattr(args, 'rainfusion_sparsity', 0.64))
        if args.atten == 'sparse':
            from x_base import attention_manager
            attention_manager.set_ada_bsa_sparse_flash_attention(args.ada_sparsity)

    def _mmh3_patch_denoiser(self, transformer):
        """Monkey-patch denoiser to copy num_condition_video_rows onto transformer."""
        if transformer is None:
            return False, None
        from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser
        _orig_call = MiniMaxH3LoopDenoiser.__call__

        def _patched_call(self_denoiser, components, block_state, i, t):
            tr = getattr(components, self_denoiser.transformer_name)
            tr._num_condition_video_rows = getattr(
                block_state, "num_condition_video_rows", 0)
            return _orig_call(self_denoiser, components, block_state, i, t)

        MiniMaxH3LoopDenoiser.__call__ = _patched_call
        return True, _orig_call

    @staticmethod
    def _probe_video_has_audio(path: str) -> bool:
        """Quick probe: does this video file have an audio stream?

        Matches the denoiser's behavior: _decode_video_file extracts audio
        iff container.streams.audio is non-empty, regardless of enable_audio.
        """
        try:
            import av
            with av.open(path) as container:
                return len(container.streams.audio) > 0
        except Exception:
            return False

    def _mmh3_switch_adaln_cache(self, args, transformer, ref_paths, rank):
        """Switch AdaLN cache conditioning_type if needed."""
        adaln_cache_file = getattr(args, "adaln_cache_file", None)
        if not adaln_cache_file or transformer is None:
            return
        _cache = getattr(transformer, "_adaln_cache", None)
        if _cache is None:
            return
        logger.debug(f"[Rank {rank}] _mmh3_switch_adaln_cache: "
                     f"ref_paths={ref_paths}, "
                     f"args.reference_paths={getattr(args, 'reference_paths', 'MISSING')}, "
                     f"args.keyframe_image={getattr(args, 'keyframe_image', None)}, "
                     f"args.keyframe_last_image={getattr(args, 'keyframe_last_image', None)}, "
                     f"args.enable_audio={getattr(args, 'enable_audio', None)}, "
                     f"task_type={getattr(args, 'task_type', None)}")
        # conditioning_type must match the constant timesteps that the
        # denoiser injects via build_row_timesteps():
        #   - 0.999 (keyframe_noise_aug) is present iff there are video
        #     condition rows (first_frame, last_frame, or ref media).
        #   - 1.0 is present iff there are audio reference rows, which
        #     happens when ref_paths contains a video file with an audio
        #     stream (the model extracts its audio track) or an audio file.
        # enable_audio controls audio GENERATION only; it does not affect
        # audio reference extraction from the video file.
        _has_video = bool(
            getattr(args, "keyframe_image", "")
            or getattr(args, "keyframe_last_image", "")
        )
        _has_audio = False
        if ref_paths:
            for _p in ref_paths:
                _ext = os.path.splitext(_p)[1].lower()
                if _ext in self._MMH3_IMAGE_EXTS:
                    _has_video = True
                elif _ext in self._MMH3_VIDEO_EXTS:
                    _has_video = True
                    if self._probe_video_has_audio(_p):
                        _has_audio = True
                elif _ext in self._MMH3_AUDIO_EXTS:
                    _has_audio = True
        _needed_cond = (
            "both" if (_has_video and _has_audio) else
            "video" if _has_video else
            "audio" if _has_audio else
            "none"
        )
        if _cache._conditioning_type != _needed_cond:
            logger.info(f"[Rank {rank}] AdaLN cache conditioning_type "
                        f"switching: {_cache._conditioning_type} -> {_needed_cond}")
            from x_base.adaln_cache import install_adaln_cache
            install_adaln_cache(
                self.pipe,
                cache_file=adaln_cache_file,
                conditioning_type=_needed_cond,
            )

    def _mmh3_cleanup(self, transformer, denoiser_patched, orig_denoiser_call):
        """Cleanup MiniMax-H3 inference: unpatch processors and denoiser."""
        if transformer is not None:
            try:
                from x_diffusers.framework.transformer.minimax_h3 import (
                    unpatch_attention_processor,
                    unpatch_block_forward,
                )
                unpatch_attention_processor()
                unpatch_block_forward()
            except ImportError:
                logger.warning("unpatch functions not available in x_diffusers, skipping processor cleanup")
            transformer._num_condition_video_rows = 0
            for block in transformer.transformer_blocks:
                block._num_ref_rows = 0
                block._num_ref_rows_local = 0
                block._block_idx = None
                block._S_full = None
                block._S_local = None
                block.attn._layer_idx = None
                block.attn._num_ref_rows = 0
                block.attn._num_ref_rows_local = 0
                block.attn._S_full = None
                block.attn._S_local = None
                block.attn._step0_key = None
                block.attn._step0_value = None
        if denoiser_patched:
            from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser
            MiniMaxH3LoopDenoiser.__call__ = orig_denoiser_call

    def _mmh3_post_super_resolution(self, result, args, rank):
        """Run SeedVR2 super-resolution post-processing."""
        videos = result.get("videos")
        if videos is None:
            logger.warning(f"[Rank {rank}] result has no 'videos' key before SR")
            return
        np_frames = self._mmh3_videos_to_numpy(videos)
        result["videos"] = np_frames

        _orig_save_path = infer_info.save_path
        _orig_frames = infer_info.frames
        _sr_temp_path = tempfile.mktemp(suffix="_sr.mp4",
                                        dir=os.path.dirname(_orig_save_path) or ".")
        infer_info.save_path = _sr_temp_path
        infer_info.frames = len(np_frames) + 100
        logger.info(f"[Rank {rank}] SR temp save_path={_sr_temp_path}, "
                    f"frames={infer_info.frames}, fps={infer_info.fps}")

        self.sr_pipe(np_frames)

        infer_info.save_path = _orig_save_path
        infer_info.frames = _orig_frames

        if rank == 0 and os.path.exists(_sr_temp_path):
            import cv2 as _cv2
            cap = _cv2.VideoCapture(_sr_temp_path)
            sr_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                sr_frames.append(
                    _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            cap.release()
            os.remove(_sr_temp_path)
            if sr_frames:
                result["videos"] = np.stack(sr_frames)
                logger.info(f"[Rank {rank}] SR video loaded: "
                            f"{len(sr_frames)} frames, shape={result['videos'].shape}")
            else:
                logger.warning(f"[Rank {rank}] SR temp file empty — "
                              f"keeping original 768p video")
        else:
            logger.warning(f"[Rank {rank}] SR temp file not found — "
                          f"keeping original 768p video")

    def _infer_minimax_h3(self, args, task_id=None, task_manager=None):
        """Run MiniMax-H3 joint video + audio generation via ModularPipeline."""
        if getattr(args, 'h3_refiner', False):
            from x_diffusers.adaptor.h3_refiner import infer
            return infer(self, args)
        rank = dist.get_rank() if dist.is_initialized() else 0

        # --- SeedVR2 super-resolution setup ---
        need_sr, orig_w, orig_h = self._mmh3_setup_super_resolution(args, rank)

        # --- Build inference params ---
        infer_params = self._mmh3_build_infer_params(args)

        # --- Reference conditioning (ref2va) ---
        ref_paths = getattr(args, "reference_paths", None)
        references = self._mmh3_load_references(ref_paths)
        if references:
            infer_params["references"] = references

        # --- Attention optimization ---
        self._mmh3_setup_attention(args)

        # --- Patch denoiser ---
        transformer = getattr(self.pipe, "transformer", None)
        denoiser_patched, orig_denoiser_call = self._mmh3_patch_denoiser(transformer)

        # --- AdaLN cache conditioning_type switching ---
        self._mmh3_switch_adaln_cache(args, transformer, ref_paths, rank)

        logger.info(f"[Rank {rank}] MiniMax-H3 inference starting (task_type={args.task_type}, "
                    f"steps={args.num_inference_steps}, frames={args.frames}, "
                    f"size={args.width}x{args.height}, audio={args.enable_audio})")

        output_keys = ["videos"]
        if args.enable_audio:
            output_keys += ["audio", "sampling_rate"]

        try:
            if getattr(args, 'h3_attention', 'legacy') != 'legacy':
                self.pipe.transformer._h3_attention_session.begin(args)
            from x_diffusers.adaptor.h3_npu_options import request_context

            with (base_temporal_padding(self.pipe, infer_params["num_frames"]),
                  request_context(transformer)):
                if getattr(args, 'h3_cuda_precompute_conditioning', False):
                    from x_diffusers.adaptor.h3_conditioning import request_conditioning
                    with request_conditioning(transformer):
                        result = self.pipe(output=output_keys, **infer_params)
                else:
                    result = self.pipe(output=output_keys, **infer_params)
        finally:
            self._mmh3_cleanup(transformer, denoiser_patched, orig_denoiser_call)

        # --- SeedVR2 super-resolution post-processing ---
        if need_sr and self.sr_pipe is not None:
            logger.info(f"[Rank {rank}] Running SeedVR2 super-resolution 768p -> 1080p...")
            self._mmh3_post_super_resolution(result, args, rank)
            args.width, args.height = orig_w, orig_h
            logger.info(f"[Rank {rank}] Super-resolution complete")

        return result

    def infer_single(self, args, task_id=None, task_manager=None) -> np.ndarray:
        rank = dist.get_rank() if dist.is_initialized() else 0

        # --- MiniMax-H3 branch (ModularPipeline, joint video+audio) ---
        if "minimax-h3" in args.model.lower():
            return self._infer_minimax_h3(args, task_id, task_manager)

        need_super_resolution = (args.width == 1920 and args.height == 1080) or (args.width == 1080 and args.height == 1920)
        
        if need_super_resolution:
            logger.info(f"[Rank {rank}] 1080P requested, will generate 720P first then super-resolve")
            original_width, original_height = args.width, args.height
            if args.width == 1920:
                args.width, args.height = 1280, 720
            else:
                args.width, args.height = 720, 1280
        
        infer_params = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "generator": torch.Generator().manual_seed(args.seed)
        }

        if args.task_type == "i2v":
            image_path = args.i2v_image_path
            if image_path.startswith('data:image/'):
                image_path = decode_base64_image(image_path)
                infer_info.i2v_image_path = image_path
            infer_info.ada_brighten = True
            image = load_image(image_path)
            height, width = self.wan_image_preprocess(self.pipe, image, args.height, args.width)
            infer_info.update_shape(width, height)
            image = image.resize((height, width))
            infer_params["height"] = height
            infer_params["width"] = width
            infer_params["image"] = image

        if task_id and task_manager:
            total_steps = args.num_inference_steps
            
            def progress_callback(pipeline, step: int, timestep: int, callback_kwargs: dict):
                progress_pct = min(100.0, (step + 1) / total_steps * 100.0)
                with task_manager.tasks_lock:
                    task = task_manager.tasks.get(task_id)
                    if task:
                        task.progress = progress_pct
                        if task.status == "cancelled":
                            raise InterruptedError(f"Task {task_id} cancelled at step {step}")
                return callback_kwargs
            
            infer_params["callback_on_step_end"] = progress_callback
            infer_params["callback_on_step_end_tensor_inputs"] = ["latents"]

        output = self.pipe(**infer_params).frames[0]
        
        if args.ten_second and self.v2v_pipe is not None:
            logger.info(f"[Rank {rank}] Running V2V extension for 10s video...")
            
            his = 17
            cond_pos_list = [0, 1, 2, 3, 4]
            noise_mult_list = [0, 0.1, 0.3, 0.5, 0.7]
            steps_v2v = 4
            last_frames = output[-his:]
            
            g_ext = torch.Generator().manual_seed(args.seed)
            ext = self.v2v_pipe(
                conditioning_video=last_frames,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                conditioning_indices=cond_pos_list,
                conditioning_noise_multipliers=noise_mult_list,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=steps_v2v,
                generator=g_ext,
                output_type="np",
            ).frames[0]
            
            output = np.concatenate([output[:-his], ext], axis=0)
            logger.info(f"[Rank {rank}] V2V extension complete, total frames: {len(output)}")

        if getattr(args, 'frame_interpolation', False) and args.save_fps > 16 and args.ten_second:
            output = self._do_frame_interpolation(output, args)
            logger.info(f"[Rank {rank}] Frame interpolation complete, final frames: {len(output)}")

        if need_super_resolution and self.sr_pipe is not None:
            logger.info(f"[Rank {rank}] Running super resolution 720P -> 1080P...")
            if self.v2v_pipe is not None:
                del self.v2v_pipe
                self.v2v_pipe = None
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"[Rank {rank}] V2V pipe released before SR")
            output = self.sr_pipe(output)
            args.width, args.height = original_width, original_height
            logger.info(f"[Rank {rank}] Super resolution complete")

        return output

    def wan_image_preprocess(self, pipe, image, height, width):
        """自适应视频帧宽帧高"""
        aspect_ratio = image.height / image.width
        max_area = height * width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        new_height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        new_width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        # Update the args
        return new_height, new_width

    def _load_interpolation_model(self, model_path: str):
        if self.interpolation_model is not None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[Rank {rank}] Loading interpolation model from {model_path}...")
        self.interpolation_model = IRFNet_S()
        self.interpolation_model.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info(f"[Rank {rank}] Interpolation model loaded")

    def _calculate_fps_config(self, fps: int) -> Tuple[int, bool]:
        multiplier = fps // 16
        is_skip = False
        
        if fps % 16 == 0:
            pass
        elif fps % 8 < 4:
            multiplier += 1
            is_skip = True
        else:
            multiplier += 1
        
        return multiplier, is_skip

    def _split_residual_frames(self, output_tensor: torch.Tensor, rank: int, world_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_frames = output_tensor.shape[0]
        
        if num_frames % world_size != 0 and rank == 0:
            mod = num_frames % world_size
            input1 = output_tensor[:mod]
            output_tensor = output_tensor[mod:]
        else:
            input1 = None
    
        return output_tensor, input1
    
    def _get_input_frames(self, output_tensor: torch.Tensor, rank: int, world_size: int, step: int) -> torch.Tensor:
        if rank != (world_size - 1):
            return output_tensor[step * rank: step * (rank + 1) + 1]
        else:
            return torch.cat([output_tensor[step * rank: step * (rank + 1)], output_tensor[-1:]], dim=0)

    def _process_residual_frames(self, input1: Optional[torch.Tensor], device, multiplier: int, is_skip: bool, out: torch.Tensor) -> torch.Tensor:
        out1 = torch.zeros_like(out)[:0].to(device).contiguous()
        
        if input1 is None:
            return out1
        
        if input1.shape[0] >= 2:
            input1_permuted = input1.permute(0, 3, 1, 2)
            out1 = vfi(input1_permuted, device, self.interpolation_model, 1, multiplier=multiplier, is_skip=is_skip)
            out1 = out1.permute(0, 2, 3, 1).contiguous().cpu()
        else:
            out1 = input1.contiguous().cpu()
        
        return out1

    def _gather_results(self, out: torch.Tensor, world_size: int) -> torch.Tensor:
        if world_size == 1:
            return out.cpu()
        
        local_shape = torch.tensor(out.shape, device=out.device)
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape)
        
        max_frames = max(s[0].item() for s in all_shapes)
        padded_out = torch.zeros(max_frames, out.shape[1], out.shape[2], out.shape[3], device=out.device, dtype=out.dtype)
        padded_out[:out.shape[0]] = out
        
        gathered = [torch.zeros(max_frames, out.shape[1], out.shape[2], out.shape[3], device=out.device, dtype=out.dtype) for _ in range(world_size)]
        dist.all_gather(gathered, padded_out)
        
        output_result = torch.cat([g[:all_shapes[i][0].item()] for i, g in enumerate(gathered)], dim=0).cpu()
        return output_result

    def _do_frame_interpolation(self, output: np.ndarray, args) -> np.ndarray:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        frame_model_path = getattr(args, 'frame_model_path', TEN_SECOND_CONFIG['frame_model_path'])
        self._load_interpolation_model(frame_model_path)

        output_tensor = torch.from_numpy(output)
        output_tensor, input1 = self._split_residual_frames(output_tensor, rank, world_size)

        step = output_tensor.shape[0] // world_size
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        input_frames = self._get_input_frames(output_tensor, rank, world_size, step)

        fps = args.save_fps
        multiplier, is_skip = self._calculate_fps_config(fps)
        logger.info(f"[Rank {rank}] Frame interpolation: fps={fps}, multiplier={multiplier}, is_skip={is_skip}")

        device = rank if torch.npu.is_available() else 'cpu'
        out = vfi(input_frames, device, self.interpolation_model, 1, multiplier=multiplier, is_skip=is_skip)[:-1]
        out = out.permute(0, 2, 3, 1).contiguous()

        out1 = torch.zeros_like(out)[:0].to(device).contiguous()
        if rank == 0:
            out1 = self._process_residual_frames(input1, device, multiplier, is_skip, out)

        output_result = self._gather_results(out, world_size)

        if input1 is not None:
            output_result = torch.cat([out1, output_result], dim=0)

        return output_result.detach().numpy()


class VideoGenerationService:
    def __init__(self, config_path: Optional[str] = None, *, maas=False, maas_fl2va_input=False,
                 maas_queue_capacity=MAAS_QUEUE_CAPACITY, maas_queue_timeout_s=MAAS_QUEUE_TIMEOUT_S,
                 maas_state_dir=None, maas_mock_backends=False):
        self.app = Flask(__name__)
        self.maas = maas
        maas_fl2va_input = maas_fl2va_input or maas
        self.maas_fl2va_input = maas_fl2va_input
        self.maas_admission = threading.BoundedSemaphore(2)
        self.maas_health = MaaSHealth() if maas_fl2va_input else None
        if CORS_AVAILABLE:
            CORS(self.app)

        self.inference_manager = None
        self.default_config = self._load_default_config(config_path)
        self.is_initialized = False

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.task_queue = None if maas_fl2va_input else queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
        self.tasks: Dict[str, AsyncTask] = {}
        self.tasks_lock = threading.Lock()
        if maas_fl2va_input:
            self.maas_pending = {}
            self.maas_queue_changed = threading.Condition(self.tasks_lock)
            self.maas_queue_capacity = maas_queue_capacity
            self.maas_queue_timeout_s = maas_queue_timeout_s

        self.max_tasks = 1000
        self.task_timeout = timedelta(hours=1)
        self.keep_completed_hours = 2
        self.cleanup_interval = 300
        
        self.shutdown = False
        self._setup_signal_handlers()

        self._setup_health_routes()
        self._setup_task_routes()
        self._setup_config_routes()
        self._setup_internal_task_routes()
        self._setup_internal_query_routes()
        if self.maas:
            from maas_v2 import MaaSV2
            self.maas_v2 = MaaSV2(self, directory=maas_state_dir, mock_backends=maas_mock_backends)

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"[Rank {self.rank}] Received signal {signum}, initiating graceful shutdown...")
            self.shutdown = True
            if self.world_size > 1 and self.rank == 0:
                try:
                    for _ in range(self.world_size - 1):
                        broadcast_int(CMD_SHUTDOWN, src=0)
                    logger.info(f"[Rank 0] Sent shutdown command to all workers")
                except Exception as e:
                    logger.error(f"[Rank 0] Failed to send shutdown command: {e}")
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        logger.info(f"[Rank {self.rank}] Service exiting, cleaning up...")
        with self.tasks_lock:
            if self.maas_fl2va_input:
                self.maas_pending.clear()
            for task in self.tasks.values():
                if self.maas_fl2va_input:
                    task.finish('failed', code='service_shutdown', stage='shutdown')
                elif task.status in ["pending", "running"]:
                    task.status = "cancelled"
                    task.error = "Service shutdown"
        if getattr(self, 'maas_v2', None):
            self.maas_v2.close(wait=False)

    def _cleanup_completed_tasks(self):
        now = datetime.now()
        cutoff = now - timedelta(hours=self.keep_completed_hours)
        with self.tasks_lock:
            if self.maas_fl2va_input:
                self._expire_maas_pending(time.monotonic())
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in ["completed", "failed", "cancelled"]:
                    if task.completed_at and task.completed_at < cutoff:
                        to_remove.append(task_id)
            for task_id in to_remove:
                del self.tasks[task_id]
            if to_remove:
                logger.info(f"[Rank 0] Cleaned up {len(to_remove)} completed tasks")

    def _load_default_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        config = {
            "model": "Wan2.2-T2V-A14B",
            "pretrained_model_name_or_path": "./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "task_type": "t2v",
            "dtype": "bf16",
            "sp": 1,
            "fsdp": "all",
            "vae_lightning": "decoder",
            "turbo_mode": "default",
            "inf_vram_blocks_num": 0,
            "atten": "laser",
            "matmul_quant": "int8",
            "rope_fused": True,
            "num_inference_steps": 40,
            "save_fps": 16,
            "ten_second": False,
            "ten_second_model_id_t2v": TEN_SECOND_CONFIG["ten_second_model_id_t2v"],
            "ten_second_model_path": TEN_SECOND_CONFIG["ten_second_model_path"],
            "ten_second_model_path_2": TEN_SECOND_CONFIG["ten_second_model_path_2"],
            "pusa_lora": TEN_SECOND_CONFIG["pusa_lora"],
            "pusa_lora2": TEN_SECOND_CONFIG["pusa_lora2"],
            "joint": False,
            "joint_model_path": TEN_SECOND_CONFIG["joint_model_path"],
            "x": False,
            "x_model_path": TEN_SECOND_CONFIG["x_model_path"],
            "x_model_path_2": TEN_SECOND_CONFIG["x_model_path_2"],
            "adopt_sr": False,
            "resolution": 480,
            "seedvr2_model_dir": TEN_SECOND_CONFIG["seedvr2_model_dir"],
            "seedvr2_model_name": TEN_SECOND_CONFIG["seedvr2_model_name"],
            "frame_interpolation": False,
            "frame_model_path": TEN_SECOND_CONFIG["frame_model_path"],
            "external_base_url": None,
            "lora_path_list": None,
            "lora_transformer_list": None,
            "lora_scale_weight_list": [1.0, 1.0]
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # If loading a MiniMax-H3 config, merge with defaults
                if file_config.get("model", "").lower() == "minimax-h3":
                    from x_diffusers.adaptor.h3_features import normalize_mapping
                    file_config = normalize_mapping(file_config)
                    config.update(DEFAULT_MINIMAX_H3_CONFIG)
                config.update(file_config)

        return config

    def _validate_and_convert_param(self, value, param_name, param_type, default=None, min_val=None, max_val=None):
        if value is None:
            return default, None
        
        try:
            if param_type == int:
                converted = int(value)
            elif param_type == float:
                converted = float(value)
            elif param_type == str:
                converted = str(value)
            else:
                converted = value
            
            if min_val is not None and converted < min_val:
                return default, f"{param_name} must be >= {min_val}, got {converted}"
            if max_val is not None and converted > max_val:
                return default, f"{param_name} must be <= {max_val}, got {converted}"
            
            return converted, None
        except (ValueError, TypeError):
            return default, f"{param_name} must be {param_type.__name__}, got {type(value).__name__}"

    ALLOWED_RESOLUTIONS = {
        '480p': [(832, 480), (480, 832)],
        '720p': [(1280, 720), (720, 1280)],
        '1080p': [(1920, 1080), (1080, 1920)],
    }
    KNOWN_PARAMS = {
        'prompt', 'negative_prompt', 'width', 'height', 'frames',
        'num_inference_steps', 'seed', 'guidance_scale', 'guidance_scale_2',
        'i2v_image_path', 'save_path', 'ten_second', 'adopt_sr',
        'resolution', 'frame_interpolation', 'duration', 'save_fps',
        # MiniMax-H3 specific
        'model', 'task_type', 'first_frame', 'last_frame',
        'ref_path', 'audio_save_path', 'attention_sparse',
        'enable_audio', 'duration', 'aspect_ratio',
    }

    def _validate_prompt(self, data: Dict[str, Any], errors: List[str]) -> Optional[str]:
        prompt = data.get('prompt')
        model = data.get('model', self.default_config.get('model', ''))
        h3 = 'minimax-h3' in model.lower()
        max_length = H3_MAX_PROMPT_LENGTH if h3 else 2000
        length = h3_prompt_length if h3 else len
        if not prompt:
            errors.append("prompt is required and cannot be empty")
        elif not isinstance(prompt, str):
            errors.append("prompt must be a string")
        elif len(prompt.strip()) == 0:
            errors.append("prompt cannot be blank")
        elif length(prompt) > max_length:
            unit = 'UTF-16 code units' if h3 else 'characters'
            errors.append(f"prompt length cannot exceed {max_length} {unit}")
        return prompt

    def _get_resolution_key(self, width: int, height: int) -> Optional[str]:
        for key, sizes in self.ALLOWED_RESOLUTIONS.items():
            if (width, height) in sizes:
                return key
        return None

    def _validate_resolution_param(self, data: Dict[str, Any], errors: List[str]) -> Tuple[int, int]:
        resolution = data.get('resolution')
        if resolution is None:
            return self._validate_width_height(data, errors)
        
        if isinstance(resolution, str):
            resolution = resolution.lower()
            if resolution not in self.ALLOWED_RESOLUTIONS:
                errors.append(f"resolution must be one of: {', '.join(self.ALLOWED_RESOLUTIONS.keys())}, got {resolution}")
                return 832, 480
            return self.ALLOWED_RESOLUTIONS[resolution][0]
        
        if isinstance(resolution, int):
            res_map = {480: '480p', 720: '720p', 1080: '1080p'}
            if resolution not in res_map:
                errors.append(f"resolution must be 480, 720, or 1080, got {resolution}")
                return 832, 480
            return self.ALLOWED_RESOLUTIONS[res_map[resolution]][0]
        
        errors.append("resolution must be string (480p/720p/1080p) or int (480/720/1080)")
        return 832, 480

    def _validate_width_height(self, data: Dict[str, Any], errors: List[str]) -> Tuple[int, int]:
        width = data.get('width')
        height = data.get('height')
        
        if width is None and height is None:
            return 832, 480
        
        width, width_err = self._validate_and_convert_param(width, 'width', int, default=832, min_val=64, max_val=2048)
        if width_err:
            errors.append(width_err)
        elif width % 2 != 0:
            errors.append(f"width must be even, got {width}")
        
        height, height_err = self._validate_and_convert_param(height, 'height', int, default=480, min_val=64, max_val=2048)
        if height_err:
            errors.append(height_err)
        elif height % 2 != 0:
            errors.append(f"height must be even, got {height}")
        
        res_key = self._get_resolution_key(width, height)
        if res_key is None:
            allowed = [f"{w}x{h} ({k})" for k, sizes in self.ALLOWED_RESOLUTIONS.items() for w, h in sizes]
            errors.append(f"Invalid resolution {width}x{height}. Allowed: {', '.join(allowed)}")
        
        return width, height

    def _validate_duration(self, data: Dict[str, Any], errors: List[str]) -> int:
        duration = data.get('duration', 5)
        if duration is None:
            return 5
        
        duration = self._convert_to_int(duration, errors, "duration")
        if duration not in (5, 10):
            errors.append(f"duration must be 5 or 10, got {duration}")
            return 5
        return duration

    def _convert_to_int(self, value: Any, errors: List[str], name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                errors.append(f"{name} must be an integer")
                return 5
        if isinstance(value, float):
            return int(value)
        errors.append(f"{name} must be an integer")
        return 5

    def _validate_boolean_param(self, value: Any, errors: List[str], name: str) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        if isinstance(value, int):
            return bool(value)
        errors.append(f"{name} must be a boolean")
        return False

    def _validate_save_fps(self, data: Dict[str, Any], errors: List[str]) -> int:
        save_fps = data.get('save_fps', 16)
        if save_fps is None:
            return 16
        save_fps = self._convert_to_int(save_fps, errors, "save_fps")
        if save_fps not in (16, 24, 30):
            errors.append(f"save_fps must be 16, 24, or 30, got {save_fps}")
            return 16
        return save_fps

    def _validate_numeric_params(self, data: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        turbo_mode = self.default_config.get('turbo_mode')
        if turbo_mode == 'default':
            num = 1
        else:
            num = 10
        params = {}
        param_specs = [
            ('frames', 'frames', int, 81, 1, 301),
            ('num_inference_steps', 'num_inference_steps', int, 40, num, 200),
            ('seed', 'seed', int, 42, 0, 2**31 - 1),
            ('guidance_scale', 'guidance_scale', float, None, 0, 100),
            ('guidance_scale_2', 'guidance_scale_2', float, None, 0, 100),
        ]
        for key, name, ptype, default, min_val, max_val in param_specs:
            value, err = self._validate_and_convert_param(data.get(key), name, ptype, default=default, min_val=min_val, max_val=max_val)
            if err:
                errors.append(err)
            params[key] = value
        return params

    def _validate_negative_prompt(self, data: Dict[str, Any], errors: List[str]) -> str:
        negative_prompt = data.get('negative_prompt')
        if negative_prompt is None:
            return ""
        if not isinstance(negative_prompt, str):
            errors.append("negative_prompt must be a string")
            return ""
        if len(negative_prompt) > 2000:
            errors.append("negative_prompt length cannot exceed 2000 characters")
            return ""
        return negative_prompt

    def _validate_i2v_params(self, data: Dict[str, Any], errors: List[str], warnings: List[str]) -> Optional[str]:
        task_type = self.default_config.get('task_type', 't2v')
        i2v_image_path = data.get('i2v_image_path')
        if task_type == 'i2v':
            if not i2v_image_path:
                errors.append("i2v_image_path is required for I2V model")
            elif not isinstance(i2v_image_path, str):
                errors.append("i2v_image_path must be a string")
        elif i2v_image_path is not None:
            warnings.append("i2v_image_path is ignored for T2V model")
        return i2v_image_path

    def _generate_param_warnings(self, frames: int, unknown_params: set, ten_second: bool, 
                                  adopt_sr: bool, resolution: int, save_fps: int, 
                                  frame_interpolation: bool, duration: int, data: Dict, 
                                  warnings: List[str]) -> bool:
        if frames and frames > 81:
            warnings.append(f"Large frame count ({frames}) will significantly increase generation time")
        if unknown_params:
            warnings.append(f"Unknown parameters ignored: {', '.join(unknown_params)}")
        if data.get('ten_second') is not None:
            explicit_ten_second = self._validate_boolean_param(data.get('ten_second'), [], 'ten_second')
            if explicit_ten_second != (duration == 10):
                warnings.append(f"ten_second={explicit_ten_second} is overridden by duration={duration}")
        if save_fps > 16 and not frame_interpolation:
            frame_interpolation = True
            warnings.append(f"save_fps={save_fps} > 16, auto-enabling frame_interpolation")
        if ten_second and adopt_sr and resolution > 720:
            warnings.append(f"High resolution ({resolution}p) with 10s + SR may require significant memory")
        return frame_interpolation

    def _validate_negative_prompt(self, data: Dict[str, Any], errors: List[str]) -> str:
        negative_prompt = data.get('negative_prompt')
        if negative_prompt is None:
            return ""
        if not isinstance(negative_prompt, str):
            errors.append("negative_prompt must be a string")
            return ""
        if len(negative_prompt) > 2000:
            errors.append("negative_prompt length cannot exceed 2000 characters")
            return ""
        return negative_prompt

    def _validate_i2v_params(self, data: Dict[str, Any], errors: List[str], warnings: List[str]) -> Optional[str]:
        task_type = self.default_config.get('task_type', 't2v')
        i2v_image_path = data.get('i2v_image_path')
        if task_type == 'i2v':
            if not i2v_image_path:
                errors.append("i2v_image_path is required for I2V model")
            elif not isinstance(i2v_image_path, str):
                errors.append("i2v_image_path must be a string")
        elif i2v_image_path is not None:
            warnings.append("i2v_image_path is ignored for T2V model")
        return i2v_image_path

    def _generate_param_warnings(self, frames: int, unknown_params: set, ten_second: bool, 
                                  adopt_sr: bool, resolution: int, save_fps: int, 
                                  frame_interpolation: bool, duration: int, data: Dict, 
                                  warnings: List[str]) -> bool:
        if frames and frames > 81:
            warnings.append(f"Large frame count ({frames}) will significantly increase generation time")
        if unknown_params:
            warnings.append(f"Unknown parameters ignored: {', '.join(unknown_params)}")
        if data.get('ten_second') is not None:
            explicit_ten_second = self._validate_boolean_param(data.get('ten_second'), [], 'ten_second')
            if explicit_ten_second != (duration == 10):
                warnings.append(f"ten_second={explicit_ten_second} is overridden by duration={duration}")
        if save_fps > 16 and not frame_interpolation:
            frame_interpolation = True
            warnings.append(f"save_fps={save_fps} > 16, auto-enabling frame_interpolation")
        if ten_second and adopt_sr and resolution > 720:
            warnings.append(f"High resolution ({resolution}p) with 10s + SR may require significant memory")
        return frame_interpolation

    def _validate_http_url(self, url: str, param_name: str, errors: List[str]) -> None:
        """Validate that an HTTP/HTTPS URL is reachable via HEAD (fallback GET)."""
        url_lower = url.lower()
        if not url_lower.startswith(('http://', 'https://')):
            if url_lower.startswith('http'):
                errors.append(f"{param_name} malformed URL (expected http:// or https://): {url}")
            return
        import urllib.request
        try:
            req = urllib.request.Request(url, method='HEAD')
            resp = urllib.request.urlopen(req, timeout=10)
            resp.close()
        except Exception:
            try:
                req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
                resp = urllib.request.urlopen(req, timeout=10)
                resp.close()
            except Exception as e:
                errors.append(f"{param_name} URL is not reachable: {url} ({e})")

    def _validate_image_input(self, input_data: str, param_name: str, errors: List[str]) -> None:
        """Validate base64 image data at request time to catch errors early."""
        if input_data.startswith('data:image/'):
            parts = input_data.split(',', 1)
            if len(parts) != 2:
                errors.append(f"{param_name}: invalid data URI format, expected 'data:image/<type>;base64,<data>'")
                return
            _, b64_data = parts
        elif _is_plain_base64(input_data):
            b64_data = input_data
        else:
            # Local file path — check existence and image validity
            if input_data.startswith('/'):
                if not os.path.isfile(input_data):
                    errors.append(f"{param_name}: file not found: {input_data}")
                else:
                    try:
                        with Image.open(input_data) as img:
                            img.verify()
                    except Exception as e:
                        errors.append(f"{param_name}: not a valid image file - {str(e)}")
            else:
                errors.append(f"{param_name}: unsupported input format, expected base64 data URI, HTTP URL, or absolute file path starting with '/'")
            return
        try:
            decoded = base64.b64decode(b64_data)
        except Exception as e:
            errors.append(f"{param_name}: invalid base64 data - {str(e)}")
            return
        if not decoded:
            errors.append(f"{param_name}: empty image data after base64 decode")
            return
        try:
            with Image.open(io.BytesIO(decoded)) as img:
                img.verify()
        except Exception as e:
            errors.append(f"{param_name}: decoded data is not a valid image - {str(e)}")

    def _validate_media_input(self, input_data: str, param_name: str,
                              errors: List[str]) -> None:
        """Validate media input (image/video/audio) for ref_path.

        Unlike _validate_image_input, this accepts video and audio files
        in addition to images. base64 input is still limited to images.
        """
        # base64 data URI or plain base64 → only images supported via base64
        if input_data.startswith('data:image/') or _is_plain_base64(input_data):
            self._validate_image_input(input_data, param_name, errors)
            return
        # HTTP URL → reachability already checked by _validate_http_url
        if input_data.startswith(('http://', 'https://')):
            return
        # Local file path — check existence and validate by extension
        if input_data.startswith('/'):
            if not os.path.isfile(input_data):
                errors.append(f"{param_name}: file not found: {input_data}")
                return
            ext = os.path.splitext(input_data)[1].lower()
            if ext in VideoInferenceManager._MMH3_IMAGE_EXTS:
                try:
                    with Image.open(input_data) as img:
                        img.verify()
                except Exception as e:
                    errors.append(f"{param_name}: not a valid image file - {str(e)}")
            elif ext in VideoInferenceManager._MMH3_VIDEO_EXTS:
                logger.info(f"{param_name}: video file accepted (no deep validation): {input_data}")
            elif ext in VideoInferenceManager._MMH3_AUDIO_EXTS:
                logger.info(f"{param_name}: audio file accepted (no deep validation): {input_data}")
            else:
                supported = (f"images {sorted(VideoInferenceManager._MMH3_IMAGE_EXTS)}, "
                             f"videos {sorted(VideoInferenceManager._MMH3_VIDEO_EXTS)}, "
                             f"audio {sorted(VideoInferenceManager._MMH3_AUDIO_EXTS)}")
                errors.append(f"{param_name}: unsupported file type '{ext}', "
                              f"supported: {supported}")
        else:
            errors.append(f"{param_name}: unsupported input format, expected base64 data URI, HTTP URL, or absolute file path starting with '/'")
        return

    def _validate_mmh3_frame_fields(self, data: Dict[str, Any], task_type: str,
                                     errors: List[str]) -> tuple:
        """Validate first_frame and last_frame for MiniMax-H3."""
        first_frame = data.get('first_frame', '')
        if task_type == 'fl2va' and not first_frame:
            errors.append("first_frame is required for fl2va task")
        if first_frame and not isinstance(first_frame, str):
            errors.append("first_frame must be a string (base64 / URL / file path)")
        elif first_frame:
            self._validate_http_url(first_frame, 'first_frame', errors)
            self._validate_image_input(first_frame, 'first_frame', errors)

        last_frame = data.get('last_frame', '')
        if last_frame and not isinstance(last_frame, str):
            errors.append("last_frame must be a string (base64 / URL / file path)")
        elif last_frame:
            self._validate_http_url(last_frame, 'last_frame', errors)
            self._validate_image_input(last_frame, 'last_frame', errors)
        return first_frame, last_frame

    def _get_media_type(self, input_data: str) -> str:
        """Determine media type from input data.

        Returns 'image', 'video', 'audio', or '' (unknown).
        """
        # base64 data URI or plain base64 → always image
        if input_data.startswith('data:image/') or _is_plain_base64(input_data):
            return 'image'
        # HTTP URL → infer from URL path extension
        if input_data.startswith(('http://', 'https://')):
            ext = os.path.splitext(input_data.split('?')[0])[1].lower()
            if ext in VideoInferenceManager._MMH3_IMAGE_EXTS:
                return 'image'
            if ext in VideoInferenceManager._MMH3_VIDEO_EXTS:
                return 'video'
            if ext in VideoInferenceManager._MMH3_AUDIO_EXTS:
                return 'audio'
            return ''
        # Local file path → infer from extension
        if input_data.startswith('/'):
            ext = os.path.splitext(input_data)[1].lower()
            if ext in VideoInferenceManager._MMH3_IMAGE_EXTS:
                return 'image'
            if ext in VideoInferenceManager._MMH3_VIDEO_EXTS:
                return 'video'
            if ext in VideoInferenceManager._MMH3_AUDIO_EXTS:
                return 'audio'
            return ''
        return ''

    def _validate_mmh3_ref_path(self, data: Dict[str, Any], task_type: str,
                                errors: List[str]):
        """Validate ref_path for MiniMax-H3 ref2va task.

        Enforces model-level modality constraints:
        - At most 9 image references
        - At most 3 video references
        - At most 3 audio references
        - Audio references cannot be used alone (must pair with ≥1 image or video)
        """
        ref_path = data.get('ref_path')
        if task_type == 'ref2va' and not ref_path:
            errors.append("ref_path is required for ref2va task")
        if not ref_path:
            return None
        if isinstance(ref_path, str):
            ref_path_list = [ref_path]
        elif isinstance(ref_path, list):
            ref_path_list = ref_path
        else:
            errors.append("ref_path must be a string or list of strings")
            return None

        img_count = video_count = audio_count = 0
        for rp in ref_path_list:
            if not isinstance(rp, str):
                errors.append("each ref_path item must be a string")
                continue
            self._validate_http_url(rp, 'ref_path', errors)
            self._validate_media_input(rp, 'ref_path', errors)
            mt = self._get_media_type(rp)
            if mt == 'image':
                img_count += 1
            elif mt == 'video':
                video_count += 1
            elif mt == 'audio':
                audio_count += 1

        if len(ref_path_list) > MAX_FILES:
            errors.append(f"ref_path: maximum {MAX_FILES} reference files total")
        for kind, count in (("image", img_count), ("video", video_count), ("audio", audio_count)):
            if count > LIMITS[kind]:
                errors.append(f"ref_path: too many {kind} references ({count}), maximum {LIMITS[kind]}")
        if audio_count > 0 and img_count == 0 and video_count == 0:
            errors.append("ref_path: audio references cannot be used alone, "
                          "must be paired with at least one image or video reference")

        return ref_path

    def _validate_mmh3_steps(self, data: Dict[str, Any], errors: List[str]) -> int:
        """Validate Base denoising steps against the loaded AdaLN schedule."""
        cache_file = self.default_config.get('adaln_cache_file')
        expected_steps = None
        if cache_file:
            try:
                expected_steps = cache_steps(cache_file)
            except ValueError as exc:
                errors.append(str(exc))
                return 50
        default_steps = expected_steps if expected_steps is not None else self.default_config.get('num_inference_steps', 50)
        steps = data.get('num_inference_steps', default_steps)
        if type(steps) is not int or not 1 <= steps <= 200:
            errors.append("num_inference_steps must be an integer in [1, 200]")
        elif expected_steps is not None and steps != expected_steps:
            errors.append(f"num_inference_steps must be {expected_steps} to match the Base AdaLN cache")
        return steps

    def _generate_mmh3_warnings(self, data: Dict[str, Any], resolution: str, warnings: List[str]) -> None:
        """Generate warnings for MiniMax-H3 parameter issues."""
        unknown_params = set(data.keys()) - self.KNOWN_PARAMS
        if unknown_params:
            warnings.append(f"Unknown parameters ignored: {', '.join(unknown_params)}")
        if 'width' in data or 'height' in data:
            warnings.append("width/height are not supported for MiniMax-H3; use 'resolution' + 'aspect_ratio' instead")
        if resolution == '1080p':
            warnings.append("1080p resolution will significantly increase generation time and memory")

    def _validate_minimax_h3_params(self, data: Dict[str, Any], errors: List[str], warnings: List[str]) -> tuple:
        """Validate parameters for MiniMax-H3 model (t2va/fl2va/ref2va)."""
        if self.default_config.get('h3_attention', 'legacy') != 'legacy':
            from x_diffusers.adaptor.h3_attention_service import validate_request as validate_sparse_request
            validate_sparse_request(self.default_config, data, errors)
        if self.default_config.get('h3_refiner', False):
            from x_diffusers.adaptor.h3_refiner_service import validate_request
            return validate_request(self, data, errors, warnings)
        prompt = self._validate_prompt(data, errors)

        resolution = data.get('resolution', self.default_config.get('resolution', '768p'))
        if resolution not in ('768p', '1080p'):
            errors.append(f"resolution must be '768p' or '1080p', got '{resolution}'")

        # aspect_ratio: controls orientation (16:9 landscape or 9:16 portrait)
        aspect_ratio = data.get('aspect_ratio', '16:9')
        if aspect_ratio not in ('16:9', '9:16'):
            errors.append(f"aspect_ratio must be '16:9' or '9:16', got '{aspect_ratio}'")

        duration_raw = data.get('duration')
        if isinstance(duration_raw, float) and duration_raw != int(duration_raw):
            errors.append(f"duration must be an integer (4-{H3_BASE_MAX_DURATION_S}), got {duration_raw}")
        duration, err = self._validate_and_convert_param(
            duration_raw, 'duration', int, 5, 4, H3_BASE_MAX_DURATION_S
        )
        if err:
            errors.append(err)
        frames = duration * 24 if duration else 120

        steps = self._validate_mmh3_steps(data, errors)
        seed, err = self._validate_and_convert_param(data.get('seed'), 'seed', int, 42, 0, 2**31 - 1)
        if err:
            errors.append(err)

        save_fps = 24

        task_type = data.get('task_type', self.default_config.get('task_type', 'fl2va'))
        if task_type not in ('t2va', 'fl2va', 'ref2va'):
            errors.append(f"task_type must be one of t2va/fl2va/ref2va, got {task_type}")

        # Service mode check: restrict allowed task types based on --service-mode
        service_mode = self.default_config.get('service_mode', '')
        if service_mode in ('t2va', 'fl2va') and task_type == 'ref2va':
            errors.append(f"task_type 'ref2va' is not supported in t2va service mode (this service supports t2va and fl2va only)")
        elif service_mode == 'ref2va' and task_type in ('t2va', 'fl2va'):
            errors.append(f"task_type '{task_type}' is not supported in ref2va service mode (this service supports ref2va only)")

        first_frame, last_frame = self._validate_mmh3_frame_fields(data, task_type, errors)
        ref_path = self._validate_mmh3_ref_path(data, task_type, errors)

        audio_save_path = data.get('audio_save_path', '')
        if audio_save_path and not isinstance(audio_save_path, str):
            errors.append("audio_save_path must be a string")

        attention_sparse = self._validate_boolean_param(data.get('attention_sparse'), errors, 'attention_sparse')

        enable_audio = data.get('enable_audio', True)
        if not isinstance(enable_audio, bool):
            errors.append("enable_audio must be a boolean (true/false)")

        adopt_sr = data.get('adopt_sr', False)
        if not isinstance(adopt_sr, bool):
            errors.append("adopt_sr must be a boolean (true/false)")

        self._generate_mmh3_warnings(data, resolution, warnings)

        if errors:
            return False, {"errors": errors, "warnings": warnings}

        validated_data = {
            'model': 'MiniMax-H3',
            'task_type': task_type,
            'prompt': prompt,
            'resolution': resolution,
            'aspect_ratio': aspect_ratio,
            'duration': duration,
            'frames': frames,
            'num_inference_steps': steps,
            'seed': seed,
            'save_fps': save_fps,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'ref_path': ref_path,
            'audio_save_path': audio_save_path,
            'attention_sparse': attention_sparse,
            'enable_audio': enable_audio,
            'adopt_sr': adopt_sr,
        }
        if data.get('save_path'):
            validated_data['save_path'] = data['save_path']

        return True, {"data": validated_data, "warnings": warnings}

    def _validate_generate_params(self, data: Dict[str, Any]) -> tuple:
        errors = []
        warnings = []

        # --- MiniMax-H3 branch ---
        model = data.get('model', self.default_config.get('model', ''))
        if "minimax-h3" in model.lower():
            return self._validate_minimax_h3_params(data, errors, warnings)

        prompt = self._validate_prompt(data, errors)
        width, height = self._validate_resolution_param(data, errors)
        numeric_params = self._validate_numeric_params(data, errors)
        negative_prompt = self._validate_negative_prompt(data, errors)
        i2v_image_path = self._validate_i2v_params(data, errors, warnings)
        
        unknown_params = set(data.keys()) - self.KNOWN_PARAMS
        duration = self._validate_duration(data, errors)
        ten_second = (duration == 10)
        adopt_sr = self._validate_boolean_param(data.get('adopt_sr'), errors, 'adopt_sr')
        
        resolution = 480
        if width == 1280 and height == 720:
            resolution = 720
        elif width == 1920 and height == 1080:
            resolution = 1080

        frame_interpolation = self._validate_boolean_param(data.get('frame_interpolation'), errors, 'frame_interpolation')
        save_fps = self._validate_save_fps(data, errors)
        
        frame_interpolation = self._generate_param_warnings(
            numeric_params['frames'], unknown_params, ten_second, adopt_sr, 
            resolution, save_fps, frame_interpolation, duration, data, warnings
        )

        if errors:
            return False, {"errors": errors, "warnings": warnings}
        
        validated_data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'frames': numeric_params['frames'],
            'num_inference_steps': numeric_params['num_inference_steps'],
            'seed': numeric_params['seed'],
            'guidance_scale': numeric_params['guidance_scale'],
            'guidance_scale_2': numeric_params['guidance_scale_2'],
            'ten_second': ten_second,
            'adopt_sr': adopt_sr,
            'resolution': resolution,
            'frame_interpolation': frame_interpolation,
            'duration': duration,
            'save_fps': save_fps,
        }
        if i2v_image_path:
            validated_data['i2v_image_path'] = i2v_image_path
        if data.get('save_path'):
            validated_data['save_path'] = data['save_path']

        return True, {"data": validated_data, "warnings": warnings}

    def _maas_readiness(self):
        return self.maas_health.describe(
            self.is_initialized, self.shutdown, self.default_config.get('sp', self.world_size), self.world_size,
        )

    def _setup_health_routes(self):
        if self.maas_fl2va_input:
            @self.app.get('/ready')
            def readiness_check():
                snapshot = self._maas_readiness()
                return jsonify(snapshot), 200 if snapshot['ready'] else 503

        @self.app.route('/health', methods=['GET'])
        def health_check():
            current_model = None
            if self.inference_manager and hasattr(self.inference_manager, 'current_model'):
                current_model = self.inference_manager.current_model

            result = {
                "status": "healthy",
                "initialized": self.is_initialized,
                "current_model": current_model,
                "rank": self.rank,
                "world_size": self.world_size,
                "supported_models": list(MODEL_REGISTRY.keys()),
                "timestamp": datetime.now().isoformat()
            }
            if self.maas_fl2va_input:
                result.update(self._maas_readiness(), model=self.default_config.get('model'),
                              profile="maas-h3-base-768p" if self.maas else "maas-fl2va-768p-5s")
            return jsonify(result)

        @self.app.route('/models', methods=['GET'])
        def list_models():
            current_model = None
            if self.inference_manager and hasattr(self.inference_manager, 'current_model'):
                current_model = self.inference_manager.current_model

            return jsonify({
                "current_model": current_model,
                "supported_models": list(MODEL_REGISTRY.keys()),
                "model_info": MODEL_REGISTRY,
                "note": "To switch model, restart the service with different config"
            })

        @self.app.route('/init', methods=['POST'])
        def init_service():
            if self.maas_fl2va_input:
                return jsonify({"error": "Start the MaaS FL2VA service with --init"}), 405
            try:
                data = request.get_json() or {}

                init_env()
                self.inference_manager = VideoInferenceManager()
                self.is_initialized = True

                logger.info(f"[Rank {self.rank}] VideoGenerationService initialized successfully")
                return jsonify({"status": "success", "message": "Service initialized", "rank": self.rank})
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to initialize service: {str(e)}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500

    def _expire_maas_pending(self, now):
        # Called with tasks_lock held. Queue deadlines follow FIFO order.
        while self.maas_pending:
            task_id = next(iter(self.maas_pending))
            task = self.tasks[task_id]
            elapsed = now - task.queued_at
            if elapsed < self.maas_queue_timeout_s:
                break
            if self.maas:
                from maas_media import cleanup
                cleanup(self.maas_pending[task_id]['data'])
            del self.maas_pending[task_id]
            task.timings_s['queue'] = round(elapsed, 3)
            task.finish('failed', code='queue_expired', stage='queue')

    def _maas_task_snapshot(self, task, now, *, platform=True):
        result = task.to_platform_dict() if platform else task.to_dict()
        if task.status == 'pending':
            result['queue_elapsed_s'] = round(now - task.queued_at, 3)
            result['queue_ahead'] = list(self.maas_pending).index(task.task_id)
        return result

    def _read_maas_submission(self, *, internal=False):
        if not self.maas:
            return read_submission(request, internal=internal)
        from maas import read_submission as read_h3_submission
        if not self.maas_admission.acquire(blocking=False):
            raise InvalidInput('body', 'Media admission is busy; retry later', 503)
        try:
            return read_h3_submission(request, self.default_config, internal=internal)
        finally:
            self.maas_admission.release()

    def _submit_maas_task(self, task_id, data, created_at_ms=None):
        try:
            return self._accept_maas_task(task_id, data, created_at_ms)
        finally:
            if self.maas and not data.get('_maas_adopted'):
                from maas_media import cleanup
                cleanup(data)

    def _accept_maas_task(self, task_id, data, created_at_ms=None):
        key = (bytes.fromhex(data['_maas_fingerprint']) if self.maas else
               hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).digest())
        with self.maas_queue_changed:
            self._expire_maas_pending(time.monotonic())
            previous = self.tasks.get(task_id)
            if previous:
                if previous.input_key != key:
                    return jsonify({'id': task_id, 'error': 'task_conflict'}), 409
                return jsonify(self._maas_task_snapshot(previous, time.monotonic())), 200
            if self.shutdown or self._maas_readiness()['state'] == 'degraded':
                return jsonify({'error': 'worker_unavailable'}), 503
            if len(self.tasks) >= self.max_tasks:
                return jsonify({'error': 'too_many_tasks'}), 429
            if len(self.maas_pending) >= self.maas_queue_capacity:
                return jsonify({'id': task_id, 'error': 'queue_full'}), 503
            task = MaaSTask(task_id, key, created_at_ms)
            if metadata := data.pop('_maas_v2_metadata', None):
                task.v2_metadata = metadata
                task.on_change = self.maas_v2.task_changed
                task.on_change(task)
            save_path = f"/tmp/generated_videos/video_{uuid.uuid4().hex}.mp4"
            self.tasks[task_id] = task
            if self.maas:
                data['_maas_adopted'] = True
            self.maas_pending[task_id] = {'task_id': task_id, 'data': data, 'save_path': save_path}
            self.maas_queue_changed.notify()
            return jsonify(self._maas_task_snapshot(task, task.queued_at)), 200

    def _cancel_maas_task(self, task_id):
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if not task:
                return jsonify({'error': 'task_not_found'}), 404
            now = time.monotonic()
            self._expire_maas_pending(now)
            if task.status == 'pending':
                if self.maas:
                    from maas_media import cleanup
                    cleanup(self.maas_pending[task_id]['data'])
                del self.maas_pending[task_id]
                task.timings_s['queue'] = round(now - task.queued_at, 3)
                task.finish('cancelled', code='cancelled_by_user', stage='queue')
            if task.status != 'cancelled':
                return jsonify({'id': task_id, 'error': 'task_not_cancellable', 'status': task.get_platform_status()}), 409
            return jsonify(task.to_platform_dict()), 200

    def _claim_maas_task(self):
        with self.maas_queue_changed:
            self.maas_queue_changed.wait_for(lambda: self.maas_pending or self.shutdown, timeout=1.0)
            now = time.monotonic()
            self._expire_maas_pending(now)
            if not self.maas_pending or self.shutdown:
                raise queue.Empty
            task_id = next(iter(self.maas_pending))
            task_info = self.maas_pending[task_id]
            task = self.tasks[task_id]
            task.status = 'running'
            task.started_at = datetime.now()
            task.timings_s['queue'] = round(now - task.queued_at, 3)
            if getattr(task, 'on_change', None):
                task.on_change(task)
            del self.maas_pending[task_id]
            return task_info, task

    def _fail_maas_jobs(self, phase):
        with self.tasks_lock:
            if self.maas:
                from maas_media import cleanup
                for job in self.maas_pending.values():
                    cleanup(job['data'])
            self.maas_pending.clear()
            for task in self.tasks.values():
                task.finish('failed', code='worker_lost', stage=phase if task.status == 'running' else 'queue')

    def _setup_task_routes(self):
        @self.app.route('/generate', methods=['POST'])
        def submit_task():
            try:
                if not self.is_initialized:
                    return jsonify({"status": "error", "message": "Service not initialized. Call /init first"}), 400
                
                if self.shutdown:
                    return jsonify({"status": "error", "message": "Service is shutting down"}), 503

                if self.maas_health and self._maas_readiness()["state"] == "degraded":
                    return jsonify({"error": "worker_unavailable"}), 503

                self._cleanup_completed_tasks()
                
                with self.tasks_lock:
                    if len(self.tasks) >= self.max_tasks:
                        return jsonify({
                            "status": "error",
                            "message": f"Too many tasks (max: {self.max_tasks}). Please wait for tasks to complete."
                        }), 429

                if self.maas_fl2va_input:
                    task_id = str(uuid.uuid4())
                    response, status = self._submit_maas_task(task_id, self._read_maas_submission())
                    if status == 200:
                        return jsonify({"status": "accepted", "task_id": task_id}), 200
                    return response, status
                else:
                    data = request.get_json()
                    if not data:
                        return jsonify({"status": "error", "message": "No JSON data provided"}), 400

                    is_valid, result = self._validate_generate_params(data)
                    if not is_valid:
                        return jsonify({
                            "status": "error",
                            "message": "Parameter validation failed",
                            "details": result
                        }), 400

                validated_data = result["data"]
                warnings = result.get("warnings", [])

                task_id = str(uuid.uuid4())
                task = AsyncTask(
                    task_id=task_id,
                    task_type="video_generation",
                    params=validated_data
                )

                with self.tasks_lock:
                    self.tasks[task_id] = task

                save_path = validated_data.get('save_path')
                if not save_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"/tmp/generated_videos/video_{task_id[:8]}_{timestamp}.mp4"

                try:
                    self.task_queue.put({
                        'task_id': task_id,
                        'data': validated_data,
                        'save_path': save_path
                    }, block=False)
                except queue.Full:
                    with self.tasks_lock:
                        del self.tasks[task_id]
                    return jsonify({
                        "status": "error",
                        "message": "Task queue is full. Please retry later."
                    }), 503

                logger.info(f"[Rank 0] Task {task_id} submitted")
                response = {
                    "status": "accepted",
                    "task_id": task_id,
                    "message": "Task submitted successfully"
                }
                if warnings:
                    response["warnings"] = warnings
                return jsonify(response)

            except InvalidInput as e:
                return jsonify({"error": "invalid_input", "field": e.field, "message": str(e)}), e.status
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to submit task: {str(e)}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route('/status/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)
                if task is not None and self.maas_fl2va_input:
                    now = time.monotonic()
                    self._expire_maas_pending(now)
                    return jsonify(self._maas_task_snapshot(task, now, platform=False))

            if not task:
                return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

            return jsonify(task.to_dict())

        @self.app.route('/result/<task_id>', methods=['GET'])
        def get_task_result(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)

            if not task:
                return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

            if task.status != "completed":
                return jsonify({
                    "status": "error",
                    "message": f"Task is not completed. Current status: {task.status}"
                }), 400

            result = task.to_dict()
            result["video_path"] = task.result.get("video_path")
            return jsonify(result)

        @self.app.route('/download/<task_id>', methods=['GET'])
        def download_task_result(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)

            if not task:
                return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

            if task.status != "completed":
                return jsonify({
                    "status": "error",
                    "message": f"Task is not completed. Current status: {task.status}"
                }), 400

            video_path = task.result.get("video_path")
            if not video_path or not os.path.exists(video_path):
                return jsonify({"status": "error", "message": f"Video file not found: {video_path}"}), 404

            filename = os.path.basename(video_path)
            return send_file(video_path, as_attachment=True, download_name=filename, mimetype='video/mp4')

        @self.app.route('/tasks', methods=['GET'])
        def list_tasks():
            with self.tasks_lock:
                tasks_list = [task.to_dict() for task in self.tasks.values()]
            return jsonify({"tasks": tasks_list, "count": len(tasks_list)})

        @self.app.route('/cancel/<task_id>', methods=['POST'])
        def cancel_task(task_id):
            if self.maas_fl2va_input:
                return self._cancel_maas_task(task_id)
            with self.tasks_lock:
                task = self.tasks.get(task_id)

                if not task:
                    return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

                if task.status in ["completed", "failed", "cancelled"]:
                    return jsonify({
                        "status": "error",
                        "message": f"Cannot cancel task with status: {task.status}"
                    }), 400

                task.status = "cancelled"
                task.completed_at = datetime.now()
                logger.info(f"[Rank 0] Task {task_id} cancelled")
                return jsonify({"status": "success", "message": f"Task {task_id} cancelled"})

    def _setup_config_routes(self):
        @self.app.route('/params', methods=['GET'])
        def get_param_spec():
            if self.maas_fl2va_input:
                if self.maas:
                    from maas import parameter_spec
                    return jsonify(parameter_spec(self.default_config))
                return jsonify(maas_parameter_spec())
            if self.default_config.get('h3_attention', 'legacy') != 'legacy':
                from x_diffusers.adaptor.h3_attention_service import parameter_spec as sparse_parameter_spec
                return jsonify(sparse_parameter_spec(self.default_config))
            if self.default_config.get('h3_refiner', False):
                from x_diffusers.adaptor.h3_refiner_service import parameter_spec
                return jsonify(parameter_spec(self.default_config))
            h3_base = 'minimax-h3' in self.default_config.get('model', '').lower()
            step_constraints = {"min": 1, "max": 200}
            default_steps = 40
            if h3_base:
                errors = []
                default_steps = self._validate_mmh3_steps({}, errors)
                if errors:
                    return jsonify({"errors": errors}), 503
                if self.default_config.get('adaln_cache_file'):
                    step_constraints = {"min": default_steps, "max": default_steps}
            return jsonify({
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt for video generation",
                        "constraints": {"min_length": 1, "max_length": H3_MAX_PROMPT_LENGTH if h3_base else 2000},
                        **({"length_unit": "utf16_code_units"} if h3_base else {})
                    },
                    "negative_prompt": {
                        "type": "string",
                        "required": False,
                        "default": "",
                        "description": "Negative prompt to avoid certain features",
                        "constraints": {"max_length": 2000}
                    },
                    "width": {
                        "type": "integer",
                        "required": False,
                        "default": 832,
                        "description": "Video width in pixels",
                        "constraints": {"min": 64, "max": 2048}
                    },
                    "height": {
                        "type": "integer",
                        "required": False,
                        "default": 480,
                        "description": "Video height in pixels",
                        "constraints": {"min": 64, "max": 2048}
                    },
                    "frames": {
                        "type": "integer",
                        "required": False,
                        "default": 81,
                        "description": "Number of frames to generate",
                        "constraints": {"min": 1, "max": 200}
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "required": False,
                        "default": default_steps,
                        "description": "Number of denoising steps",
                        "constraints": step_constraints
                    },
                    "seed": {
                        "type": "integer",
                        "required": False,
                        "default": 42,
                        "description": "Random seed for reproducibility",
                        "constraints": {"min": 0}
                    },
                    "guidance_scale": {
                        "type": "number",
                        "required": False,
                        "default": None,
                        "description": "First guidance scale (model dependent)",
                        "constraints": {"min": 0}
                    },
                    "guidance_scale_2": {
                        "type": "number",
                        "required": False,
                        "default": None,
                        "description": "Second guidance scale (model dependent)",
                        "constraints": {"min": 0}
                    },
                    "save_path": {
                        "type": "string",
                        "required": False,
                        "default": "auto-generated",
                        "description": "Path to save the generated video"
                    },
                    "i2v_image_path": {
                        "type": "string",
                        "required": False,
                        "description": "Input image path for i2v tasks"
                    },
                    "ten_second": {
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Enable 10-second video generation (two-stage: 5s base + 5s extension)"
                    },
                    "adopt_sr": {
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Enable super-resolution (SeedVR2)"
                    },
                    "resolution": {
                        "type": "integer",
                        "required": False,
                        "default": 480,
                        "description": "Target resolution for super-resolution (480/720/1080)",
                        "constraints": {"min": 480, "max": 2160}
                    },
                    "frame_interpolation": {
                        "type": "boolean",
                        "required": False,
                        "default": False,
                        "description": "Enable frame interpolation for smoother video"
                    },
                    "duration": {
                        "type": "integer",
                        "required": False,
                        "default": 5,
                        "description": (
                            f"Video duration in seconds (4 to {H3_BASE_MAX_DURATION_S})."
                            if h3_base else
                            "Video duration in seconds (5 or 10). duration=10 enables two-stage generation."
                        ),
                        "constraints": (
                            {"min": 4, "max": H3_BASE_MAX_DURATION_S}
                            if h3_base else {"allowed_values": [5, 10]}
                        )
                    }
                },
                "example": {
                    "prompt": "A cat playing with a ball",
                    "width": 832,
                    "height": 480,
                    "frames": 81,
                    "num_inference_steps": 40,
                    "seed": 42,
                    "duration": 5
                },
                "example_10s": {
                    "prompt": "A cat playing with a ball",
                    "width": 832,
                    "height": 480,
                    "frames": 81,
                    "num_inference_steps": 6,
                    "seed": 42,
                    "duration": 10,
                    "adopt_sr": True,
                    "resolution": 1080
                }
            })

        @self.app.route('/config', methods=['GET', 'POST'])
        def manage_config():
            if request.method == 'GET':
                return jsonify(self.default_config)
            else:
                if self.maas_fl2va_input:
                    return jsonify({"error": "MaaS FL2VA profile is fixed at startup"}), 405
                data = request.get_json()
                self.default_config.update(data)
                return jsonify({"status": "success", "config": self.default_config})

        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            with self.tasks_lock:
                if self.maas_fl2va_input:
                    self._expire_maas_pending(time.monotonic())
                completed = [t for t in self.tasks.values() if t.status == "completed" and t.infer_time]
                if completed:
                    infer_times = [t.infer_time for t in completed]
                    avg_infer_time = sum(infer_times) / len(infer_times)
                    min_infer_time = min(infer_times)
                    max_infer_time = max(infer_times)
                else:
                    avg_infer_time = min_infer_time = max_infer_time = 0

                total_tasks = len(self.tasks)
                running_tasks = sum(1 for t in self.tasks.values() if t.status == "running")
                pending_tasks = sum(1 for t in self.tasks.values() if t.status == "pending")
                failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")

            return jsonify({
                "model": self.default_config.get("model"),
                "world_size": self.world_size,
                "tasks": {
                    "total": total_tasks,
                    "completed": len(completed),
                    "running": running_tasks,
                    "pending": pending_tasks,
                    "failed": failed_tasks
                },
                "performance": {
                    "avg_infer_time": round(avg_infer_time, 2),
                    "min_infer_time": round(min_infer_time, 2),
                    "max_infer_time": round(max_infer_time, 2)
                }
            })

    def _parse_string_input(self, external_input: str, parsed: Dict[str, Any]) -> bool:
        external_input = external_input.strip()
        if not external_input:
            raise ValueError("input string cannot be empty")
        
        if external_input.startswith('{') or external_input.startswith('['):
            try:
                return False
            except json.JSONDecodeError:
                parsed['prompt'] = external_input
                return True
        else:
            parsed['prompt'] = external_input
            return True

    def _parse_image_data(self, image_data: Any, parsed: Dict[str, Any]) -> None:
        if isinstance(image_data, list) and len(image_data) > 0:
            parsed['i2v_image_path'] = image_data[0]
        elif isinstance(image_data, str):
            parsed['i2v_image_path'] = image_data

    def _extract_basic_params(self, input_dict: Dict[str, Any], parsed: Dict[str, Any]) -> None:
        for key in ['prompt', 'negative_prompt']:
            if key in input_dict:
                parsed[key] = input_dict[key]
        
        if 'image' in input_dict:
            self._parse_image_data(input_dict['image'], parsed)
        
        for key in ['width', 'height', 'frames', 'seed', 'num_inference_steps',
                    'i2v_image_path', 'save_fps', 'guidance_scale', 'guidance_scale_2',
                    # MiniMax-H3 specific
                    'model', 'task_type', 'resolution', 'adopt_sr', 'enable_audio',
                    'first_frame', 'last_frame', 'ref_path', 'audio_save_path',
                    'attention_sparse', 'duration', 'aspect_ratio', 'save_path']:
            if key in input_dict:
                parsed[key] = input_dict[key]

    def _parse_size_param(self, size: str, parsed: Dict[str, Any]) -> None:
        if isinstance(size, str) and '*' in size:
            try:
                w, h = map(int, size.split('*'))
                parsed['width'] = w
                parsed['height'] = h
            except ValueError:
                pass
        elif isinstance(size, str):
            size_lower = size.lower()
            size_map = {'480': (832, 480), '720': (1280, 720), '1080': (1920, 1080), '2k': (1920, 1080)}
            for key, (w, h) in size_map.items():
                if key in size_lower:
                    parsed['width'], parsed['height'] = w, h
                    break

    def _parse_parameters(self, params: Dict[str, Any], parsed: Dict[str, Any]) -> None:
        if 'size' in params:
            self._parse_size_param(params['size'], parsed)
        
        for key in ['duration', 'seed', 'num_inference_steps']:
            if key in params:
                parsed[key] = params[key]
        
        if 'fps' in params:
            parsed['save_fps'] = params['fps']

    def _detect_task_type(self, external_input: Dict[str, Any], parsed: Dict[str, Any]) -> None:
        # Don't overwrite task_type if user explicitly set it
        if 'task_type' in parsed:
            return
        if 'model' not in external_input:
            return
        model_name = external_input['model']
        for alias, info in MODEL_REGISTRY.items():
            if model_name in [alias, info['model'], info.get('model', '')]:
                parsed['task_type'] = info['task_type']
                break

    def _parse_dict_input(self, external_input: Dict[str, Any], parsed: Dict[str, Any]) -> None:
        if 'input' in external_input and isinstance(external_input['input'], dict):
            inner_input = external_input['input']
            self._extract_basic_params(inner_input, parsed)
        else:
            self._extract_basic_params(external_input, parsed)
        
        if 'parameters' in external_input and isinstance(external_input['parameters'], dict):
            self._parse_parameters(external_input['parameters'], parsed)
        
        self._detect_task_type(external_input, parsed)

    def _parse_external_input(self, external_input: Any) -> Dict[str, Any]:
        parsed = {}
        
        if external_input is None:
            raise ValueError("input cannot be None")
        
        if isinstance(external_input, str):
            if self._parse_string_input(external_input, parsed):
                return parsed
            try:
                external_input = json.loads(external_input)
            except json.JSONDecodeError:
                return parsed
        
        if isinstance(external_input, dict):
            self._parse_dict_input(external_input, parsed)
        
        if 'prompt' not in parsed:
            raise ValueError("Could not extract prompt from input")
        
        return parsed

    def _setup_internal_task_routes(self):
        @self.app.route('/v1/tasks', methods=['POST'])
        def internal_create_task():
            try:
                if not self.is_initialized:
                    return jsonify({"error": "Service not initialized"}), 503
                
                if self.shutdown:
                    return jsonify({"error": "Service is shutting down"}), 503
                
                if self.maas_health and self._maas_readiness()["state"] == "degraded":
                    return jsonify({"error": "worker_unavailable"}), 503

                if self.maas_fl2va_input:
                    task_id, created_at_ms, validated_data = self._read_maas_submission(internal=True)
                    return self._submit_maas_task(task_id, validated_data, created_at_ms)
                else:
                    data = request.get_data()
                    if not data:
                        return jsonify({"error": "No JSON data provided"}), 400
                    data = json.loads(data)
                    task_id = data.get('id')
                    if not task_id:
                        return jsonify({"error": "Missing required field: id"}), 400

                    external_input = data.get('input')
                    created_at_ms = data.get('created_at')

                    try:
                        parsed_params = self._parse_external_input(external_input)
                    except ValueError as e:
                        logger.error(f"[Rank 0] Failed to parse input for task {task_id}: {e}")
                        return jsonify({
                            "id": task_id,
                            "status": "failed",
                            "output": "",
                            "error": str(e)
                        }), 400


                with self.tasks_lock:
                    if task_id in self.tasks:
                        existing = self.tasks[task_id]
                        return jsonify(existing.to_platform_dict()), 200
                
                if not self.maas_fl2va_input:
                    is_valid, result = self._validate_generate_params(parsed_params)
                    if not is_valid:
                        error_msg = "; ".join(result.get("errors", []))
                        logger.error(f"[Rank 0] Validation failed for task {task_id}: {error_msg}")
                        validated_data = None
                        task = AsyncTask(
                            task_id=task_id,
                            task_type="video_generation",
                            params=validated_data,
                            created_at_ms=created_at_ms
                        )
                        task.output = error_msg
                        task.status = "failed"
                        self.tasks[task_id] = task
                        return jsonify({
                            "id": task_id,
                            "status": "failed",
                            "output": error_msg,
                            "error": error_msg
                        }), 200

                    validated_data = result["data"]


                task = AsyncTask(
                    task_id=task_id,
                    task_type="video_generation",
                    params=validated_data,
                    created_at_ms=created_at_ms
                )
                
                with self.tasks_lock:
                    self.tasks[task_id] = task
                
                save_path = validated_data.get('save_path')
                if not save_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"/tmp/generated_videos/video_{task_id[:8]}_{timestamp}.mp4"
                
                try:
                    self.task_queue.put({
                        'task_id': task_id,
                        'data': validated_data,
                        'save_path': save_path
                    }, block=False)
                except queue.Full:
                    with self.tasks_lock:
                        del self.tasks[task_id]
                    return jsonify({
                        "id": task_id,
                        "status": "failed",
                        "output": "",
                        "error": "Task queue is full"
                    }), 503
                
                logger.info(f"[Rank 0] Internal task {task_id} created and queued")
                return jsonify(task.to_platform_dict()), 200
                
            except InvalidInput as e:
                return jsonify({"error": "invalid_input", "field": e.field, "message": str(e)}), e.status
            except Exception as e:
                logger.error(f"[Rank 0] Failed to create internal task: {str(e)}", exc_info=True)
                return jsonify({"error": str(e)}), 500

    def _setup_internal_query_routes(self):
        @self.app.route('/v1/tasks', methods=['GET'])
        def internal_list_tasks():
            try:
                with self.tasks_lock:
                    if self.maas_fl2va_input:
                        now = time.monotonic()
                        self._expire_maas_pending(now)
                        tasks_list = [self._maas_task_snapshot(task, now) for task in self.tasks.values()]
                    else:
                        tasks_list = [task.to_platform_dict() for task in self.tasks.values()]
                return jsonify({"tasks": tasks_list}), 200
            except Exception as e:
                logger.error(f"[Rank 0] Failed to list tasks: {str(e)}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @self.app.route('/v1/tasks/<task_id>', methods=['GET'])
        def internal_get_task(task_id):
            try:
                with self.tasks_lock:
                    task = self.tasks.get(task_id)
                    if task is not None and self.maas_fl2va_input:
                        now = time.monotonic()
                        self._expire_maas_pending(now)
                        return jsonify(self._maas_task_snapshot(task, now)), 200
                
                if not task:
                    return jsonify({"error": f"Task {task_id} not found"}), 404

                result = task.to_platform_dict()
                return jsonify(result), 200
            except Exception as e:
                logger.error(f"[Rank 0] Failed to get task: {str(e)}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @self.app.route('/v1/tasks/<task_id>', methods=['DELETE'])
        def internal_delete_task(task_id):
            if self.maas_fl2va_input:
                return self._cancel_maas_task(task_id)
            try:
                with self.tasks_lock:
                    task = self.tasks.get(task_id)
                    
                    if not task:
                        return jsonify({"error": f"Task {task_id} not found"}), 404
                    
                    if task.status in ["running"]:
                        task.status = "cancelled"
                        task.completed_at = datetime.now()
                        task.error = "Task cancelled by user"
                    elif task.status in ["pending", "queued"]:
                        task.status = "cancelled"
                        task.completed_at = datetime.now()
                        task.error = "Task cancelled before execution"
                    else:
                        del self.tasks[task_id]
                        logger.info(f"[Rank 0] Task {task_id} deleted")
                        return jsonify({"id": task_id}), 200
                
                logger.info(f"[Rank 0] Task {task_id} cancelled")
                return jsonify({"id": task_id}), 200
                
            except Exception as e:
                logger.error(f"[Rank 0] Failed to delete task: {str(e)}", exc_info=True)
                return jsonify({"error": str(e)}), 500

    def _update_basic_config(self, config: Dict[str, Any], data: Dict[str, Any], prompt: Optional[str]) -> None:
        update_config = {
            'negative_prompt': data.get('negative_prompt', ""),
            'height': data.get('height', 480),
            'width': data.get('width', 832),
            'frames': data.get('frames', 81),
            'num_inference_steps': data.get('num_inference_steps', 40),
            'seed': data.get('seed', 42),
        }
        if prompt:
            update_config['prompt'] = prompt
        config.update(update_config)

    def _update_guidance_config(self, config: Dict[str, Any], data: Dict[str, Any]) -> None:
        if data.get('guidance_scale') is not None:
            config['guidance_scale'] = data['guidance_scale']
        if data.get('guidance_scale_2') is not None:
            config['guidance_scale_2'] = data['guidance_scale_2']

    def _update_path_config(self, config: Dict[str, Any], data: Dict[str, Any], save_path: Optional[str]) -> None:
        if save_path:
            config['save_path'] = save_path
        elif data.get('save_path'):
            config['save_path'] = data['save_path']
        if data.get('i2v_image_path'):
            config['i2v_image_path'] = data['i2v_image_path']

    def _apply_ten_second_config(self, config: Dict[str, Any]) -> None:
        config['ten_second'] = True
        config['fsdp'] = 'text_encoder'
        config['inf_vram_blocks_num'] = 1
        config['save_memory'] = True
        config['vae_lightning'] = 'decoder'
        config['atten'] = 'laser'
        config['matmul_quant'] = 'int8'
        config['rope_fused'] = True

    def _update_ten_second_config(self, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        duration = data.get('duration', 5)
        if duration == 10:
            data = dict(data)
            data['ten_second'] = True
        ten_second = data.get('ten_second', False)
        if not ten_second:
            return data
        self._apply_ten_second_config(config)
        task_type = config.get('task_type', 't2v')
        sp = config.get('sp', 1)
        logger.info(f"[Rank {self.rank}] 10s mode with {sp} GPU(s)")
        if task_type == 'i2v':
            config['joint'] = False
        else:
            config['x_model_path'] = TEN_SECOND_CONFIG['ten_second_model_path']
            config['x_model_path_2'] = TEN_SECOND_CONFIG['ten_second_model_path_2']
        return data

    def _update_resolution_config(self, config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        width = data.get('width')
        height = data.get('height')
        if width and height and ((width == 1920 and height == 1080) or (width == 1080 and height == 1920)):
            data = dict(data)
            data['resolution'] = 1080
            logger.info(f"[Rank {self.rank}] Detected 1080p from {width}x{height}")
        if data.get('adopt_sr', False):
            config['adopt_sr'] = True
            config['resolution'] = data.get('resolution', 1080)
        resolution = data.get('resolution', 480)
        if resolution == 1080:
            config['adopt_sr'] = True
            config['resolution'] = 1080
            config['seedvr2_model_dir'] = data.get('seedvr2_model_dir', TEN_SECOND_CONFIG['seedvr2_model_dir'])
            config['seedvr2_model_name'] = data.get('seedvr2_model_name', TEN_SECOND_CONFIG['seedvr2_model_name'])
        return data

    def _update_fps_config(self, config: Dict[str, Any], data: Dict[str, Any]) -> None:
        save_fps = data.get('save_fps', 16)
        config['save_fps'] = save_fps
        if save_fps > 16 or data.get('frame_interpolation', False):
            config['frame_interpolation'] = True
            config['frame_model_path'] = data.get('frame_model_path', TEN_SECOND_CONFIG['frame_model_path'])

    def _config_to_arg_list(self, config: Dict[str, Any]) -> List[str]:
        from x_diffusers.adaptor.h3_cuda_options import BOOL_OPTIONS
        from x_diffusers.adaptor.h3_features import normalize_mapping
        config = normalize_mapping(config)
        excluded_keys = {'external_base_url'}
        arg_list = []
        for key, value in config.items():
            if key in excluded_keys or value is None or value == '':
                continue
            if key == 'prompt' and getattr(self, 'maas_fl2va_input', False):
                arg_list.append(f'--prompt={value}')
            elif isinstance(value, bool):
                if value:
                    arg_list.append(f'--{key}')
                elif key in BOOL_OPTIONS or key.startswith('h3_opt_'):
                    arg_list.append(f'--no-{key}')
            elif isinstance(value, list):
                arg_list.append(f'--{key}')
                arg_list.extend([str(v) for v in value])
            else:
                arg_list.extend([f'--{key}', str(value)])
        return arg_list

    def _build_args(self, data: Dict[str, Any], save_path: Optional[str] = None, require_prompt: bool = True):
        config = self.default_config.copy()
        prompt = data.get('prompt')
        if require_prompt and not prompt:
            raise ValueError("prompt is required in request")

        # --- MiniMax-H3: skip Wan2.2-specific config updates ---
        if "minimax-h3" in config.get("model", "").lower():
            steps = data.get('num_inference_steps', config.get('num_inference_steps', 50))
            self._update_basic_config(config, data, prompt)
            config['num_inference_steps'] = steps
            self._update_path_config(config, data, save_path)
            # MiniMax-H3 specific params from request
            if data.get('task_type'):
                config['task_type'] = data['task_type']
            if self.maas:
                config.update(keyframe_image='', keyframe_last_image='')
                config.pop('reference_paths', None)
            # Resolve input files (base64/http/local → local path) and map to argparse names
            if data.get('first_frame'):
                config['keyframe_image'] = _resolve_input_file(data['first_frame'])
            if data.get('last_frame'):
                config['keyframe_last_image'] = _resolve_input_file(data['last_frame'])
            if data.get('ref_path'):
                _rp = data['ref_path']
                _rp_list = [_rp] if isinstance(_rp, str) else _rp
                config['reference_paths'] = [_resolve_input_file(p) for p in _rp_list]
            if data.get('audio_save_path'):
                config['audio_save_path'] = data['audio_save_path']
            if data.get('attention_sparse'):
                config['attention_sparse'] = True
            # Duration → frames (fps fixed at 24)
            config['save_fps'] = 24
            _duration = data.get('duration', 5)
            config['frames'] = _duration * 24

            # Resolution + aspect_ratio -> width/height
            resolution = data.get('resolution', config.get('resolution', '768p'))
            aspect_ratio = data.get('aspect_ratio', '16:9')
            if resolution in MMH3_RESOLUTION_MAP and aspect_ratio in MMH3_RESOLUTION_MAP[resolution]:
                config['width'], config['height'] = MMH3_RESOLUTION_MAP[resolution][aspect_ratio]
            else:
                config['width'], config['height'] = MMH3_RESOLUTION_MAP['768p']['16:9']
            if self.maas and data.get('_maas_size'):
                config['width'], config['height'] = data['_maas_size']

            # Pop Wan2.2-specific keys that MiniMax-H3 doesn't use
            # (they leak from the Wan2.2 default config and cause argparse errors)
            # Also pop server-level params not in argparse:
            #   resolution, enable_audio, adopt_sr — not in argparse, set manually
            #   seedvr2_model_dir/name — in argparse as --seedvr2-model-dir (hyphens),
            #     but _config_to_arg_list produces --seedvr2_model_dir (underscores).
            #     Set manually to avoid hyphen/underscore ambiguity.
            _mmh3_adopt_sr = data.get('adopt_sr', config.get('adopt_sr', False))
            # 1080p auto-enables super-resolution (generate 768p → upscale)
            if resolution == '1080p':
                _mmh3_adopt_sr = True
            _mmh3_seedvr2_dir = config.get('seedvr2_model_dir')
            _mmh3_seedvr2_name = config.get('seedvr2_model_name')
            for _k in ('resolution', 'enable_audio', 'adopt_sr',
                        'first_frame', 'last_frame', 'ref_path', 'duration',
                        'ten_second_model_id_t2v', 'ten_second_model_path',
                        'ten_second_model_path_2', 'pusa_lora', 'pusa_lora2',
                        'joint_model_path', 'x_model_path', 'x_model_path_2',
                        'seedvr2_model_dir', 'seedvr2_model_name', 'frame_model_path',
                        'joint', 'x', 'cfg_parallel_size', 'service_mode'):
                config.pop(_k, None)

            arg_list = self._config_to_arg_list(config)
            old_argv = sys.argv
            sys.argv = ['video_server.py'] + arg_list
            try:
                args = parse_args(bootstrap_environment=False)
            finally:
                sys.argv = old_argv
            # Server-level params not in argparse — set manually after parse_args
            args.enable_audio = data.get('enable_audio', True)
            if self.maas:
                args._maas_watermark = data.get('_maas_watermark', False)
            args.adopt_sr = _mmh3_adopt_sr
            args.seedvr2_model_dir = _mmh3_seedvr2_dir
            args.seedvr2_model_name = _mmh3_seedvr2_name
            if self.default_config.get('h3_refiner', False):
                from x_diffusers.adaptor.h3_refiner_service import configure_args
                configure_args(self.default_config, data, args)
            if self.default_config.get('h3_attention', 'legacy') != 'legacy':
                from x_diffusers.adaptor.h3_attention_service import configure_args as configure_sparse_args
                configure_sparse_args(self.default_config, data, args)
            if args.task_type == "ref2va" and data.get("ref_path"):
                reference_conditioning(args)
            return args

        # --- Wan2.2 original path ---
        self._update_basic_config(config, data, prompt)
        self._update_guidance_config(config, data)
        self._update_path_config(config, data, save_path)
        data = self._update_ten_second_config(config, data)
        data = self._update_resolution_config(config, data)
        self._update_fps_config(config, data)
        
        if not config.get('adopt_sr', False):
            config.pop('resolution', None)
        
        arg_list = self._config_to_arg_list(config)
        old_argv = sys.argv
        sys.argv = ['video_server.py'] + arg_list
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        return args

    def _get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                w, h = map(int, result.stdout.strip().split('x'))
                return w, h
        except Exception as e:
            logger.warning(f"Failed to get video resolution: {e}")
        return 0, 0

    def _resize_video_to_target(self, video_path: str, target_width: int, target_height: int) -> bool:
        w, h = self._get_video_resolution(video_path)
        if w == target_width and h == target_height:
            return True
        if w == 0 or h == 0:
            logger.error(f"Cannot get video resolution for {video_path}")
            return False
        
        temp_path = video_path + '.temp.mp4'
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2',
            '-q:v', '2', temp_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(temp_path):
                os.replace(temp_path, video_path)
                logger.info(f"Resized video from {w}x{h} to {target_width}x{target_height}")
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"Failed to resize video: {e}")
        return False

    def _prepare_task(self, task_id: str) -> bool:
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.warning(f"[Rank 0] Task {task_id} not found, skipping")
                return False
            if task.status == "cancelled":
                logger.info(f"[Rank 0] Task {task_id} was cancelled, skipping")
                return False
            task.status = "running"
            task.started_at = datetime.now()
            return True

    def _set_guidance_scale(self, args):
        # MiniMax-H3 is guidance-distilled — no guidance scale needed
        if "minimax-h3" in args.model.lower():
            return
        if args.guidance_scale is None or args.guidance_scale_2 is None:
            args.guidance_scale, args.guidance_scale_2 = GUIDANCE_SCALE_MAP[args.model]
            logger.info(f"Guidance scale set to: {args.guidance_scale}, {args.guidance_scale_2}")

    def _save_video(self, output, save_path: str, args):
        # --- MiniMax-H3: output is dict with videos + audio ---
        if "minimax-h3" in args.model.lower():
            return self._save_minimax_h3_output(output, save_path, args)

        if not getattr(args, 'adopt_sr', False):
            export_to_video(output, save_path, quality=8, fps=args.save_fps)
        else:
            resolution = getattr(args, 'resolution', 1080)
            if resolution == 1080 and args.task_type == "t2v":
                self._resize_video_to_target(save_path, 1920, 1080)

    @staticmethod
    def _flatten_frames(videos):
        """Recursively flatten nested list/tuple of frames into a flat iterator."""
        for item in videos:
            if isinstance(item, (list, tuple)):
                yield from VideoGenerationService._flatten_frames(item)
            else:
                yield item

    @staticmethod
    def _frame_to_uint8_hwc(frame):
        """Convert a single frame (Image/Tensor/ndarray) to uint8 HWC ndarray."""
        if isinstance(frame, Image.Image):
            return np.array(frame)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[0] in (3, 4):
                frame = frame.transpose(1, 2, 0)
            if frame.dtype == np.uint8:
                return frame
            frame = np.clip(frame, 0, 1) * 255
            return frame.astype(np.uint8)
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    def _save_mmh3_video_ffmpeg(self, frames, audio_path, save_path, args):
        """Save video frames via ffmpeg subprocess pipe."""
        first_frame = self._frame_to_uint8_hwc(frames[0])
        height, width = first_frame.shape[0], first_frame.shape[1]
        fps = getattr(args, "save_fps", None) or 24

        ffmpeg_bin = os.getenv("FFMPEG_PATH", "ffmpeg")
        cmd = [ffmpeg_bin, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
        if audio_path:
            cmd += ["-i", audio_path]
        preset = "veryfast" if getattr(args, "h3_cuda_precision", "none") in ("fp8-bf16", "fp8-sage") else "fast"
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", "10", "-pix_fmt", "yuv420p"]
        if audio_path:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += ["-loglevel", "error", save_path]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        convert = self._frame_to_uint8_hwc
        if getattr(args, '_maas_watermark', False):
            from maas_artifacts import watermark_converter
            convert = watermark_converter(convert, width, height)
        for frame in frames:
            proc.stdin.write(convert(frame).tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
        if audio_path:
            logger.info(f"Saved MiniMax-H3 video+audio to {save_path}")
        else:
            logger.info(f"Saved MiniMax-H3 video to {save_path}")

    def _save_mmh3_video_fallback(self, videos, save_path):
        """Fallback: save video via export_to_video when ffmpeg fails."""
        video_frames = []
        for frame in self._flatten_frames(videos):
            if isinstance(frame, Image.Image):
                video_frames.append(frame)
            elif isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    frame = frame.transpose(1, 2, 0)
                video_frames.append(frame)
            elif isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    frame = frame.transpose(1, 2, 0)
                video_frames.append(frame)
        export_to_video(video_frames, save_path, quality=8, fps=24)
        logger.info(f"Saved MiniMax-H3 video to {save_path} (fallback)")

    def _save_minimax_h3_output(self, output, save_path: str, args):
        """Save MiniMax-H3 video + audio into a single MP4 with embedded audio."""
        if getattr(args, 'h3_refiner', False):
            if output.get('saved_path') != str(Path(save_path).resolve()):
                raise RuntimeError('H3 Refiner did not finish the requested output')
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        videos = output.get("videos")
        if getattr(args, "enable_audio", True):
            audio = output.get("audio")
            sampling_rate = output.get("sampling_rate")
        else:
            audio = None
            sampling_rate = None

        trim_padding = getattr(args, "frames", None) == H3_BASE_MAX_DURATION_S * 24
        if trim_padding and audio is not None and sampling_rate is not None:
            audio = audio[..., :H3_BASE_MAX_DURATION_S * sampling_rate]

        # Save audio to temporary WAV
        audio_path = None
        if audio is not None and sampling_rate is not None:
            from x_base.vae_parallelism.audio_utils import save_audio_to_wav, cleanup_audio
            audio_path = save_audio_to_wav(
                audio, sampling_rate,
                audio_path=getattr(args, "audio_save_path", "") or None,
                save_path=save_path,
            )
        else:
            from x_base.vae_parallelism.audio_utils import cleanup_audio

        frames = []
        try:
            if videos is None:
                return
            frames = list(self._flatten_frames(videos))
            if trim_padding:
                frames = frames[:args.frames]
            if not frames:
                logger.warning("No video frames to save")
                return
            self._save_mmh3_video_ffmpeg(frames, audio_path, save_path, args)
        except Exception as e:
            if self.maas_fl2va_input:
                raise
            logger.warning(f"ffmpeg save failed ({e}), falling back to export_to_video")
            self._save_mmh3_video_fallback(frames if trim_padding else videos, save_path)
        finally:
            if audio_path:
                cleanup_audio(audio_path)

    def _complete_task(self, task_id: str, save_path: str, infer_time: float):
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if not task or task.status != "running":
                return
            
            task.status = "completed"
            task.progress = 100.0
            result_data = {
                "video_path": save_path,
                "infer_time": round(infer_time, 2)
            }
            
            external_base_url = self.default_config.get("external_base_url")
            if external_base_url:
                download_url = f"{external_base_url.rstrip('/')}/download/{task_id}"
                result_data["download_url"] = download_url

            obs_local_path = os.environ.get("OBS_LOCAL_PATH")
            if obs_local_path:
                from obs_client import ObsStorageClient

                file_util.copy_file(save_path, obs_local_path)
                obs_client = ObsStorageClient()
                filename = Path(save_path).name
                download_url = obs_client.generate_url(filename)
                obs_client.close()
                result_data["download_url"] = download_url
                
            task.result = result_data
            task.output = json.dumps(result_data)
            task.completed_at = datetime.now()
            task.infer_time = infer_time

    def _handle_task_interruption(self, task_id: str, error: str):
        logger.info(f"[Rank 0] Task {task_id} interrupted: {error}")
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "cancelled"
                task.error = error
                task.completed_at = datetime.now()

        self._clear_cache()
        logger.info(f"[Rank 0] NPU resources released for cancelled task {task_id}")

    def _handle_task_failure(self, task_id: str, error: str):
        logger.error(f"[Rank 0] Task {task_id} failed: {error}", exc_info=True)
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "failed"
                task.error = error
                task.completed_at = datetime.now()

    def _clear_cache(self):
        gc.collect()
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()

    def _publish_maas_task(self, task, output, save_path, args, infer_time):
        with self.tasks_lock:
            task.timings_s['inference'] = round(infer_time, 3)
        started = time.monotonic()
        publish_options = {}
        if self.maas:
            frames = args.frames if args.frames == 360 else ((args.frames-5+16)//17)*17+5
            publish_options['expected'] = (args.width, args.height, frames)
        publish_video(save_path, lambda path: self._save_video(output, path, args),
                      require_audio=args.enable_audio, **publish_options)
        encoded = time.monotonic()
        with self.tasks_lock:
            task.timings_s['encode'] = round(encoded - started, 3)
        result = {'video_path': save_path, 'infer_time': round(infer_time, 2)}
        if self.maas:
            result.update(width=args.width, height=args.height, resolution='768p')
        base_url = self.default_config.get('external_base_url')
        if base_url:
            result['download_url'] = f"{base_url.rstrip('/')}/download/{task.task_id}"
        if obs_directory := os.environ.get('OBS_LOCAL_PATH'):
            try:
                result['download_url'] = publish_obs(save_path, obs_directory)
            except StorageError:
                try:
                    Path(save_path).unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not clean unpublished output for %s", task.task_id)
                raise
        with self.tasks_lock:
            task.timings_s['delivery'] = round(time.monotonic() - encoded, 3)
            task.finish('completed', result=result)
            timings = dict(task.timings_s)
        logger.info("MaaS task %s completed timings_s=%s", task.task_id, timings)

    def _process_task(self, task_id: str, data: Dict[str, Any], save_path: str, *, maas_task=None):
        try:
            if maas_task is None and not self._prepare_task(task_id):
                return

            args = self._build_args(data, save_path)
            self._set_guidance_scale(args)

            if self.maas_health and not self.maas_health.model_loaded:
                self.maas_health = self.maas_health.advance('loading')
            self.inference_manager.initialize_pipe(args)
            if self.maas_health:
                self.maas_health = self.maas_health.advance('generating', model_loaded=True)
            
            infer_info.update_info(args)
            infer_info.save_path = save_path
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.cuda.synchronize()
            clock = time.monotonic if maas_task is not None else time.time
            start_time = clock()

            output = self.inference_manager.infer_single(args, task_id=task_id, task_manager=self)

            torch.cuda.synchronize()
            infer_time = clock() - start_time

            if self.maas_health:
                self.maas_health = self.maas_health.advance('saving')
            if maas_task is not None:
                self._publish_maas_task(maas_task, output, save_path, args, infer_time)
                output = None
                from x_diffusers.adaptor.h3_memory import release_request_memory
                release_request_memory()
                return True
            self._save_video(output, save_path, args)
            if "minimax-h3" in args.model.lower():
                output = None
                from x_diffusers.adaptor.h3_memory import release_request_memory
                release_request_memory()
            self._complete_task(task_id, save_path, infer_time)

            logger.info(f"[Rank 0] Task {task_id} completed in {infer_time:.2f}s")

        except Exception as e:
            if maas_task is not None:
                phase = self.maas_health.phase
                if phase == 'saving':
                    failure_stage = 'storage' if isinstance(e, StorageError) else 'encode'
                    code = f'{failure_stage}_failed'
                else:
                    failure_stage, code = phase, 'worker_lost'
                with self.tasks_lock:
                    maas_task.finish('failed', code=code, stage=failure_stage)
                logger.exception("MaaS task %s failed (%s)", task_id, code)
                if phase != 'saving':
                    self.maas_health = self.maas_health.advance(phase, failed=True)
                    self._fail_maas_jobs(phase)
                return False
            if isinstance(e, InterruptedError):
                self._handle_task_interruption(task_id, str(e))
                return
            self._handle_task_failure(task_id, str(e))
            if self.default_config.get('h3_refiner', False):
                if (getattr(self.inference_manager, '_h3_refiner_failed', False) or
                        getattr(getattr(self.inference_manager, 'sr_pipe', None), 'failed', False)):
                    # This is the GPU task thread: exit the owned worker process
                    # so torchrun can tear down a failed collective group.
                    self.is_initialized = False
                    self.shutdown = True
                    os._exit(1)
        finally:
            _cleanup_temp_files()
            if self.maas:
                from maas_media import cleanup
                cleanup(data)

    def _worker_loop(self):
        logger.info(f"[Rank {self.rank}] Worker ready, entering request loop...")

        while True:
            try:
                command = broadcast_int(CMD_HEARTBEAT, src=0)

                if command == CMD_HEARTBEAT:
                    continue

                if command == CMD_SHUTDOWN:
                    logger.info(f"[Rank {self.rank}] Received shutdown command, exiting worker loop")
                    break

                if command == CMD_EXECUTE:
                    broadcast_data = broadcast_dict({}, src=0)
                    data = broadcast_data['data']
                    save_path = broadcast_data['save_path']
                    
                    dist.barrier()
                    logger.info(f"[Rank {self.rank}] All ranks ready, starting inference")

                    args = self._build_args(data, save_path)

                    # MiniMax-H3 is guidance-distilled — skip guidance scale
                    if "minimax-h3" not in args.model.lower():
                        if args.guidance_scale is None or args.guidance_scale_2 is None:
                            args.guidance_scale, args.guidance_scale_2 = GUIDANCE_SCALE_MAP[args.model]

                    self.inference_manager.initialize_pipe(args)
                    infer_info.update_info(args)
                    infer_info.save_path = save_path
                    self.inference_manager.infer_single(args)
                    if "minimax-h3" in args.model.lower():
                        from x_diffusers.adaptor.h3_memory import release_request_memory
                        release_request_memory()

            except Exception as e:
                logger.error(f"[Rank {self.rank}] Worker loop error: {str(e)}", exc_info=True)
                if self.maas_health:
                    raise
                if self.default_config.get('h3_refiner', False):
                    if (getattr(self.inference_manager, '_h3_refiner_failed', False) or
                            getattr(getattr(self.inference_manager, 'sr_pipe', None), 'failed', False)):
                        raise
                time.sleep(1)
            finally:
                if self.maas_fl2va_input:
                    _cleanup_temp_files()

    def _task_processor_thread(self):
        heartbeat_interval = 30
        last_heartbeat_time = 0 if self.maas_health else time.time()
        if self.maas_health:
            self.maas_health = self.maas_health.advance('idle')
        
        while not self.shutdown:
            try:
                maas_task = None
                try:
                    if self.maas_health:
                        task_info, maas_task = self._claim_maas_task()
                    else:
                        task_info = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    current_time = time.time()
                    if self.world_size > 1 and (current_time - last_heartbeat_time) >= heartbeat_interval:
                        broadcast_int(CMD_HEARTBEAT, src=0)
                        last_heartbeat_time = current_time
                        if self.maas_health:
                            self.maas_health = self.maas_health.advance('idle', group_confirmed=True)
                    continue

                if self.maas_health:
                    self.maas_health = self.maas_health.advance('dispatching')
                if self.world_size > 1:
                    broadcast_int(CMD_EXECUTE, src=0)
                    broadcast_dict({'data': task_info['data'], 'save_path': task_info['save_path']}, src=0)
                    dist.barrier()
                    logger.info(f"[Rank 0] All workers ready, starting task execution")

                if self.maas_health:
                    self.maas_health = self.maas_health.advance('dispatching', group_confirmed=True)
                completed = self._process_task(
                    task_info['task_id'],
                    task_info['data'],
                    task_info['save_path'], maas_task=maas_task,
                )

                if self.maas_health:
                    if self.maas_health.failed:
                        return
                    if completed and not self.maas_health.warmed:
                        self.maas_health = self.maas_health.advance('confirming_workers')
                        if self.world_size > 1:
                            broadcast_int(CMD_HEARTBEAT, src=0)
                        self.maas_health = self.maas_health.advance('idle', warmed=True)
                    else:
                        self.maas_health = self.maas_health.advance('idle')
                last_heartbeat_time = time.time()

                if "minimax-h3" not in self.default_config.get('model', '').lower():
                    gc.collect()
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"[Rank 0] Task processor error: {str(e)}", exc_info=True)
                if self.maas_health:
                    self.maas_health = self.maas_health.advance(self.maas_health.phase, failed=True)
                    self._fail_maas_jobs(self.maas_health.phase)
                    return
                time.sleep(1)

    def _cleanup_thread(self):
        while not self.shutdown:
            try:
                time.sleep(self.cleanup_interval)
                if not self.shutdown:
                    self._cleanup_completed_tasks()
            except Exception as e:
                logger.error(f"[Rank 0] Cleanup thread error: {str(e)}", exc_info=True)

    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        if self.world_size > 1 and self.rank != 0:
            self._worker_loop()
            return

        processor_thread = threading.Thread(target=self._task_processor_thread, daemon=True)
        processor_thread.start()
        logger.info(f"[Rank 0] Task processor thread started")

        cleanup_thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        cleanup_thread.start()
        logger.info(f"[Rank 0] Cleanup thread started (interval: {self.cleanup_interval}s)")

        logger.info(f"[Rank 0] Starting VideoGenerationService on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

    def _initialize_model(self):
        if not self.is_initialized or self.inference_manager is None:
            logger.warning(f"[Rank {self.rank}] Service not initialized, skip model loading")
            return
        
        if self.inference_manager.pipe is not None:
            logger.info(f"[Rank {self.rank}] Model already loaded")
            return
        
        init_args = self._build_args({}, require_prompt=False)
        
        # MiniMax-H3 is guidance-distilled — skip guidance scale
        if "minimax-h3" not in init_args.model.lower():
            if init_args.guidance_scale is None or init_args.guidance_scale_2 is None:
                init_args.guidance_scale, init_args.guidance_scale_2 = GUIDANCE_SCALE_MAP[init_args.model]
        
        logger.info(f"[Rank {self.rank}] Loading model...")
        self.inference_manager.initialize_pipe(init_args)
        logger.info(f"[Rank {self.rank}] Model loaded successfully")

    def warmup(self, warmup_steps: int = 10):
        if not self.is_initialized or self.inference_manager is None:
            logger.warning("[Rank 0] Service not initialized, skip warmup")
            return
        
        if self.rank != 0:
            return
        
        task_type = self.default_config.get('task_type', 't2v')
        ten_second = self.default_config.get('ten_second', False)
        duration = 10 if ten_second else 5
        
        logger.info(f"[Rank 0] Submitting warmup task (duration={duration}s, resolution=1080P, task_type={task_type})...")
        
        warmup_params = {
            "prompt": "warmup test",
            "width": 1920,
            "height": 1080,
            "frames": 16 * duration + 1,
            "num_inference_steps": warmup_steps,
            "seed": 0,
            "duration": duration,
            "guidance_scale": 3.5,
            "guidance_scale_2": 3.5
        }
        
        if task_type == 'i2v':
            warmup_image_path = "/tmp/warmup_input.png"
            warmup_image = Image.new('RGB', (1920, 1080), color=(128, 128, 128))
            warmup_image.save(warmup_image_path)
            warmup_params['i2v_image_path'] = warmup_image_path
            logger.info(f"[Rank 0] Created warmup image: {warmup_image_path}")
        
        task_id = str(uuid.uuid4())
        save_path = "/tmp/warmup_video.mp4"
        task = AsyncTask(
            task_id=task_id,
            task_type="warmup",
            params=warmup_params
        )
        
        with self.tasks_lock:
            self.tasks[task_id] = task
        
        try:
            self.task_queue.put({
                'task_id': task_id,
                'data': warmup_params,
                'save_path': save_path
            }, block=False)
            logger.info(f"[Rank 0] Warmup task {task_id} submitted")
        except queue.Full:
            with self.tasks_lock:
                del self.tasks[task_id]
            logger.warning("[Rank 0] Task queue full, skip warmup")


def _apply_ten_second_config_args(args) -> None:
    """Apply command line arguments to TEN_SECOND_CONFIG."""

    arg_to_config = ['ten_second_model_id_t2v', 'ten_second_model_path', 'ten_second_model_path_2', 'pusa_lora', 'pusa_lora2', 'joint_model_path', 
    'seedvr2_model_dir', 'seedvr2_model_name', 'x_model_path', 'x_model_path_2', 'frame_model_path']

    for arg_name in arg_to_config:
        arg_value = getattr(args, arg_name, None)
        if arg_value:
            TEN_SECOND_CONFIG[arg_name] = arg_value


def _apply_mmh3_config(service: VideoGenerationService, args, rank: int) -> None:
    """Apply MiniMax-H3 specific command line arguments."""
    from x_diffusers.adaptor.h3_cuda_options import STARTUP_OPTIONS
    for name in ('h3_vae_backend', 'h3_vae_model_path', 'h3_vae_trust_remote_code', 'h3_reference_image_short_edge'):
        value = getattr(args, name, None)
        if value is not None:
            service.default_config[name] = value
    for name in STARTUP_OPTIONS + ('h3_parallel_mode', 'h3_dit_tp', 'h3_tp_comm', 'h3_cp_comm',
                                   'h3_tp_comm_chunk_elements', 'h3_parallel_warmup_mib'):
        value = getattr(args, name, None)
        if value is not None:
            service.default_config[name] = value
    if getattr(args, 'h3_cuda_precision', None) is not None:
        service.default_config['h3_cuda_precision'] = args.h3_cuda_precision
    if getattr(args, 'text_encoder_tp', None):
        service.default_config['text_encoder_tp'] = args.text_encoder_tp
        logger.info(f"[Rank {rank}] Text encoder TP set to: {args.text_encoder_tp}")
    if getattr(args, 'no_phaa', False):
        service.default_config['phaa_num'] = None
        logger.info(f"[Rank {rank}] PHAA disabled")
    elif getattr(args, 'phaa_num', None):
        service.default_config['phaa_num'] = args.phaa_num
        logger.info(f"[Rank {rank}] PHAA num set to: {args.phaa_num}")
    service_mode = getattr(args, 'service_mode', None) or ''
    if service_mode:
        service.default_config['service_mode'] = service_mode
        logger.info(f"[Rank {rank}] Service mode set to: {service_mode}")
    if getattr(args, 'adaln_cache_file', None):
        service.default_config['adaln_cache_file'] = args.adaln_cache_file
        logger.info(f"[Rank {rank}] AdaLN cache file set to: {args.adaln_cache_file}")
    if getattr(args, 'transformer_path', None):
        service.default_config['transformer_path'] = args.transformer_path
        logger.info(f"[Rank {rank}] Transformer path set to: {args.transformer_path}")
    if getattr(args, 'vae_matmul_quant', None) is not None:
        service.default_config['vae_matmul_quant'] = args.vae_matmul_quant
        logger.info(f"[Rank {rank}] VAE matmul quant set to: {args.vae_matmul_quant}")
    if getattr(args, 'seedvr2_model_dir', None):
        service.default_config['seedvr2_model_dir'] = args.seedvr2_model_dir
        logger.info(f"[Rank {rank}] SeedVR2 model dir set to: {args.seedvr2_model_dir}")
    if getattr(args, 'seedvr2_model_name', None):
        service.default_config['seedvr2_model_name'] = args.seedvr2_model_name
        logger.info(f"[Rank {rank}] SeedVR2 model name set to: {args.seedvr2_model_name}")
    


def _apply_service_config(service: VideoGenerationService, args, rank: int) -> None:
    """Apply command line arguments to service configuration."""
    if args.model:
        model_info = MODEL_REGISTRY[args.model]
        service.default_config['model'] = model_info['model']
        service.default_config['pretrained_model_name_or_path'] = model_info['path']
        service.default_config['task_type'] = model_info['task_type']
        if "minimax-h3" in model_info['model'].lower():
            service.default_config.update(DEFAULT_MINIMAX_H3_CONFIG)
            service.default_config['model'] = model_info['model']
            service.default_config['pretrained_model_name_or_path'] = model_info['path']
            service.default_config['task_type'] = model_info['task_type']
        logger.info(f"[Rank {rank}] Model set to: {args.model}")

    if args.model_path:
        service.default_config['pretrained_model_name_or_path'] = args.model_path
        logger.info(f"[Rank {rank}] Model path set to: {args.model_path}")

    if args.dtype:
        service.default_config['dtype'] = args.dtype
        logger.info(f"[Rank {rank}] Dtype set to: {args.dtype}")

    if args.sp:
        service.default_config['sp'] = args.sp
        logger.info(f"[Rank {rank}] Sequence parallel set to: {args.sp}")

    if args.no_fsdp:
        service.default_config['fsdp'] = False
        logger.info(f"[Rank {rank}] FSDP disabled")

    # MiniMax-H3 specific args
    _apply_mmh3_config(service, args, rank)

    if args.ten_second:
        service.default_config['ten_second'] = True
        service.default_config['fsdp'] = 'all'
        service.default_config['inf_vram_blocks_num'] = 0
        logger.info(f"[Rank {rank}] 10-second mode enabled (fsdp=all, inf_vram_blocks_num=0)")
    if args.x:
        service.default_config['x'] = True
        x_model_path = args.x_model_path
        x_model_path_2 = args.x_model_path_2
        if x_model_path:
            service.default_config['x_model_path'] = x_model_path
            logger.info(f"[Rank {rank}] X model path set to: {x_model_path}")
        if x_model_path_2:
            service.default_config['x_model_path_2'] = x_model_path_2
            logger.info(f"[Rank {rank}] X model path 2 set to: {x_model_path_2}")
        logger.info(f"[Rank {rank}] Distilled model mode enabled")
    if args.turbo_mode != 'default':
        service.default_config['turbo_mode'] = args.turbo_mode
        logger.info(f"[Rank {rank}] turbo_mode set to: {args.turbo_mode}")
    if args.lora_path_list:
        service.default_config['lora_path_list'] = args.lora_path_list
        logger.info(f"[Rank {rank}] lora_path_list set to: {args.lora_path_list}")
    if args.lora_transformer_list:
        service.default_config['lora_transformer_list'] = args.lora_transformer_list
        logger.info(f"[Rank {rank}] lora_transformer_list set to: {args.lora_transformer_list}")
    external_url = args.external_base_url or os.environ.get('EXTERNAL_BASE_URL')
    if external_url:
        service.default_config['external_base_url'] = external_url
        logger.info(f"[Rank {rank}] External base URL set to: {external_url}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Generation Service")
    from x_diffusers.adaptor.h3_refiner_config import add_arguments
    add_arguments(parser)
    from x_diffusers.adaptor.h3_attention_config import add_arguments as add_h3_attention_arguments
    add_h3_attention_arguments(parser, offline=False)
    from x_diffusers.adaptor.h3_cuda_options import add_arguments as add_cuda_arguments
    add_cuda_arguments(parser, service=True)
    parser.add_argument('--h3_reference_image_short_edge', type=int, default=None,
                        help='Ref2VA Base reference image short edge, 32..2048 in multiples of 32; default 2048')

    parser.add_argument('--host', type=str, default='0.0.0.0', help='Service host')
    parser.add_argument('--port', type=int, default=5001, help='Service port')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--init', action='store_true', help='Initialize service on startup')
    parser.add_argument('--maas', '--maas-fl2va-input', dest='maas', action='store_true',
                        help='Enable H3 Base 768P MaaS validation, media admission and reliability APIs')
    parser.add_argument('--maas-queue-capacity', type=int, default=MAAS_QUEUE_CAPACITY,
                        help='Maximum number of waiting MaaS requests (default: %(default)s)')
    parser.add_argument('--maas-queue-timeout-s', type=int, default=MAAS_QUEUE_TIMEOUT_S,
                        help='MaaS request wait limit in seconds (default: %(default)s)')
    parser.add_argument('--maas-state-dir', default=None,
                        help='Per-replica V2 task history directory (or XGEN_MAAS_STATE_DIR)')
    parser.add_argument('--maas-mock-backends', action='store_true',
                        help='TEST ONLY: enable explicitly labelled fake IR and 2K backends')
    parser.add_argument('--warmup', action='store_true', help='Run warmup inference on startup')
    parser.add_argument('--warmup-steps', type=int, default=1, help='Warmup inference steps')

    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model name (wan2.2-i2v-14b, wan2.2-t2v-14b, or minimax-h3)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Model path (overrides default)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                        help='Data type')
    parser.add_argument('--sp', type=int, default=None,
                        help='Sequence parallel number (number of GPUs)')
    parser.add_argument('--fsdp', action='store_true', default=True,
                        help='Enable FSDP')
    parser.add_argument('--no-fsdp', action='store_true',
                        help='Disable FSDP')
    # MiniMax-H3 specific args
    parser.add_argument('--text_encoder_tp', type=int, default=None,
                        help='Tensor parallel size for text encoder (MiniMax-H3)')
    parser.add_argument('--phaa_num', type=int, default=None,
                        help='PHAA number (MiniMax-H3)')
    parser.add_argument('--no-phaa', action='store_true',
                        help='Disable PHAA (MiniMax-H3)')
    parser.add_argument('--service-mode', type=str, default=None,
                        choices=['t2va', 'fl2va', 'ref2va'],
                        help='Service mode: t2va (allows t2va+fl2va) or ref2va (allows ref2va only)')
    parser.add_argument('--h3_cuda_precision', choices=['auto', 'none', 'fp8-bf16', 'fp8-sage'], default=None,
                        help='CUDA H3 Base defaults to fp8-bf16; none selects legacy; NPU auto stays unchanged')
    parser.add_argument('--h3_vae_backend', choices=['legacy', 'lightning', 'native-stream'], default=None,
                        help='H3 decoder; changing the backend requires restarting the service')
    parser.add_argument('--h3_vae_model_path', default=None, help='Trusted native H3 video_vae directory')
    parser.add_argument('--h3_vae_trust_remote_code', action='store_true', default=None,
                        help='Authorize loading Python code from the native H3 VAE directory')
    parser.add_argument('--h3_parallel_mode', choices=['sp', 'tp_cp', 'tp_cp_shared', 'tp_sp'], default=None,
                        help='Experimental CUDA H3 TP + pure-Ulysses; default preserves SP')
    parser.add_argument('--h3_dit_tp', type=int, default=None)
    parser.add_argument('--h3_tp_comm', choices=['bf16', 'fp8'], default=None)
    parser.add_argument('--h3_cp_comm', choices=['auto', 'bf16', 'fp8'], default=None)
    parser.add_argument('--h3_tp_comm_chunk_elements', type=int, default=None)
    parser.add_argument('--h3_parallel_warmup_mib', type=int, default=None)
    parser.add_argument('--adaln_cache_file', type=str, default=None,
                        help='Path to precomputed AdaLN cache file (MiniMax-H3)')
    parser.add_argument('--transformer_path', type=str, default=None,
                        help='Custom transformer checkpoint directory (MiniMax-H3)')
    parser.add_argument('--vae_matmul_quant', type=str, default=None,
                        choices=['none', 'int8', 'int4', 'mxfp4'],
                        help='VAE matmul quantization type')
    parser.add_argument('--ten-second', action='store_true',
                        help='Enable 10-second video generation mode')
    parser.add_argument('--external-base-url', type=str, default=None,
                        help='External base URL for download links')
    parser.add_argument('--x', action='store_true', default=False,
                    help='Enable distilled model')
    parser.add_argument('--ten-second-model-id-t2v', type=str, default=None,
                        help='10-second T2V model ID path')
    parser.add_argument('--ten-second-model-path', type=str, default=None,
                        help='10-second model path (transformer_14B)')
    parser.add_argument('--ten-second-model-path-2', type=str, default=None,
                        help='10-second model path 2 (transformer_14B_2)')
    parser.add_argument('--pusa-lora', type=str, default=None,
                        help='Pusa LoRA high path')
    parser.add_argument('--pusa-lora2', type=str, default=None,
                        help='Pusa LoRA low path')
    parser.add_argument('--joint-model-path', type=str, default=None,
                        help='Joint model path (1.3B)')
    parser.add_argument('--seedvr2-model-dir', type=str, default=None,
                        help='SeedVR2 model directory')
    parser.add_argument('--seedvr2-model-name', type=str, default=None,
                        help='SeedVR2 model name')
    parser.add_argument('--x-model-path', type=str, default=None,
                        help='X model path (6steps i2v 14B_1)')
    parser.add_argument('--x-model-path-2', type=str, default=None,
                        help='X model path 2 (6steps i2v 14B_2)')
    parser.add_argument('--frame-model-path', type=str, default=None,
                        help='Frame interpolation model path (IFRNet)')
    parser.add_argument('--lora-path-list', nargs='*', type=str, default=None,
                    help='List of LoRA weight paths (multiple paths supported)')
    parser.add_argument("--lora_transformer_list", nargs='+', default=None, type=int, choices=[0, 1, 2],
                       help="The list of which transformer need load lora, 0 means all transformer, 1 means transformer_1, 2 means transformer_2.")
    parser.add_argument('--turbo_mode', type=str, default="default", choices=["default", "faiz", "next_faiz"],
                       help="Use different turbo mode inference.")

    from x_diffusers.adaptor.h3_features import add_aliases
    add_aliases(parser)
    args = parser.parse_args()
    if args.maas and (args.maas_queue_capacity < 1 or args.maas_queue_timeout_s < 1):
        parser.error('MaaS queue capacity and timeout must be positive')
    if not args.maas and (args.maas_mock_backends or args.maas_state_dir):
        parser.error('MaaS state and mock backends require --maas')
    
    _apply_ten_second_config_args(args)

    # Only process startup may set allocator/NCCL defaults. _build_args does
    # not mutate the environment while validating a live worker's request.
    startup_config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, encoding='utf-8') as config_file:
            startup_config = json.load(config_file)
    preset = args.h3_cuda_preset if args.h3_cuda_preset is not None else startup_config.get('h3_cuda_preset')
    if (args.model or startup_config.get('model')) == 'MiniMax-H3' and device_type() == 'cuda':
        from types import SimpleNamespace
        from x_diffusers.adaptor.h3_cuda_options import select_default_recipe
        effective = dict(DEFAULT_MINIMAX_H3_CONFIG, **startup_config)
        explicit = {parser._option_string_actions[token.split('=', 1)[0]].dest
                    for token in sys.argv[1:] if token.split('=', 1)[0] in parser._option_string_actions}
        effective.update({key: getattr(args, key) for key in explicit})
        effective['h3_cuda_preset'] = preset
        if args.no_phaa:
            effective['phaa_num'] = None
        startup = SimpleNamespace(**effective)
        select_default_recipe(startup, 'cuda')
        preset = startup.h3_cuda_preset
    if preset not in (None, 'none'):
        from x_diffusers.adaptor.h3_cuda_options import bootstrap_preset_environment
        bootstrap_preset_environment(preset)

    if args.init:
        init_env()
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    service = VideoGenerationService(config_path=args.config, maas=args.maas,
                                     maas_queue_capacity=args.maas_queue_capacity,
                                     maas_queue_timeout_s=args.maas_queue_timeout_s,
                                     maas_state_dir=args.maas_state_dir,
                                     maas_mock_backends=args.maas_mock_backends)
    
    _apply_service_config(service, args, rank)
    if args.h3_attention != 'legacy' or service.default_config.get('h3_attention', 'legacy') != 'legacy':
        from x_diffusers.adaptor.h3_attention_service import (
            apply_startup_settings as apply_sparse_settings, validate_warmup as validate_sparse_warmup)
        apply_sparse_settings(service.default_config, args, sys.argv[1:])
        validate_sparse_warmup(service.default_config, args.warmup)

    if args.h3_refiner:
        from x_diffusers.adaptor.h3_refiner_service import apply_startup_settings
        apply_startup_settings(service.default_config, args, sys.argv[1:])
    if args.maas:
        if '--maas-fl2va-input' in sys.argv:
            logger.warning('--maas-fl2va-input is deprecated; use --maas')
        if not args.init or service.default_config.get('sp') != world_size:
            parser.error('--maas requires --init and SP equal to the distributed world size')
        from maas import validate_startup
        validate_startup(service.default_config, device_type(), args.warmup)
        validate_media_tools()
    if service.default_config.get('h3_refiner', False):
        if args.warmup:
            raise ValueError('H3 Refiner does not use the legacy reduced-step warmup; submit a normal request')
        from x_diffusers.adaptor.h3_refiner_config import validate_settings
        validate_settings(service._build_args({}, require_prompt=False))

    if args.init:
        service.inference_manager = VideoInferenceManager()
        service.is_initialized = True
        logger.info(f"[Rank {rank}] Service pre-initialized (lazy loading mode)")

    if args.warmup and rank == 0:
        logger.info(f"[Rank {rank}] Running warmup with {args.warmup_steps} steps...")
        service.warmup(warmup_steps=args.warmup_steps)

    if world_size > 1 and rank != 0:
        service._worker_loop()
        return

    service.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()

