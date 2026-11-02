"""NPUTurbo node — optional runtime override for NPU acceleration features.

This node is a **standalone control panel** — it does NOT need to be
connected in the model data flow.  Place it anywhere in the workflow;
ComfyUI will execute it as a root node (no dependencies).

Environment variables that control NPU features at import time::

  X_SP_SIZE=8           — sequence parallel size (init-time, affects model structure)
  X_FSDP=true           — enable FSDP (init-time, affects model structure)

Environment variables that can be **overridden at runtime** by NPUTurbo::

  X_ATTENTION_MODE=sageattn — attention backend (sdpa/sageattn/laserattn/rainfusion/sparse)
  X_PHAA_SPLIT=4        — PHAA overlap split number (1=disabled, 2/4=enabled)

Environment variables that can be **overridden at runtime** by NPUTurbo
(takes effect on next model load, because they modify weights/structure)::

  X_QUANT=int8          — weight/activation quantization
  X_ADALN_CACHE_FILE=/path/to/cache.safetensors — AdaLN cache file
  X_ADALN_TRANSFORMER_PATH=/path/to/stripped/transformer — adaln-stripped checkpoint (file or dir)

The NPUTurbo node is **optional**.  If env vars are set, they take effect at
import time.  Add this node to the workflow only when you want to override
the env var defaults for a specific session — e.g., toggling PHAA
on/off without restarting ComfyUI.

NPUTurbo exposes 8 parameters:
  - attention_mode          : immediate effect
  - phaa_split               : immediate effect
  - quant                    : takes effect on next model load (enable/disable)
  - adaln                    : takes effect on next model load (enable/disable)
  - adaln_cache_file         : takes effect on next model load (empty=use env var)
  - adaln_transformer_path   : takes effect on next model load (empty=use env var)
  - text_tp                  : takes effect on next model load (enable/disable, size from X_TEXT_TP)
  - rope_fused               : immediate effect (X_ROPE_FUSED)

For quant and adaln, the node provides enable/disable toggles.
For adaln, the cache file path and transformer path can be overridden
via text inputs — when non-empty, they override X_ADALN_CACHE_FILE
and X_ADALN_TRANSFORMER_PATH env vars.  When empty, the env vars
are used as fallback.

When adaln is enabled, adaln-related weights are stripped from the state
dict during model loading, saving ~26GB of memory.  Two strategies:
  1. If X_ADALN_TRANSFORMER_PATH is set, load the pre-stripped checkpoint
     from that path (faster, keys already removed).
  2. If not set, filter adaln keys from the original state dict at load
     time (works with any checkpoint).
conditioning_type is auto-detected from the cache file at load time —
no user-facing parameter is needed.

Usage:
    Just add the node to the workflow (no connections needed).
    Select the desired options from the dropdowns.
"""

import logging
import os

logger = logging.getLogger("x_comfyui")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class NPUTurbo:
    """NPU runtime control panel.

    Standalone node — no inputs, no outputs.
    Overrides env var defaults for 8 NPU acceleration features:
      - attention_mode (immediate)
      - phaa_split (immediate)
      - quant (next model load, enable/disable)
      - adaln (next model load, enable/disable)
      - adaln_cache_file (next model load, empty=use env var)
      - adaln_transformer_path (next model load, empty=use env var)
      - text_tp (next model load, enable/disable, size from X_TEXT_TP)
      - rope_fused (immediate, X_ROPE_FUSED)

    Place anywhere in the workflow; execution order is guaranteed
    before any KSampler (KSampler has dependencies, this node has none).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_mode": (
                    ["sdpa", "sageattn", "laserattn", "rainfusion", "sparse"],
                    {"default": "sdpa"},
                ),
                "phaa_split": (
                    ["1", "2", "4"],
                    {"default": "1"},
                ),
                "quant": (
                    ["enable", "disable"],
                    {"default": "disable"},
                ),
                "adaln": (
                    ["enable", "disable"],
                    {"default": "disable"},
                ),
                "adaln_cache_file": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "adaln_transformer_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "text_tp": (
                    ["enable", "disable"],
                    {"default": "disable"},
                ),
                "rope_fused": (
                    ["enable", "disable"],
                    {"default": "disable"},
                ),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "apply"
    CATEGORY = "NPU"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def apply(self, attention_mode, phaa_split, quant,
             adaln, adaln_cache_file, adaln_transformer_path,
             text_tp, rope_fused):
        """Set global NPU runtime flags.

        Overrides X_ATTENTION_MODE, X_PHAA_SPLIT env vars
        (immediate effect), and X_QUANT, X_ADALN_CACHE_FILE,
        X_ADALN_TRANSFORMER_PATH env vars (takes effect on next model load).
        """
        # --- 1. Attention mode ---
        from x_base import attention_manager

        attention_manager.disable_sage_attention()
        attention_manager.disable_laser_attention()
        attention_manager.disable_rainfusion_attention()
        attention_manager.disable_sparse_attention()

        if attention_mode == "sageattn":
            attention_manager.enable_sage_attention()
        elif attention_mode == "laserattn":
            attention_manager.enable_laser_attention()
        elif attention_mode == "rainfusion":
            attention_manager.enable_rainfusion_attention()
        elif attention_mode == "sparse":
            attention_manager.enable_sparse_attention()
        # "sdpa" = default, all flags already disabled

        # --- 2. PHAA ---
        # x_base does not export enable_phaa/disable_phaa from __init__.
        # PHAA globals live in x_base.phaa.globals as ENABLE_PHAA (bool)
        # and PHAA_SPLIT_NUM (int or None). There is no disable_phaa() —
        # we set ENABLE_PHAA = False directly.
        import x_base.phaa.globals as _phaa_globals
        split_num = int(phaa_split)
        if split_num > 1:
            _phaa_globals.ENABLE_PHAA = True
            _phaa_globals.PHAA_SPLIT_NUM = split_num
        else:
            _phaa_globals.ENABLE_PHAA = False
            _phaa_globals.PHAA_SPLIT_NUM = None

        # --- 3. Quant (takes effect on next model load) ---
        # When enabled, get_quant() returns 'int8' (reads X_QUANT for mode).
        # When disabled, get_quant() returns '' (no quantization).
        from x_comfyui.adaptor.env import set_quant_override
        set_quant_override(quant == "enable")

        # --- 4. AdaLN cache (takes effect on next model load) ---
        # When enabled, get_adaln_cache_file() returns the cache file path
        # (from text input or X_ADALN_CACHE_FILE env var).
        # Adaln weights are stripped from the state dict during loading,
        # saving ~26GB.  If adaln_transformer_path is also set, a
        # pre-stripped checkpoint is loaded from that path instead of
        # filtering the original state dict.
        # conditioning_type is auto-detected from the cache file at load time.
        # When disabled, get_adaln_cache_file() returns None (no cache).
        from x_comfyui.adaptor.env import (
            set_adaln_override, set_adaln_cache_file_override,
            set_adaln_transformer_path_override,
        )
        set_adaln_override(adaln == "enable")
        set_adaln_cache_file_override(
            adaln_cache_file if adaln_cache_file else None)

        # --- 5. AdaLN transformer path (takes effect on next model load) ---
        # When non-empty, overrides X_ADALN_TRANSFORMER_PATH env var.
        # When empty, falls back to env var.
        # Only takes effect when adaln is enabled.
        set_adaln_transformer_path_override(
            adaln_transformer_path if adaln_transformer_path else None)

        # --- 6. Text Encoder TP (takes effect on next model load) ---
        # When enabled, uses X_TEXT_TP env var for TP size.
        # When disabled, forces TP size to 0 (disabled).
        from x_comfyui.adaptor.env import set_text_tp_override
        set_text_tp_override(text_tp == "enable")

        # --- 7. RoPE fused (immediate effect, global singleton) ---
        # NOTE: rope_manager is a global singleton — toggling is_rope_fused here
        # affects ALL models using rope_manager (Wan/Flux/MiniMax-H3).
        # See plan guardrail G7.
        from x_base.operator.rope import rope_manager
        if rope_fused == "enable":
            rope_manager.enable_rope_fused()
        else:
            rope_manager.is_rope_fused = False
        from x_comfyui.adaptor.env import set_rope_fused_override
        set_rope_fused_override(rope_fused == "enable")

        logger.info(
            f"NPUTurbo: attention_mode={attention_mode}, "
            f"phaa_split={phaa_split}, "
            f"quant={quant}, adaln={adaln}, "
            f"adaln_cache_file={adaln_cache_file or '(env)'}, "
            f"adaln_transformer_path={adaln_transformer_path or '(env)'}, "
            f"text_tp={text_tp}, "
            f"rope_fused={rope_fused}"
        )
        return ()


class NPUAVDecodeSave:
    """Merged node: VAE Decode + VAE Decode Audio + Create Video + Save Video.

    Replaces the 4-node chain (VAEDecode + VAEDecodeAudio + CreateVideo +
    SaveVideo) with a single node that handles audio decode, video decode
    with streaming, and MP4 output in one pass.

    Implements Route B (audio-first decode): audio VAE decode → save WAV →
    video VAE decode with streaming output via SaveAVStream (ffmpeg dual
    input: stdin video + WAV audio).

    Inputs (match the original 4-node chain):
      Required:
        - video_vae: MiniMax-H3 Video VAE model (from VAELoader)
        - video_latent: Latent from sampler for video decode
        - audio_vae: MiniMax-H3 Audio VAE model (from VAELoader)
        - audio_latent: Latent from sampler for audio decode
        - 帧率: Frames per second (default 24, from CreateVideo)
        - bit_depth: 8 or 10 (from CreateVideo, default 8)
        - 文件名前缀: Output path prefix (from SaveVideo)
        - 格式: Output format (from SaveVideo, default "auto")
        - 编码器: Video codec (from SaveVideo, default "auto")

    `video_latent` is passed to `video_vae.decode()` and `audio_latent`
    is passed to `audio_vae.decode()`.

    Output:
      - UI dict with video file path for ComfyUI preview
    """

    # Codec mapping: user-facing codec name → ffmpeg encoder name
    _ENCODER_MAP = {"auto": None, "h264": "libx264"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "video_latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "audio_latent": ("LATENT",),
                "帧率": ("INT", {"default": 24, "min": 1, "max": 60}),
                "bit_depth": ([8, 10], {"default": 8}),
                "文件名前缀": ("STRING", {"default": "video/MiniMax_H3"}),
                "格式": (["auto", "mp4"], {"default": "auto"}),
                "编码器": (["auto", "h264"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "decode_save"
    CATEGORY = "NPU"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @staticmethod
    def _extract_latent(latent, is_video=True):
        """Extract tensor from latent dict or NestedTensor."""
        # ComfyUI latent format: dict with "samples" key
        if isinstance(latent, dict) and "samples" in latent:
            tensor = latent["samples"]
        else:
            tensor = latent

        # Unwrap NestedTensor (ComfyUI packs video+audio latents into a NestedTensor)
        # Video latent is at index [0] (matching ComfyUI VAEDecode)
        # Audio latent is at index [-1] (matching ComfyUI vae_decode_audio)
        if hasattr(tensor, 'is_nested') and tensor.is_nested:
            tensor = tensor.unbind()[0] if is_video else tensor.unbind()[-1]

        return tensor

    @staticmethod
    def _compute_save_path(文件名前缀):
        """Compute unique save path for output file."""
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except (ImportError, AttributeError):
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        prefix_dir = os.path.dirname(文件名前缀)
        prefix_name = os.path.basename(文件名前缀)
        subfolder = prefix_dir if prefix_dir else ""
        full_output_folder = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(full_output_folder, exist_ok=True)

        # Find next available index
        counter = 1
        while os.path.exists(os.path.join(full_output_folder, f"{prefix_name}_{counter:05d}.mp4")):
            counter += 1
        save_path = os.path.join(full_output_folder, f"{prefix_name}_{counter:05d}.mp4")

        return save_path, subfolder

    @staticmethod
    def _decode_and_save_audio(audio_vae, a_latent, save_path):
        """Decode audio and save to WAV file."""
        try:
            import torch as _torch
            # Decode audio using the audio VAE (matching ComfyUI's vae_decode_audio)
            audio_decoded = audio_vae.decode(a_latent).movedim(-1, 1)

            # Volume normalization (matching ComfyUI's vae_decode_audio)
            std = _torch.std(audio_decoded, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            audio_decoded = audio_decoded / std

            # Get sample rate from VAE (matching ComfyUI's vae_decode_audio)
            vae_sample_rate = getattr(
                audio_vae, "audio_sample_rate_output",
                getattr(audio_vae, "audio_sample_rate", 44100)
            )

            # Save to WAV
            from x_base.vae_parallelism.audio_utils import save_audio_to_wav
            audio_path = save_audio_to_wav(
                audio_decoded,
                sampling_rate=vae_sample_rate,
                audio_path=save_path.replace(".mp4", ".wav"),
            )
            logger.info(f"NPUAVDecodeSave: audio saved to {audio_path}")
            return audio_path
        except Exception as e:
            logger.warning(f"NPUAVDecodeSave: audio decode failed: {e}")
            return None

    @staticmethod
    def _compute_video_dimensions(video_vae, v_latent):
        """Compute video dimensions from latent shape."""
        vae_spatial_ratio = getattr(video_vae, "vae_ratio", 8)
        # v_latent shape: (B, C, T, H, W) or (B, C, S)
        if v_latent.dim() >= 4:
            height = v_latent.shape[-2] * vae_spatial_ratio
            width = v_latent.shape[-1] * vae_spatial_ratio
        else:
            # Fallback for flattened latent
            height = 720
            width = 1280
            logger.warning(
                f"NPUAVDecodeSave: cannot determine video dimensions from latent "
                f"shape {v_latent.shape}, using default {width}x{height}"
            )
        return width, height

    def decode_save(self, video_vae, video_latent, audio_vae, audio_latent,
                    帧率, bit_depth, 文件名前缀, 格式, 编码器):
        """Decode video and audio, save to MP4 via streaming.

        Route B flow:
        1. Extract latent tensors from video_latent and audio_latent
        2. Decode audio (audio_vae.decode) → save WAV
        3. Pass av_stream_config to video_vae.decode — VAE manages SaveAVStream lifecycle
        """
        # --- 1. Extract latent tensors ---
        v_latent = self._extract_latent(video_latent, is_video=True)
        a_latent = self._extract_latent(audio_latent, is_video=False)

        # --- 2. Compute save path ---
        save_path, subfolder = self._compute_save_path(文件名前缀)

        # --- 3. Decode audio ---
        audio_path = self._decode_and_save_audio(audio_vae, a_latent, save_path)

        # --- 4. Compute video dimensions ---
        width, height = self._compute_video_dimensions(video_vae, v_latent)

        # --- 5. Video VAE decode with streaming ---
        # Pass config dict — VAE internally creates, uses, and closes SaveAVStream
        codec = self._ENCODER_MAP.get(编码器, None)
        try:
            video_vae.decode(v_latent, vae_options={
                "av_stream_config": {
                    "save_path": save_path,
                    "audio_path": audio_path,
                    "fps": 帧率,
                    "bit_depth": bit_depth,
                    "codec": codec,
                }
            })
        except Exception as e:
            logger.error(f"NPUAVDecodeSave: video decode failed: {e}")
            raise

        logger.info(
            f"NPUAVDecodeSave: saved {save_path} "
            f"({width}x{height}, {帧率}fps, bit_depth={bit_depth}, "
            f"audio={'yes' if audio_path else 'no'})"
        )

        filename = os.path.basename(save_path)
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS["NPUTurbo"] = NPUTurbo
NODE_DISPLAY_NAME_MAPPINGS["NPUTurbo"] = "NPU Turbo"

NODE_CLASS_MAPPINGS["NPUAVDecodeSave"] = NPUAVDecodeSave
NODE_DISPLAY_NAME_MAPPINGS["NPUAVDecodeSave"] = "NPU AV Decode & Save"
