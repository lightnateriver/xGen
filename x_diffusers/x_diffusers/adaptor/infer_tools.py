import os
import csv
import shutil
import random
from itertools import product
import time
from copy import deepcopy
import torch.distributed as dist
import math
import numpy as np
import torch
from diffusers.utils import export_to_video, load_image, logging
from PIL import Image, ImageEnhance
import ffmpeg
import gc
import inspect

from .load_pipe import load_pipe, load_v2v_pipe, load_seedvr2_pipe, update_scheduler, update_lora, update_pipe, \
    pipe_to_device
from x_base import (infer_info, prepare_video_and_mask, attention_manager,
                    read_text, list_cases, split_rectangle, lcm, floor_to_multiple, get_patch_hw,
                    SaveVideoStream, save_audio_to_wav, cleanup_audio)
from ..framework.vae.wan import vfi
from ..framework.vae.IFRNet_S_arch import IRFNet_S

logger = logging.get_logger("infer")

X_GUIDANCE_SCALE = 1.0
# v2v配置
V2V_HISTORY_FRAMES = 17
V2V_COND_POS_LIST = [0, 1, 2, 3, 4]
V2V_NOISE_MULT_LIST = [0, 0.1, 0.3, 0.5, 0.7]
V2V_INFERENCE_STEPS = 4
V2V_OUTPUT_TYPE = "np"

GUIDANCE_SCALE_MAP = {
    "Wan2.1-T2V-14B": (5.0, None),
    "Wan2.1-I2V-14B": (5.0, None),
    "Wan2.1-T2V-1.3B": (5.0, None),
    "CogVideoX-5b": (5.0, None),
    "HunyuanVideo-T2V-13B": (5.0, None),
    "Wan2.2-I2V-A14B": (3.5, 3.5),
    "Wan2.2-T2V-A14B": (3.0, 4.0),
    "MiniMax-H3": (None, None),  # guidance-distilled — no guidance scale
}


class InferenceManager(object):
    def __init__(self):
        self.init_args = None
        self.pipe = None
        self.sr_pipe = None
        self.v2v_pipe = None
        # pipe abilities args
        self.need_reload_weight_abilities_list = [
            "sp",
            "turbo_mode",
            "quality_enhance_mode",
            "x",
            "x_model_path",
            "fsdp",
            "matmul_quant",
            "inf_vram_blocks_num",
            "h3_turbo_sla_lora",
            "h3_turbo8",
            "h3_nvfp4_checkpoint",
            "h3_nvfp4_resident",
            "h3_nvfp4_compile",
            "h3_cuda_vae_sp",
        ]
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def check_abilities_args_consistent(self, args):
        if self.init_args is None:
            return False
        if getattr(args, "h3_turbo_sla_lora", None) or getattr(self.init_args, "h3_turbo_sla_lora", None):
            if args.model != self.init_args.model or args.task_type != self.init_args.task_type:
                return False
        if getattr(args, "h3_turbo8", False) and any(
                getattr(args, name) != getattr(self.init_args, name)
                for name in ("transformer_path", "adaln_cache_file", "pretrained_model_name_or_path")):
            return False
        if args.fuse_lora and args.lora_path_list != self.init_args.lora_path_list:
            return False
        if args.fuse_lora is False and self.init_args.fuse_lora is True:
            return False
        for abilities_args in self.need_reload_weight_abilities_list:
            if getattr(self.init_args, abilities_args, None) != getattr(args, abilities_args, None):
                return False
        return True

    def wan_image_preprocess(self, image, height, width):
        # ---- input / constraint source ----
        img_h, img_w = int(image.height), int(image.width)

        if img_h <= 0 or img_w <= 0:
            raise ValueError(f"Invalid image size: height={img_h}, width={img_w}")

        aspect_ratio = img_h / img_w

        max_area = int(height) * int(width)
        base_h, base_w = int(height), int(width)

        # ---- model factors ----
        s = int(self.pipe.vae_scale_factor_spatial)  # 空间压缩率，例如 8
        patch_size = self.pipe.transformer.config.patch_size  # (1,2,2)
        patch_h, patch_w = get_patch_hw(patch_size)  # (2,2)
        mod_value = s * patch_w  # or s*patch_h (same here)

        # ---- target size from area + aspect ----
        raw_target_h = int(round(np.sqrt(max_area * aspect_ratio)))
        raw_target_w = int(round(np.sqrt(max_area / aspect_ratio)))

        infer_info.save_width = raw_target_w if raw_target_w % 2 == 0 else raw_target_w + 1
        infer_info.save_height = raw_target_h if raw_target_h % 2 == 0 else raw_target_h + 1

        target_h = floor_to_multiple(raw_target_h, mod_value)
        target_w = floor_to_multiple(raw_target_w, mod_value)

        # ---- choose grid in latent space ----
        lat_h0, lat_w0 = target_h // s, target_w // s
        tile_world_size = infer_info.get_tile_world_size(self.world_size)
        num_rows, num_cols = split_rectangle(lat_h0, lat_w0, tile_world_size)

        # ---- enforce divisibility in latent space ----
        # 让 latent_h % num_rows == 0, latent_w % num_cols == 0
        step_h = lcm(mod_value, s * num_rows)
        step_w = lcm(mod_value, s * num_cols)

        new_h = max(floor_to_multiple(target_h, step_h), step_h)
        new_w = max(floor_to_multiple(target_w, step_w), step_w)

        # ---- derived diagnostics ----
        new_lat_h, new_lat_w = new_h // s, new_w // s
        tile_lat_h, tile_lat_w = new_lat_h / num_rows, new_lat_w / num_cols  # 期望是整数

        if getattr(self, "rank", 0) == 0:
            logger.info(f"[wan_image_preprocess] original_hw=({img_h},{img_w}) -> final_hw=({new_h},{new_w})")

        return new_h, new_w

    @staticmethod
    def ada_bright_image_preprocess(image, args):
        if args.ada_brighten:
            img_gray = image.convert('L')
            pixels = list(img_gray.getdata())
            img_lum = sum(pixels) / len(pixels)
            args.ada_brighten = (img_lum > 190)
            infer_info.update_adabrighten(args.ada_brighten)
        if args.ada_brighten:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.7)
        return image

    @staticmethod
    def restore_brightness(frames):
        out = []
        for f in frames:
            numpy_array = np.clip(f * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(numpy_array)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.0 / 0.7)
            out.append(img)
        return out

    @staticmethod
    def change_fps_filter(input_path, output_path, out_fps=32, out_crf='10'):
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
                logger.info(f"自动创建输出目录成功 → {output_dir}")
            stream = (
                ffmpeg
                .input(input_path)
                .filter('fps', fps=out_fps, round='near')
                .output(output_path, vcodec='libx264', preset='fast', crf=out_crf, acodec='copy', loglevel='error')
                .overwrite_output()
            )
            ffmpeg.run(stream)
            os.remove(input_path)
            os.rename(output_path, input_path)
        except ffmpeg.Error as e:
            logger.error(f'高分辨率失败：{str(e)}')

    def save_output(self, output, save_path, args):
        if args.task_type == "i2v" or args.task_type == "t2v":
            if args.adopt_sr:
                if args.ten_second and torch.distributed.get_rank() == 0 and args.save_fps != 16:
                    directory = os.path.dirname(args.save_path)
                    filename = os.path.basename(args.save_path)
                    save_path = os.path.join(directory, "tmp_" + filename)
                    self.change_fps_filter(args.save_path, save_path, out_fps=args.save_fps)
                else:
                    return
            if not args.ten_second and args.vae_lightning in ["decoder", "encoder_and_decoder"] and args.sp > 1:
                return
            if args.task_type == "i2v" and args.ada_brighten:
                output = self.restore_brightness(output)
            if torch.distributed.get_rank() == 0 and output is not None:
                if not args.ten_second:
                    export_to_video(output, save_path, quality=8, fps=args.save_fps)
                else:
                    # 10s场景下需要将帧数补全
                    directory = os.path.dirname(args.save_path)
                    filename = os.path.basename(args.save_path)
                    save_path = os.path.join(directory, "tmp_" + filename)
                    save_fps = args.save_fps * 14 // 16
                    export_to_video(output, save_path, quality=8, fps=save_fps)
                    self.change_fps_filter(save_path, args.save_path, out_fps=args.save_fps)

        if args.task_type == "t2i":
            output = output.squeeze(0).clip(-1, 1)
            output = (output * 255).astype(np.uint8)
            im = Image.fromarray(output)
            im.save(save_path)

    def parse_test_matrix(self, args, mode):
        """根据模式获取测试矩阵，支持用户通过命令行参数覆盖默认值"""
        # 1. 设定默认值
        if mode == "speed":
            default_res = [(832, 480), (1280, 720)]
            default_frames = [81, 49, 17]
            default_targets = ["none", "encoder", "decoder", "encoder_and_decoder"]
        else:  # accuracy
            default_res = [(832, 480)]
            default_frames = [81]
            default_targets = ["none", "encoder_and_decoder"]

        # 2. 解析分辨率
        resolutions = default_res
        if args.batch_test_resolutions:
            try:
                res_list = []
                for r in args.batch_test_resolutions.split(","):
                    w, h = r.lower().split("x")
                    res_list.append((int(w), int(h)))
                resolutions = res_list
            except Exception as e:
                print(f"解析 --batch_test_resolutions 失败，使用默认值。错误: {e}")

        # 3. 解析帧数
        frames = default_frames
        if args.batch_test_frames:
            try:
                frames = [int(f.strip()) for f in args.batch_test_frames.split(",")]
            except Exception as e:
                print(f"解析 --batch_test_frames 失败，使用默认值。错误: {e}")

        # 4. 解析加速对象
        targets = default_targets
        if args.batch_test_targets:
            targets = [t.strip() for t in args.batch_test_targets.split(",")]

        return resolutions, frames, targets

    def run_test_pipeline(self, args):
        """
        根据 args.batch_test_mode 决定执行常规推理、速度测试还是精度测试。
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if rank == 0 and args.batch_test_mode in ["speed", "accuracy"]:
            os.makedirs(args.batch_test_report_dir, exist_ok=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # ================= 1. 获取并抽样数据集 =================
        sampled_dirs = self._sample_test_cases(args)
        args.sampled_case_dirs = sampled_dirs  # 透传给 infer_batch_from_testdata

        # ================= 2. 分支执行 =================
        if args.batch_test_mode == "normal":
            self.infer_batch(args)
        elif args.batch_test_mode == "speed":
            self._run_speed_test(args, sampled_dirs, rank)
        elif args.batch_test_mode == "accuracy":
            self._run_accuracy_test(args, sampled_dirs, rank)

    def _sample_test_cases(self, args):
        """抽取测试用例"""
        all_case_dirs = list_cases(args.batch_test_input_dir)
        if args.batch_test_sample_size > 0 and len(all_case_dirs) > args.batch_test_sample_size:
            # 固定随机种子，保证每次运行抽样的 case 都是一样的
            random.seed(args.seed if hasattr(args, 'seed') else 42)
            return random.sample(all_case_dirs, args.batch_test_sample_size)
        return all_case_dirs

    def _run_speed_test(self, args, sampled_dirs, rank):
        """执行速度测试逻辑"""
        report_path = os.path.join(args.batch_test_report_dir, "report_speed.csv")
        if rank == 0:
            logger.info(f">>> 启动 [速度验证] 测试，共 {len(sampled_dirs)} 个 case...\n")
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w", newline="", encoding="utf-8") as f:
                f.write("Case_ID,Resolution,Frames,VAE_Lighting_Target,Encode_Time(ms),Decode_Time(ms)\n")

        resolutions, frames, targets = self.parse_test_matrix(args, "speed")

        # 使用 itertools.product 替代三层嵌套循环，直接降低圈复杂度
        for (w, h), f, target in product(resolutions, frames, targets):
            test_args = deepcopy(args)
            test_args.enable_vae_time_count = True
            test_args.width = w
            test_args.height = h
            test_args.frames = f
            test_args.vae_lightning = target

            if rank == 0:
                logger.info(f"--- 速度测试组合: {w}x{h}, {f}帧, {target} ---")
            self.infer_batch(test_args)

        if rank == 0:
            logger.info(f">>> [速度验证] 测试完成，报告已保存至 {report_path}")

    def _run_accuracy_test(self, args, sampled_dirs, rank):
        """执行精度测试逻辑"""
        report_path = os.path.join(args.batch_test_report_dir, "prompt.txt")
        task_desc_path = os.path.join(args.batch_test_report_dir, "任务描述.txt")

        if rank == 0:
            logger.info(f">>> 启动 [精度验证] 测试，共 {len(sampled_dirs)} 个 case...")
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(task_desc_path, "w", encoding="utf-8") as f:
                f.write("任务背景：VAE并行加速精度测试\n\n原始目录：original\n\n对比目录：target\n")

            with open(report_path, "w", encoding="utf-8") as f:
                for case_dir in sampled_dirs:
                    case_id = os.path.basename(case_dir)
                    prompt_file = os.path.join(case_dir, "prompt.txt")
                    prompt_text = read_text(prompt_file).strip().replace('\n', '')
                    f.write(f"{case_id}.mp4,{prompt_text}\n")

        resolutions, frames, targets = self.parse_test_matrix(args, "accuracy")

        # 同样使用 product 压平嵌套循环
        for (w, h), f, target in product(resolutions, frames, targets):
            test_args = deepcopy(args)
            test_args.enable_vae_time_count = False
            test_args.width = w
            test_args.height = h
            test_args.frames = f
            test_args.vae_lightning = target

            if rank == 0:
                logger.info(f"--- 精度测试组合: {w}x{h}, {f}帧, VAE加速对象：{target} ---")
            self.infer_batch(test_args)

        if rank == 0:
            logger.info(f">>> [精度验证] 测试完成，报告已保存至 {report_path}")

    def _get_output_dir(self, case_args, case_dir: str, report_dir: str) -> str:
        """抽离：根据不同测试模式动态计算输出目录"""
        if case_args.batch_test_mode == "speed":
            return os.path.join(report_dir, f"{case_args.width}x{case_args.height}_{case_args.frames}f",
                                f"lighting-{case_args.vae_lightning}")

        if case_args.batch_test_mode == "accuracy":
            sub_folder = "original" if case_args.vae_lightning == "none" else "target"
            return os.path.join(report_dir, sub_folder)

        # 常规模式
        output_dir = os.path.join(case_dir, case_args.task_type)
        is_lightning_mode = (
                case_args.vae_lightning in ["encoder", "decoder", "encoder_and_decoder"]
                and case_args.vae_pad_mode in ["tail_only", "all_sides", "all_sides_valid"]
        )

        if not is_lightning_mode:
            return output_dir

        output_dir = os.path.join(output_dir, case_args.vae_pad_mode, case_args.vae_lightning)
        if case_args.vae_pad_latent_size <= 0:
            return os.path.join(output_dir, "no_pad")

        if case_args.vae_pad_mode == "all_sides":
            output_dir = os.path.join(output_dir, f"pad_content-{case_args.vae_pad_content}")
        if case_args.vae_pad_mode in ["tail_only", "all_sides"]:
            output_dir = os.path.join(output_dir, f"use_blend-{case_args.vae_use_blend}")

        return os.path.join(output_dir, f"pad_latent_size-{case_args.vae_pad_latent_size}")

    def _append_speed_report(self, args, case_id: str):
        """抽离：仅速度测试时追加CSV记录"""
        if args.batch_test_mode != "speed":
            return
        speed_report = os.path.join(getattr(args, "batch_test_report_dir", "./"), "report_speed.csv")
        with open(speed_report, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                case_id, f"{args.width}x{args.height}", args.frames,
                args.vae_lightning, infer_info.vae_last_encode_time, infer_info.vae_last_decode_time
            ])

    def infer_batch_from_testdata(
            self,
            args,
            image_name: str = "image.jpg",
            prompt_name: str = "prompt.txt",
            out_video_name: str = "output.mp4",
    ):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        case_dirs = getattr(args, "sampled_case_dirs", None) or list_cases(args.batch_test_input_dir)
        if rank == 0:
            logger.info(f"[BATCH] Processing {len(case_dirs)} cases...")

        failed = []

        for idx, case_dir in enumerate(case_dirs):
            case_id = os.path.basename(case_dir)
            if rank == 0:
                logger.info(f"[BATCH] ({idx + 1}/{len(case_dirs)}) case={case_id} dir={case_dir}")

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            case_args = deepcopy(args)
            case_args.i2v_image_path = os.path.join(case_dir, image_name)
            case_args.prompt = read_text(os.path.join(case_dir, prompt_name))

            # 1. 调用抽离的路径生成逻辑
            report_dir = getattr(case_args, "batch_test_report_dir", "")
            output_dir = self._get_output_dir(case_args, case_dir, report_dir)

            out_name = f"{case_id}.mp4" if case_args.batch_test_mode in ["speed", "accuracy"] else out_video_name
            case_args.save_path = os.path.join(output_dir, out_name)

            if rank == 0:
                os.makedirs(output_dir, exist_ok=True)
                if case_args.vae_output_dir:
                    case_args.vae_output_dir = os.path.join(output_dir, case_args.vae_output_dir)
                    os.makedirs(case_args.vae_output_dir, exist_ok=True)

            infer_info.update_info(case_args)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            torch.cuda.synchronize()
            start_time = time.time()

            try:
                self.infer_single(case_args)
                if rank == 0:
                    # 2. 调用抽离的写报告逻辑
                    self._append_speed_report(case_args, case_id)
                    logger.info(f"[BATCH] case={case_id} OK -> {case_args.save_path}")
            except Exception as e:
                logger.error(f"[BATCH] case={case_id} Failed: {e}")
                failed.append({"case_id": case_id, "error": str(e)})

            torch.cuda.synchronize()
            if rank == 0:
                logger.info(f"time cost:{time.time() - start_time}")

        return failed

    # ------------------------------------------------------------------
    # MiniMax-H3 inference (ModularPipeline — joint video + audio)
    # ------------------------------------------------------------------

    _MMH3_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    _MMH3_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    _MMH3_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aac", ".ogg"}

    def _infer_minimax_h3(self, args):
        """Run MiniMax-H3 joint video + audio generation via ModularPipeline."""
        if getattr(args, "h3_refiner", False):
            from .h3_refiner import infer
            return infer(self, args)
        infer_params = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "num_inference_steps": args.num_inference_steps,
            "generator": torch.Generator().manual_seed(args.seed),
        }
        if getattr(args, "h3_turbo8", False):
            from .h3_turbo8 import validate
            validate(args)
        from .h3_turbo_sla import inference_params
        infer_params.update(inference_params(self.pipe, args))

        # Keyframe conditioning (fl2va / fl2va_last_frame)
        if getattr(args, "keyframe_image", ""):
            infer_params["image"] = load_image(args.keyframe_image)
        if getattr(args, "keyframe_last_image", ""):
            infer_params["last_image"] = load_image(args.keyframe_last_image)

        # Reference conditioning (ref2va)
        ref_paths = getattr(args, "reference_paths", None)
        if ref_paths:
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
            if references:
                infer_params["references"] = references

        # Rainfusion / sparse attention setup (MiniMax-H3 early-returns from infer_single,
        # so the generic setup at L669 is never reached — must set up here).
        if args.atten == 'rainfusion':
            num_latent_frames = (args.frames - 1) // 4 + 1
            # MiniMax-H3 VAE spatial_downsample_factors = (2,2,2,2,1,1) → product = 16
            latent_height = args.height // 16
            latent_width = args.width // 16
            attention_manager.set_rainfusion_attention(
                [num_latent_frames, latent_height, latent_width],
                skip_timesteps=int(args.rainfusion_ratio * args.num_inference_steps),
                sparsity=getattr(args, 'rainfusion_sparsity', 0.64))
        if args.atten == 'sparse':
            attention_manager.set_ada_bsa_sparse_flash_attention(args.ada_sparsity)

        # RoPE table cache: reuse cos/sin tables across denoise steps
        if getattr(args, "rope_table_cache", False):
            from x_base.operator.rope import enable_rope_cache
            enable_rope_cache()

        # ModularPipeline returns a dict when ``output=`` is specified.
        if getattr(args, 'h3_attention', 'legacy') != 'legacy':
            self.pipe.transformer._h3_attention_session.begin(args)
        result = self.pipe(output=["videos", "audio", "sampling_rate"], **infer_params)

        videos = result.get("videos")
        audio = result.get("audio")
        sampling_rate = result.get("sampling_rate")

        torch.cuda.empty_cache()

        # Route B: SaveAVStream 已在 _decode_ascend 内部完成保存
        if getattr(infer_info, "_route_b_saved", False):
            infer_info._route_b_saved = False  # cleanup
            return

        self.save_minimax_h3_output(videos, audio, sampling_rate, args)

    @staticmethod
    def _flatten_frames(lst):
        """Recursively yield items from nested lists/tuples."""
        for item in lst:
            if isinstance(item, (list, tuple)):
                yield from InferenceManager._flatten_frames(item)
            else:
                yield item

    @staticmethod
    def _frame_to_uint8_hwc(frame):
        """Convert any frame type to uint8 HWC numpy array.

        Pipeline already converts VAE output to [0,1] via ImageNet
        denormalization, so we only need ×255 here.
        """
        if isinstance(frame, Image.Image):
            return np.array(frame)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[0] in (3, 4):
                frame = frame.transpose(1, 2, 0)  # CHW → HWC
            if frame.dtype == np.uint8:
                return frame
            # float [0,1] → uint8
            frame = np.clip(frame, 0, 1) * 255
            return frame.astype(np.uint8)
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    @staticmethod
    def _build_ffmpeg_cmd(width, height, fps, audio_path, save_path):
        """Build ffmpeg command for raw video pipe input to MP4."""
        ffmpeg_bin = os.getenv("FFMPEG_PATH", "ffmpeg")
        cmd = [ffmpeg_bin, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
        if audio_path:
            cmd += ["-i", audio_path]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p"]
        if audio_path:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += ["-loglevel", "error", save_path]
        return cmd

    def _save_minimax_h3_with_ffmpeg(self, frames, audio_path, args):
        """Save video frames + audio via ffmpeg subprocess to MP4 (libx264 + aac)."""
        import subprocess

        first_frame = InferenceManager._frame_to_uint8_hwc(frames[0])
        height, width = first_frame.shape[0], first_frame.shape[1]
        fps = getattr(args, "save_fps", None) or 24

        cmd = InferenceManager._build_ffmpeg_cmd(width, height, fps, audio_path, args.save_path)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(InferenceManager._frame_to_uint8_hwc(frame).tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
        logger.info(f"Saved MiniMax-H3 video+audio to {args.save_path}")

    @staticmethod
    def _tensor_to_video_frames(tensor):
        """Convert a video tensor (5D/4D/3D) to a list of HWC numpy frames."""
        if tensor.dim() == 5:
            tensor = tensor[0]
        frames = []
        if tensor.dim() == 4:
            for t in range(tensor.shape[1]):
                frame = tensor[:, t].permute(1, 2, 0)
                frames.append(frame.cpu().numpy())
        elif tensor.dim() == 3:
            frame = tensor
            if frame.shape[0] in (3, 4):
                frame = frame.permute(1, 2, 0)
            frames.append(frame.cpu().numpy())
        return frames

    @staticmethod
    def _list_to_video_frames(videos):
        """Convert a nested list of frames (PIL/Tensor/ndarray) to a list of frames."""
        frames = []
        for frame in InferenceManager._flatten_frames(videos):
            if isinstance(frame, Image.Image):
                frames.append(frame)
            elif isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    frame = frame.transpose(1, 2, 0)
                frames.append(frame)
            elif isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    frame = frame.transpose(1, 2, 0)
                frames.append(frame)
        return frames

    def _save_minimax_h3_video_fallback(self, videos, args):
        """Fallback: save video-only via export_to_video when ffmpeg fails."""
        if isinstance(videos, torch.Tensor):
            video_frames = InferenceManager._tensor_to_video_frames(videos)
        else:
            video_frames = InferenceManager._list_to_video_frames(videos)
        export_to_video(video_frames, args.save_path, quality=8, fps=24)
        logger.info(f"Saved MiniMax-H3 video to {args.save_path} (fallback)")

    def save_minimax_h3_output(self, videos, audio, sampling_rate, args):
        """Save MiniMax-H3 video + audio into a single MP4 with embedded audio.

        Route A (non-optimized): full decode completes first, then save.
        Step 1: audio tensor -> temporary WAV file (via save_audio_to_wav)
        Step 2: video frames + WAV -> ffmpeg subprocess -> MP4 (libx264 + aac)
        Step 3: delete temporary WAV (via cleanup_audio)
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            return

        audio_path = save_audio_to_wav(
            audio, sampling_rate,
            audio_path=getattr(args, "audio_save_path", "") or None,
            save_path=args.save_path,
        )

        try:
            if videos is None:
                return
            frames = list(InferenceManager._flatten_frames(videos))
            if not frames:
                logger.warning("No video frames to save")
                return
            try:
                self._save_minimax_h3_with_ffmpeg(frames, audio_path, args)
            except Exception as e:
                logger.warning(f"ffmpeg save failed ({e}), falling back to export_to_video")
                self._save_minimax_h3_video_fallback(videos, args)
        finally:
            cleanup_audio(audio_path)

    def infer_single(self, args):
        # --- MiniMax-H3 branch (ModularPipeline, joint video+audio) ---
        if "MiniMax-H3" in args.model:
            self._infer_minimax_h3(args)
            return

        infer_params = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator": torch.Generator().manual_seed(args.seed)
        }
        if "Wan2.2" in args.model:
            infer_params["guidance_scale_2"] = args.guidance_scale_2

        save_path = args.save_path

        if "VACE" in args.model:
            video, mask, reference_image = prepare_video_and_mask(args)
            del infer_params["guidance_scale_2"]
            infer_params["video"] = video
            infer_params["mask"] = mask
            infer_params["reference_image"] = reference_image
        elif args.task_type == "i2v":
            image = load_image(args.i2v_image_path)
            if args.i2v_image_preprocess == "wan":
                height, width = self.wan_image_preprocess(image, height=args.height, width=args.width)
                infer_info.update_shape(width, height)
            else:
                width, height = args.width, args.height
            image = image.resize((width, height))
            image = self.ada_bright_image_preprocess(image, args)
            infer_params["image"] = image
            infer_params["height"] = height
            infer_params["width"] = width
        elif args.task_type == "t2v":
            width, height = args.width, args.height

        if args.atten == 'rainfusion':
            num_latent_frames = (infer_params["num_frames"] - 1) // 4 + 1
            latent_height = infer_params["height"] // 8
            latent_width = infer_params["width"] // 8
            attention_manager.set_rainfusion_attention(
                [num_latent_frames, latent_height, latent_width],
                skip_timesteps=int(args.rainfusion_ratio * args.num_inference_steps),
                sparsity=args.rainfusion_sparsity)
        if args.atten == 'sparse':
            attention_manager.set_ada_bsa_sparse_flash_attention(args.ada_sparsity)

        output = self.pipe(**infer_params).frames[0]

        torch.cuda.empty_cache()

        if args.ten_second:
            his = 17
            cond_pos_list = [0, 1, 2, 3, 4]
            noise_mult_list = [0, 0.1, 0.3, 0.5, 0.7]
            steps_v2v = 4
            last_frames = output[-his:]

            # Stage B: V2V extension  (force numpy output)
            g_ext = torch.Generator().manual_seed(args.seed)
            ext = self.v2v_pipe(
                conditioning_video=last_frames,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                conditioning_indices=cond_pos_list,
                conditioning_noise_multipliers=noise_mult_list,
                height=height,
                width=width,
                num_frames=args.frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=steps_v2v,
                generator=g_ext,
                output_type="np",  # <-- ensure NumPy frames
            ).frames[0]

            output = np.concatenate([output[:-his], ext], axis=0)  # (T_total, H, W, C)

        if args.ten_second and args.frame_interpolation:
            output = self.frame_interpolation(args, output)
            infer_info.fps = args.save_fps * 14 // 16  # 10s视频生成需由14帧保存才能获得真正的10s

        if args.adopt_sr:
            output = self.sr_pipe(output)

        self.save_output(output, save_path, args)
        return

    @staticmethod
    def frame_interpolation(args, output):
        output = torch.from_numpy(output)

        rank = torch.distributed.get_rank()
        size = torch.distributed.get_world_size()

        mod = output.shape[0] % size
        step = output.shape[0] // size
        output = output.permute(0, 3, 1, 2)
        # input shape：F C H W
        if mod != 0 and rank == 0:
            input1 = output[:mod]
            output = output[mod:]
        if rank != (size - 1):
            input = output[step * rank: step * (rank + 1) + 1]
        else:
            input = torch.cat([output[step * rank: step * (rank + 1)], output[-1:]], dim=0)

        interpolation_model = IRFNet_S()
        interpolation_model.load_state_dict(torch.load(args.frame_model_path, weights_only=True))
        if infer_info.fps % 16 == 0:
            multiplier = args.save_fps // 16
            is_skip = False
        elif args.save_fps % 8 < 4:
            multiplier = args.save_fps // 16 + 1
            is_skip = True
        else:
            multiplier = args.save_fps // 16 + 1
            is_skip = False

        out = vfi(input, rank, interpolation_model, 1, multiplier=multiplier, is_skip=is_skip)[:-1].permute(
            0, 2, 3, 1).contiguous()
        out1 = torch.zeros_like(out)[:0].to(rank).contiguous()
        if rank == 0 and mod != 0 and input1.shape[0] >= 2:
            out1 = vfi(input1, rank, interpolation_model, 1, multiplier=multiplier,
                       is_skip=is_skip).permute(0, 2, 3, 1).contiguous().cpu()
        elif rank == 0 and mod != 0:
            out1 = input1.permute(0, 2, 3, 1).contiguous().cpu()
        out2 = [torch.empty_like(out) for _ in range(size)]
        torch.distributed.all_gather(out2, out)
        output = torch.cat(out2, dim=0).cpu()
        if mod != 0:
            output = torch.cat([out1.cpu(), output.cpu()], dim=0)
        output = output.numpy()  # (T_total, H, W, C)
        return output

    def _prepare_pipeline(self, args):
        """抽离出的公共逻辑：处理 pipeline 重建、LoRA 更新和设备分配"""
        if getattr(args, "h3_refiner", False):
            from .h3_refiner import prepare
            return prepare(self, args)
        if getattr(self, "_h3_refiner_key", None) is not None:
            raise ValueError("Restart the manager to disable H3 Refiner and restore the legacy backend")
        consistent = self.check_abilities_args_consistent(args)

        if not consistent:
            logger.info("Different abilities require rebuilding pipe")
            self.pipe = load_pipe(args)
            # 先加载LoRA，再进行FSDP包装
            update_lora(
                self.pipe,
                args.model,
                self.init_args.lora_path_list if self.init_args is not None else None,
                args.lora_path_list,
                fuse_lora=args.fuse_lora,
                weights_list=args.lora_scale_weight_list
            )
            update_pipe(self.pipe, args)
            if args.adopt_sr:
                self.sr_pipe = load_seedvr2_pipe(args)
            if args.ten_second:
                v2v_args = deepcopy(args)
                self.v2v_pipe = load_v2v_pipe(v2v_args)
                v2v_args.matmul_quant = 'none'
                v2v_args.joint = False
                update_pipe(self.v2v_pipe, v2v_args)
        else:
            # consistent为True时，只更新LoRA（不重建pipeline）
            update_lora(
                self.pipe,
                args.model,
                self.init_args.lora_path_list if self.init_args is not None else None,
                args.lora_path_list,
                fuse_lora=args.fuse_lora,
                weights_list=args.lora_scale_weight_list
            )

        if not consistent:
            self.pipe = pipe_to_device(self.pipe, args)
            if args.ten_second:
                self.v2v_pipe = pipe_to_device(self.v2v_pipe, args)

    def infer_batch(self, args):
        self._prepare_pipeline(args)

        self.init_args = deepcopy(args)
        return self.infer_batch_from_testdata(args)

    def infer(self, args) -> float:
        if (getattr(args, 'h3_attention', 'legacy') != 'legacy' or
                getattr(self.init_args, 'h3_attention', 'legacy') != 'legacy'):
            from .h3_attention_config import key
            if self.init_args is not None and key(self.init_args) != key(args):
                raise ValueError('Restart the manager to change H3 attention backend or predictor')
        if args.sp not in [1, 2, 4, 8, 16]:
            raise ValueError(f"--sp must be one of [1, 2, 4, 8, 16], got {args.sp}")

        if args.ulysses_degree is not None and args.ring_degree is not None:
            if args.ulysses_degree * args.ring_degree != args.sp:
                raise ValueError(
                    f"--ulysses_degree ({args.ulysses_degree}) * --ring_degree ({args.ring_degree}) "
                    f"must equal --sp ({args.sp})"
                )
            if args.sp == 16 and args.ulysses_degree == 16:
                raise ValueError(
                    "--ulysses_degree cannot be set to 16 when --sp is 16"
                )
        else:
            if args.ulysses_degree is not None or args.ring_degree is not None:
                raise ValueError(
                    "--ulysses_degree and --ring_degree must be specified together"
                )

        self._prepare_pipeline(args)

        infer_info.update_info(args)
        self.init_args = deepcopy(args)

        try:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory: {args.save_path}")
            return 0.0

        if args.guidance_scale is None or ("Wan2.2" in args.model and args.guidance_scale_2 is None):
            args.guidance_scale, args.guidance_scale_2 = GUIDANCE_SCALE_MAP[args.model]
            logger.info(f"The guidance scale was not set correctly and has been reset to "
                        f"guidance scale={args.guidance_scale}, guidance scale 2={args.guidance_scale_2}.")

        if args.x:
            args.guidance_scale = X_GUIDANCE_SCALE
            if "Wan2.2" in args.model:
                args.guidance_scale_2 = X_GUIDANCE_SCALE

        torch.distributed.barrier()
        torch.cuda.synchronize()
        start_time = time.time()

        self.infer_single(args)

        torch.cuda.synchronize()
        end_time = time.time()
        infer_cost = end_time - start_time
        logger.info(f"time cost:{infer_cost}")

        return infer_cost


class ImageInferenceManager(object):
    def __init__(self):
        self.pipe = None

    @staticmethod
    def load_image(image_path: str):
        if not os.path.exists(image_path):
            raise RuntimeError(f"image path {image_path} is not exists")
        return Image.open(image_path).convert("RGB")

    def set_infer_params(self, args):
        infer_params = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
            generator=torch.Generator().manual_seed(args.seed),
            cfg_parallel_size=args.cfg_parallel_size,
            true_cfg_scale=args.true_cfg_scale,
        )

        if args.guidance_scale is not None:
            infer_params['guidance_scale'] = args.guidance_scale

        # for image to image
        if args.image_path is not None and args.image_path != "":
            image = self.load_image(args.image_path)
            infer_params.update(image=image)
        elif args.image_path_list:
            image = [self.load_image(image_path) for image_path in args.image_path_list]
            infer_params.update(image=image)

        return infer_params

    def infer(self, args) -> float:
        _t_load = time.time()
        self.pipe = load_pipe(args)

        update_scheduler(
            self.pipe,
            args.model,
            args.lora_path_list
        )

        update_lora(
            self.pipe,
            args.model,
            init_lora_path_list=None,
            lora_path_list=args.lora_path_list,
            fuse_lora=args.fuse_lora,
            weights_list=args.lora_scale_weight_list
        )

        update_pipe(self.pipe, args)

        self.pipe = pipe_to_device(self.pipe, args)
        _load_t = time.time() - _t_load
        logger.info(f"[STAGE] load_pipe + setup: {_load_t:.3f}s")

        try:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory: {args.save_path}")
            return 0.0

        infer_params = self.set_infer_params(args)
        # 获取pipe的参数签名
        signature = inspect.signature(self.pipe)
        # 筛选出pipe接受的参数，不同pipeline支持的参数不一样
        filtered_params = {k: v for k, v in infer_params.items() if k in signature.parameters}

        torch.npu.synchronize()
        _infer_t0 = time.time()

        out = self.pipe(**filtered_params)

        torch.npu.synchronize()
        _infer_t = time.time() - _infer_t0
        logger.info(f"[STAGE] pipe inference: {_infer_t:.3f}s")

        images = out["images"] if isinstance(out, dict) else getattr(out, "images", out)
        image = images[0]

        _save_t0 = time.time()
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                image.save(args.save_path)
                logger.info(f"save image in {args.save_path}")
        else:
            image.save(args.save_path)
            logger.info(f"save image in {args.save_path}")
        _save_t = time.time() - _save_t0
        logger.info(f"[STAGE] save_output: {_save_t:.3f}s")

        infer_cost = _infer_t
        logger.info(f"time cost:{infer_cost}")

        return infer_cost


def configure_vae_lightning(args, pipe):
    if args.resolution == 480 and not args.ten_second:
        args.vae_lightning = "decoder"
        pipe.enable_vae_lightning(return_output=False)
    else:
        args.vae_lightning = "none"
        pipe.enable_vae_lightning(return_output=True)


def _get_temp_save_path(original_path: str) -> str:
    """
    生成临时保存路径

    Args:
        original_path: 原始保存路径

    Returns:
        str: 拼接了tmp_前缀的临时路径
    """
    if not original_path:
        raise ValueError("Original save path cannot be empty!")
    TEMP_FILE_PREFIX = "tmp_"
    directory = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    return os.path.join(directory, f"{TEMP_FILE_PREFIX}{filename}")


def save_output_server(
        output,
        original_save_path: str,
        args
) -> None:
    """
    保存推理输出结果，处理不同场景下的视频保存/帧率调整逻辑

    Args:
        output: 模型推理输出的视频张量/数据
        original_save_path: 原始保存路径
        args: 命令行参数对象，包含adopt_sr/ten_second/vae_lightning等配置
    """
    # 提前判断分布式进程是否为主进程（减少重复调用）
    is_main_process = torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True

    # 场景1：启用超分，且是10s场景的主进程 → 调整帧率并返回
    if args.adopt_sr:
        if args.ten_second and is_main_process:
            temp_save_path = _get_temp_save_path(original_save_path)
            InferenceManager().change_fps_filter(original_save_path, temp_save_path, out_fps=args.save_fps)
        return  # 非10s场景/非主进程也直接返回

    # 场景2：非10s + vae_lightning + sp>1 → 直接返回（无需保存）
    if not args.ten_second and args.vae_lightning and args.sp > 1:
        return

    # 场景3：i2v任务 + 自动亮度调整 → 恢复亮度
    if args.task_type == "i2v" and args.ada_brighten:
        output = InferenceManager().restore_brightness(output)

    # 场景4：主进程 + 10s场景 → 补全帧数并保存视频
    if is_main_process and args.ten_second:
        try:
            # 计算目标帧率（向上取整）
            save_fps = math.ceil((args.save_fps * 14) / 16)
            # 导出视频
            export_to_video(output, original_save_path, quality=8, fps=save_fps
                            )
            # 调整最终帧率
            temp_save_path = _get_temp_save_path(original_save_path)
            InferenceManager().change_fps_filter(original_save_path, temp_save_path, out_fps=args.save_fps)
        except Exception as e:
            # 异常捕获，避免单步失败导致整个流程中断
            print(f"Error saving 10s video: {e}", flush=True)
            raise  # 可选：重新抛出异常，让上层感知错误


def load_and_preprocess_image(image_path: str, args) -> Image.Image:
    """
    加载并预处理i2v任务的输入图片
    """
    image = load_image(image_path)
    width, height = args.width, args.height
    image = image.resize((width, height))
    # 亮度预处理
    image = InferenceManager().ada_bright_image_preprocess(image, args)
    return image


def get_pipe_common_kwargs(args, image=None) -> dict:
    """
    生成pipe调用的通用参数
    """
    common_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height if image is None else args.height,
        "width": args.width if image is None else args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    # i2v任务添加image参数，t2v任务不添加
    if image is not None:
        common_kwargs["image"] = image
    return common_kwargs


def run_v2v_extension(base_frames: np.ndarray, args, v2v_pipe) -> np.ndarray:
    """
    执行V2V扩展逻辑
    """
    # 1. 准备V2V扩展的输入
    last_frames = base_frames[-V2V_HISTORY_FRAMES:]
    g_ext = torch.Generator().manual_seed(args.seed)

    # 2. 执行V2V推理
    v2v_kwargs = {
        "conditioning_video": last_frames,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "conditioning_indices": V2V_COND_POS_LIST,
        "conditioning_noise_multipliers": V2V_NOISE_MULT_LIST,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": V2V_INFERENCE_STEPS,
        "generator": g_ext,
        "output_type": V2V_OUTPUT_TYPE,
    }
    v2v_output = v2v_pipe(**v2v_kwargs).frames[0]

    # 3. 拼接基础帧和扩展帧
    output = np.concatenate([base_frames[:-V2V_HISTORY_FRAMES], v2v_output], axis=0)
    return output


def infer_server(args, pipe, sr_pipe, v2v_pipe=None):
    infer_info.update_info(args)
    if args.guidance_scale is None or ("Wan2.2" in args.model and args.guidance_scale_2 is None):
        args.guidance_scale, args.guidance_scale_2 = GUIDANCE_SCALE_MAP[args.model]
        logger.info(f"The guidance scale was not set correctly and has been reset to "
                    f"guidance scale={args.guidance_scale}, guidance scale 2={args.guidance_scale_2}.")
    logger.info("start infer_server")
    torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()

    directory = os.path.dirname(args.save_path)
    filename = os.path.basename(args.save_path)
    save_path = os.path.join(directory, "tmp_" + filename)
    infer_info.save_path = save_path

    configure_vae_lightning(args, pipe)

    if args.task_type == "t2v":
        # T2V任务：基础推理
        pipe_kwargs = get_pipe_common_kwargs(args)
        base_output = pipe(**pipe_kwargs).frames[0]

        if args.ten_second:
            # 10s场景：执行V2V扩展
            infer_info.update_info(args)
            output = run_v2v_extension(base_output, args, v2v_pipe)
        else:
            # 非10s场景：直接使用基础输出
            output = base_output

    else:
        # I2V任务：先加载预处理图片
        image = load_and_preprocess_image(args.i2v_image_path, args)
        pipe_kwargs = get_pipe_common_kwargs(args, image=image)
        base_output = pipe(**pipe_kwargs).frames[0]

        if args.ten_second:
            # 10s场景：基础推理 + V2V扩展
            infer_info.update_info(args)
            output = run_v2v_extension(base_output, args, v2v_pipe)
        else:
            # 非10s场景：直接推理
            output = base_output
    pipe.text_encoder = pipe.text_encoder.to("cpu")
    if args.ten_second:
        if args.resolution == 1080:
            infer_info.fps = 14
        else:
            output = InferenceManager().frame_interpolation(args, output)
            infer_info.fps = math.ceil((args.save_fps * 14) / 16)
    if args.resolution in [720, 1080] and sr_pipe is not None:
        output = sr_pipe(output)
    if args.ada_brighten and not args.vae_lightning and args.resolution == 480:
        output = InferenceManager().restore_brightness(output)
    save_output_server(output, args.save_path, args)
    pipe.text_encoder = pipe.text_encoder.to("npu")
    del output
    del base_output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    end_time = time.time()
    infer_cost = end_time - start_time
    logger.info(f"time cost:{infer_cost}")

    return infer_cost
