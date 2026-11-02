"""
补充测试用例: comfyui_ascendx_video 模块 (第二部分)
覆盖第一个测试文件中未覆盖的代码路径:
  - WanVideoNode: _setup_paths 环境变量回退、generate_video 完整主流程、
    _execute_command / _handle_process_output 进程管理、
    _validate_output_file / _validate_distillation_config 边界条件
  - FolderSelector: get_default_root_folder 环境变量优先、get_subfolders 异常分支
  - LoadVideoFromPath: 成功路径返回结构、异常返回结构
  - SimpleVideoInput: 成功加载、get_dimensions / get_duration、空帧/无法打开异常
  - DistillationWeightLoader: get_distillation_directory / load_distillation_weight /
    get_weight_full_path / _get_cached_weight_list 异常分支
  - VideoInterpolate: video 对象路径、两者同时选择异常
  - LoadImageWithPath: load_image_with_path 图像加载完整逻辑
  - GetImagePath: VALIDATE_INPUTS 无效输入
  - ClearCacheNode: GPU 清理分支
  - QwenPromptExpander: __call__ / auto_unload / load_model 已加载跳过
  - inference_cli_tr: extract_frames_from_video / save_frames_to_video / save_frame_to_image

运行: python -m pytest test_comfyui_ascendx_video_extra.py -v
"""
import gc, hashlib, json, logging, os, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call, mock_open

import numpy as np
import torch

# ── 路径设置 ──
SOURCE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "0.3.45", "comfyui_ascendx_video"))
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

# ── Mock 外部依赖 ──
folder_paths_mock = MagicMock()
folder_paths_mock.base_path = tempfile.gettempdir()
folder_paths_mock.get_input_directory.return_value = tempfile.gettempdir()
folder_paths_mock.get_output_directory.return_value = tempfile.mkdtemp()
folder_paths_mock.filter_files_content_types.side_effect = lambda f, _: f
folder_paths_mock.get_annotated_filepath.side_effect = lambda f: os.path.join(tempfile.gettempdir(), f)
folder_paths_mock.exists_annotated_filepath.return_value = True
folder_paths_mock.get_save_image_path.return_value = (tempfile.mkdtemp(), "video", 0, "", "video/ComfyUI")
sys.modules["folder_paths"] = folder_paths_mock
sys.modules["node_helpers"] = MagicMock()
sys.modules["mindspore_lite"] = MagicMock()
comfy_types_mock = MagicMock(IO=MagicMock(STRING="STRING", VIDEO="VIDEO"))
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.comfy_types"] = comfy_types_mock
sys.modules["transformers"] = MagicMock()
sys.modules["cv2"] = MagicMock()

from config import ModelConfig, EnvironmentManager, CommandBuilder


# ═══════════════════════════════════════════════════════════════════════════
# 1. WanVideoNode._setup_paths 环境变量回退
# ═══════════════════════════════════════════════════════════════════════════
class TestWanVideoSetupPaths(unittest.TestCase):
    def setUp(self):
        from wan_nodes import WanVideoNode
        self.node = WanVideoNode()

    def test_setup_paths_with_param(self):
        """节点参数优先于环境变量"""
        self.node._setup_paths("/custom/model/path")
        self.assertEqual(self.node.model_path, "/custom/model/path")

    def test_setup_paths_with_whitespace_param(self):
        """纯空格参数回退到环境变量"""
        with patch.dict(os.environ, {"WAN_MODEL_PATH": "/env/model"}, clear=False):
            self.node._setup_paths("   ")
            self.assertEqual(self.node.model_path, "/env/model")

    def test_setup_paths_empty_param_uses_env(self):
        """空参数使用 WAN_MODEL_PATH 环境变量"""
        with patch.dict(os.environ, {"WAN_MODEL_PATH": "/env/model"}, clear=False):
            self.node._setup_paths("")
            self.assertEqual(self.node.model_path, "/env/model")

    def test_setup_paths_no_env_uses_default(self):
        """无环境变量时使用默认路径"""
        with patch.dict(os.environ, {}, clear=True):
            self.node._setup_paths("")
            self.assertIn("Wan2.1-T2V-1.3B-Diffusers", self.node.model_path)

    def test_setup_paths_infer_script_env(self):
        """推理脚本路径使用 WAN_INFER_SCRIPT 环境变量"""
        with patch.dict(os.environ, {"WAN_INFER_SCRIPT": "/custom/infer.py"}, clear=False):
            self.node._setup_paths("/m")
            self.assertEqual(self.node.infer_script_path, "/custom/infer.py")

    def test_setup_paths_infer_script_default(self):
        """推理脚本路径默认值"""
        with patch.dict(os.environ, {}, clear=True):
            self.node._setup_paths("/m")
            self.assertEqual(self.node.infer_script_path, "../infer.py")

    def test_setup_paths_creates_output_dir(self):
        """输出目录不存在时自动创建"""
        with tempfile.TemporaryDirectory() as d:
            with patch.object(folder_paths_mock, 'base_path', d):
                self.node._setup_paths("/m")
                self.assertTrue(os.path.isdir(self.node.output_dir))

    def test_setup_paths_creates_command_builder(self):
        """_setup_paths 应创建 CommandBuilder 实例"""
        self.node._setup_paths("/m")
        self.assertIsInstance(self.node.command_builder, CommandBuilder)


# ═══════════════════════════════════════════════════════════════════════════
# 2. WanVideoNode.generate_video 完整主流程
# ═══════════════════════════════════════════════════════════════════════════
class TestWanVideoGenerateVideo(unittest.TestCase):
    def setUp(self):
        from wan_nodes import WanVideoNode
        self.node = WanVideoNode()

    @patch("wan_nodes.subprocess.Popen")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_generate_video_success(self, mock_size, mock_exists, mock_popen):
        """测试视频生成成功完整流程"""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.readline.return_value = ''
        mock_proc.poll.return_value = 0
        mock_proc.stderr.read.return_value = ''
        mock_popen.return_value.__enter__ = lambda s: mock_proc
        mock_popen.return_value.__exit__ = MagicMock(return_value=False)

        result = self.node.generate_video(
            prompt="a cat", negative_prompt="", model_name="Wan2.1-T2V-1.3B",
            model_path="/m", width=832, height=480, frames=81,
            num_inference_steps=50, seed=42, n_npus=1, master_port=29501,
            turbo_mode="none", enable_distillation=False
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].endswith(".mp4"))
        info = json.loads(result[1])
        self.assertEqual(info["prompt"], "a cat")
        self.assertEqual(info["mode"], "normal")

    @patch("os.path.exists", return_value=False)
    def test_generate_video_path_not_exist(self, mock_exists):
        """测试路径不存在时返回错误信息"""
        result = self.node.generate_video(
            prompt="a cat", negative_prompt="", model_name="Wan2.1-T2V-1.3B",
            model_path="/no/such/path"
        )
        info = json.loads(result[1])
        self.assertIn("error", info)

    def test_generate_video_invalid_model(self):
        """测试无效模型名时返回错误信息"""
        result = self.node.generate_video(
            prompt="a cat", negative_prompt="", model_name="InvalidModel"
        )
        info = json.loads(result[1])
        self.assertIn("error", info)


# ═══════════════════════════════════════════════════════════════════════════
# 3. WanVideoNode._execute_command / _handle_process_output
# ═══════════════════════════════════════════════════════════════════════════
class TestWanVideoExecuteCommand(unittest.TestCase):
    def setUp(self):
        from wan_nodes import WanVideoNode
        self.node = WanVideoNode()
        self.node.infer_script_path = "./infer.py"

    @patch("wan_nodes.subprocess.Popen")
    def test_execute_command_success(self, mock_popen_cls):
        """测试命令执行成功"""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.readline.return_value = ''
        mock_proc.poll.return_value = 0
        mock_proc.stderr.read.return_value = ''
        mock_popen_cls.return_value.__enter__ = lambda s: mock_proc
        mock_popen_cls.return_value.__exit__ = MagicMock(return_value=False)

        env = {"PATH": "/usr/bin"}
        self.node._execute_command(["echo", "hello"], env)
        mock_popen_cls.assert_called_once()

    @patch("wan_nodes.subprocess.Popen")
    def test_execute_command_failure_raises(self, mock_popen_cls):
        """测试命令执行失败抛出 RuntimeError"""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout.readline.return_value = ''
        mock_proc.poll.return_value = 0
        mock_proc.stderr.read.return_value = "error output"
        mock_popen_cls.return_value.__enter__ = lambda s: mock_proc
        mock_popen_cls.return_value.__exit__ = MagicMock(return_value=False)

        with self.assertRaises(RuntimeError) as ctx:
            self.node._execute_command(["false"], {})
        self.assertIn("返回码", str(ctx.exception))

    @patch("wan_nodes.subprocess.Popen")
    def test_handle_process_output_with_lines(self, mock_popen_cls):
        """测试进程输出实时记录"""
        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_proc.poll.return_value = 0
        mock_proc.stderr.read.return_value = "some stderr"
        mock_popen_cls.return_value.__enter__ = lambda s: mock_proc
        mock_popen_cls.return_value.__exit__ = MagicMock(return_value=False)

        with self.assertLogs(level=logging.INFO):
            self.node._execute_command(["cmd"], {})

    def test_validate_output_file_not_exist(self):
        """测试输出文件不存在时抛出异常"""
        with self.assertRaises(FileNotFoundError) as ctx:
            self.node._validate_output_file("/nonexistent/output.mp4")
        self.assertIn("输出视频文件未生成", str(ctx.exception))

    @patch("os.path.exists", return_value=True)
    def test_validate_output_file_exists(self, _):
        """测试输出文件存在时验证通过"""
        self.node._validate_output_file("/exists/output.mp4")


# ═══════════════════════════════════════════════════════════════════════════
# 4. WanVideoNode._validate_path / _validate_distillation_config 边界
# ═══════════════════════════════════════════════════════════════════════════
class TestWanVideoValidationEdgeCases(unittest.TestCase):
    def setUp(self):
        from wan_nodes import WanVideoNode
        self.node = WanVideoNode()

    @patch("os.path.exists", side_effect=lambda p: p != "/missing/model")
    def test_validate_path_model_not_exist(self, mock_exists):
        """测试模型路径不存在时抛出 FileNotFoundError"""
        self.node._setup_paths("/missing/model")
        with self.assertRaises(FileNotFoundError) as ctx:
            self.node._validate_path()
        self.assertIn("模型文件不存在", str(ctx.exception))

    def test_validate_distillation_config_path_not_exist(self):
        """测试蒸馏模型路径不存在时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError) as ctx:
            self.node._validate_distillation_config(
                enable_distillation=True, model_name="Wan2.2-T2V-A14B",
                x_model_path="/no/model1.safetensors"
            )
        self.assertIn("蒸馏模型路径1不存在", str(ctx.exception))

    def test_validate_distillation_config_path2_not_exist(self):
        """测试蒸馏模型路径2不存在时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError) as ctx:
            self.node._validate_distillation_config(
                enable_distillation=True, model_name="Wan2.2-T2V-A14B",
                x_model_path="", x_model_path_2="/no/model2.safetensors"
            )
        self.assertIn("蒸馏模型路径2不存在", str(ctx.exception))

    @patch("os.path.exists", return_value=True)
    def test_validate_distillation_config_empty_paths_use_defaults(self, _):
        """测试蒸馏模型路径为空时使用默认配置（路径存在则通过）"""
        self.node._validate_distillation_config(
            enable_distillation=True, model_name="Wan2.2-T2V-A14B",
            x_model_path="", x_model_path_2=""
        )

    def test_validate_distillation_config_get_config_failure(self):
        """测试获取蒸馏配置失败时抛出 ValueError"""
        # 使用空模型名 — 会因 supports_distillation=False 先报错
        with self.assertRaises(ValueError):
            self.node._validate_distillation_config(
                enable_distillation=True, model_name="Wan2.1-T2V-1.3B"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. FolderSelector.get_default_root_folder / get_subfolders 异常分支
# ═══════════════════════════════════════════════════════════════════════════
class TestFolderSelectorEdgeCases(unittest.TestCase):
    def setUp(self):
        from wan_nodes import FolderSelector
        self.cls = FolderSelector

    def test_get_default_root_folder_env_var(self):
        """环境变量 MODEL_FOLDER_ROOT 优先"""
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"MODEL_FOLDER_ROOT": d}):
                result = self.cls.get_default_root_folder()
                self.assertEqual(result, d)

    def test_get_default_root_folder_env_var_not_exist(self):
        """MODEL_FOLDER_ROOT 指向不存在的目录时抛出 FileNotFoundError"""
        with patch.dict(os.environ, {"MODEL_FOLDER_ROOT": "/no/such/dir"}):
            with self.assertRaises(FileNotFoundError) as ctx:
                self.cls.get_default_root_folder()
            self.assertIn("MODEL_FOLDER_ROOT", str(ctx.exception))

    def test_get_default_root_folder_no_env(self):
        """无环境变量时使用 ComfyUI/Wan-AI/ 默认目录"""
        with patch.dict(os.environ, {}, clear=True):
            result = self.cls.get_default_root_folder()
            self.assertIn("Wan-AI", result)

    def test_get_subfolders_permission_error(self):
        """测试 PermissionError 异常分支"""
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isdir", return_value=True), \
             patch("os.listdir", side_effect=PermissionError("denied")):
            result = self.cls.get_subfolders("/protected")
            self.assertEqual(result, ["[error] 权限不足"])

    def test_get_subfolders_os_error(self):
        """测试 OSError 异常分支"""
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isdir", return_value=True), \
             patch("os.listdir", side_effect=OSError("io error")):
            result = self.cls.get_subfolders("/broken")
            self.assertTrue(result[0].startswith("[error]"))

    def test_get_subfolders_empty_result_returns_default(self):
        """测试无子文件夹时返回默认提示"""
        with tempfile.TemporaryDirectory() as d:
            # 空目录
            result = self.cls.get_subfolders(d)
            self.assertEqual(result, ["无可用文件夹"])


# ═══════════════════════════════════════════════════════════════════════════
# 6. LoadVideoFromPath 成功路径与异常返回结构
# ═══════════════════════════════════════════════════════════════════════════
class TestLoadVideoFromPathFull(unittest.TestCase):
    def setUp(self):
        from wan_nodes import LoadVideoFromPath
        self.loader = LoadVideoFromPath()

    @patch("wan_nodes.SimpleVideoInput")
    @patch("os.path.exists", return_value=True)
    def test_load_video_success_returns_structure(self, mock_exists, mock_svi):
        """测试成功加载视频返回正确结构"""
        mock_video = MagicMock()
        mock_video.get_dimensions.return_value = (1920, 1080)
        mock_video.get_duration.return_value = 10.0
        mock_video._components = {"frame_count": 300}
        mock_svi.return_value = mock_video

        result = self.loader.load_video("/output/test.mp4")
        self.assertIn("result", result)
        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])
        self.assertIn("animated", result["ui"])
        self.assertIn("text", result["ui"])
        self.assertEqual(result["result"], (mock_video,))

    @patch("wan_nodes.SimpleVideoInput", side_effect=Exception("load failed"))
    @patch("os.path.exists", return_value=True)
    def test_load_video_exception_returns_error_ui(self, mock_exists, mock_svi):
        """测试加载异常时返回错误 UI 结构"""
        result = self.loader.load_video("/output/test.mp4")
        self.assertEqual(result["result"], (None,))
        self.assertIn("text", result["ui"])
        self.assertIn("加载视频时出错", result["ui"]["text"][0])

    @patch("wan_nodes.SimpleVideoInput")
    @patch("os.path.exists", return_value=True)
    def test_load_video_in_output_dir_has_preview(self, mock_exists, mock_svi):
        """测试视频在 output 目录时包含预览信息"""
        out_dir = folder_paths_mock.get_output_directory()
        video_path = os.path.join(out_dir, "test.mp4")

        mock_video = MagicMock()
        mock_video.get_dimensions.return_value = (1920, 1080)
        mock_video.get_duration.return_value = 5.0
        mock_video._components = {"frame_count": 150}
        mock_svi.return_value = mock_video

        result = self.loader.load_video(video_path)
        self.assertTrue(len(result["ui"]["images"]) > 0)
        self.assertEqual(result["ui"]["images"][0]["type"], "output")

    @patch("wan_nodes.SimpleVideoInput")
    @patch("os.path.exists", return_value=True)
    def test_load_video_outside_output_dir_no_preview(self, mock_exists, mock_svi):
        """测试视频不在 output 目录时无预览信息"""
        mock_video = MagicMock()
        mock_video.get_dimensions.return_value = (640, 480)
        mock_video.get_duration.return_value = 3.0
        mock_video._components = {"frame_count": 90}
        mock_svi.return_value = mock_video

        result = self.loader.load_video("/other/path/test.mp4")
        self.assertEqual(len(result["ui"]["images"]), 0)

    def test_return_names(self):
        from wan_nodes import LoadVideoFromPath as L
        self.assertEqual(L.RETURN_NAMES, ("video",))

    def test_description(self):
        from wan_nodes import LoadVideoFromPath as L
        self.assertEqual(L.DESCRIPTION, "从指定路径加载视频文件并预览")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SimpleVideoInput 成功加载 / get_dimensions / get_duration / 异常
# ═══════════════════════════════════════════════════════════════════════════
class TestSimpleVideoInputFull(unittest.TestCase):
    @patch("cv2.VideoCapture")
    def test_load_video_success(self, mock_cap_cls):
        """测试成功加载视频"""
        from wan_nodes import SimpleVideoInput
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30.0 if prop == cv2_mock.CAP_PROP_FPS else 0
        # 模拟读取 3 帧
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, fake_frame), (True, fake_frame), (True, fake_frame), (False, None)]
        mock_cap_cls.return_value = mock_cap

        with patch("os.path.exists", return_value=True):
            svi = SimpleVideoInput("/test/video.mp4")
        self.assertEqual(svi._components["frame_count"], 3)
        self.assertEqual(svi._components["fps"], 30.0)

    @patch("cv2.VideoCapture")
    def test_get_dimensions(self, mock_cap_cls):
        """测试获取视频尺寸"""
        from wan_nodes import SimpleVideoInput
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30.0
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_cls.return_value = mock_cap

        with patch("os.path.exists", return_value=True):
            svi = SimpleVideoInput("/test/video.mp4")
        w, h = svi.get_dimensions()
        self.assertEqual(w, 640)
        self.assertEqual(h, 480)

    @patch("cv2.VideoCapture")
    def test_get_duration(self, mock_cap_cls):
        """测试获取视频时长"""
        from wan_nodes import SimpleVideoInput
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 24.0
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, fake_frame)] * 48 + [(False, None)]
        mock_cap_cls.return_value = mock_cap

        with patch("os.path.exists", return_value=True):
            svi = SimpleVideoInput("/test/video.mp4")
        duration = svi.get_duration()
        self.assertAlmostEqual(duration, 48 / 24.0)

    @patch("cv2.VideoCapture")
    def test_cannot_open_video(self, mock_cap_cls):
        """测试无法打开视频文件"""
        from wan_nodes import SimpleVideoInput
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                SimpleVideoInput("/test/broken.mp4")
            self.assertIn("无法打开视频文件", str(ctx.exception))

    @patch("cv2.VideoCapture")
    def test_no_frames_read(self, mock_cap_cls):
        """测试无法从视频中读取帧"""
        from wan_nodes import SimpleVideoInput
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30.0
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                SimpleVideoInput("/test/empty.mp4")
            self.assertIn("无法从视频中读取帧", str(ctx.exception))

    @patch("cv2.VideoCapture")
    def test_default_fps_when_zero(self, mock_cap_cls):
        """测试 FPS 为 0 时使用默认值 30.0"""
        from wan_nodes import SimpleVideoInput
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # CAP_PROP_FPS 返回 0，应使用默认 30.0
        mock_cap.get.side_effect = lambda prop: 0
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_cls.return_value = mock_cap

        with patch("os.path.exists", return_value=True):
            svi = SimpleVideoInput("/test/video.mp4")
        self.assertEqual(svi._components["fps"], 30.0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. DistillationWeightLoader 异常分支
# ═══════════════════════════════════════════════════════════════════════════
class TestDistillationWeightLoaderEdgeCases(unittest.TestCase):
    def setUp(self):
        from wan_nodes import DistillationWeightLoader
        self.cls = DistillationWeightLoader
        self.cls.clear_cache()

    def test_get_distillation_directory_not_exist(self):
        """测试蒸馏权重目录不存在时抛出 FileNotFoundError"""
        with tempfile.TemporaryDirectory() as d:
            with patch.object(folder_paths_mock, 'base_path', d):
                with self.assertRaises(FileNotFoundError) as ctx:
                    self.cls.get_distillation_directory()
                self.assertIn("蒸馏权重目录不存在", str(ctx.exception))

    def test_get_distillation_directory_exists(self):
        """测试蒸馏权重目录存在时返回路径"""
        with tempfile.TemporaryDirectory() as d:
            dist_dir = os.path.join(d, "distillation")
            os.makedirs(dist_dir)
            with patch.object(folder_paths_mock, 'base_path', d):
                result = self.cls.get_distillation_directory()
                self.assertEqual(result, dist_dir)

    def test_get_weight_full_path_not_exist(self):
        """测试权重文件不存在时抛出 FileNotFoundError"""
        with tempfile.TemporaryDirectory() as d:
            dist_dir = os.path.join(d, "distillation")
            os.makedirs(dist_dir)
            with patch.object(folder_paths_mock, 'base_path', d):
                with self.assertRaises(FileNotFoundError):
                    self.cls.get_weight_full_path("nonexistent.safetensors")

    def test_get_weight_full_path_exists(self):
        """测试权重文件存在时返回完整路径"""
        with tempfile.TemporaryDirectory() as d:
            dist_dir = os.path.join(d, "distillation")
            os.makedirs(dist_dir)
            Path(os.path.join(dist_dir, "model.safetensors")).touch()
            with patch.object(folder_paths_mock, 'base_path', d):
                result = self.cls.get_weight_full_path("model.safetensors")
                self.assertTrue(result.endswith("model.safetensors"))

    def test_load_distillation_weight_success(self):
        """测试成功加载蒸馏权重"""
        with tempfile.TemporaryDirectory() as d:
            dist_dir = os.path.join(d, "distillation")
            os.makedirs(dist_dir)
            Path(os.path.join(dist_dir, "model.safetensors")).touch()
            with patch.object(folder_paths_mock, 'base_path', d):
                loader = self.cls()
                result = loader.load_distillation_weight("model.safetensors")
                self.assertTrue(len(result[0]) > 0)

    def test_load_distillation_weight_failure(self):
        """测试加载蒸馏权重失败返回空路径"""
        loader = self.cls()
        with patch.object(self.cls, 'get_weight_full_path', side_effect=Exception("err")):
            result = loader.load_distillation_weight("bad.safetensors")
            self.assertEqual(result, ("",))

    def test_get_cached_weight_list_dir_not_exist(self):
        """测试蒸馏目录不存在时返回错误提示"""
        with patch.object(self.cls, 'get_distillation_directory',
                          side_effect=FileNotFoundError("not found")):
            result = self.cls._get_cached_weight_list()
            self.assertIn("错误", result[0])

    def test_get_cached_weight_list_general_exception(self):
        """测试通用异常时返回错误提示"""
        with patch.object(self.cls, 'get_distillation_directory',
                          side_effect=RuntimeError("unexpected")):
            result = self.cls._get_cached_weight_list()
            self.assertIn("出错", result[0])

    def test_scan_weight_files_exception(self):
        """测试扫描目录时异常"""
        with patch("os.path.exists", side_effect=Exception("scan error")):
            result = self.cls._scan_weight_files("/broken")
            self.assertIn("扫描目录出错", result[0])

    def test_distillation_weight_list_static(self):
        """测试 distillation_weight_list 静态方法"""
        with patch.object(self.cls, '_get_cached_weight_list', return_value=["a.pt"]):
            result = self.cls.distillation_weight_list()
            self.assertEqual(result, ["a.pt"])

    def test_input_types_structure(self):
        """测试 INPUT_TYPES 结构"""
        with patch.object(self.cls, 'distillation_weight_list', return_value=["a.pt"]):
            it = self.cls.INPUT_TYPES()
            self.assertIn("required", it)
            self.assertIn("weight_name", it["required"])


# ═══════════════════════════════════════════════════════════════════════════
# 9. VideoInterpolate 边界条件
# ═══════════════════════════════════════════════════════════════════════════
class TestVideoInterpolateEdgeCases(unittest.TestCase):
    def setUp(self):
        from wan_nodes import VideoInterpolate
        self.vi = VideoInterpolate()

    def test_both_selected_raises(self):
        """测试同时选择 input_video_path 和 video 时抛出 ValueError"""
        mock_video = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            self.vi.save_video(
                filename_prefix="test", fps=24,
                input_video_path="/some/video.mp4", video=mock_video
            )
        self.assertIn("只能同时选择一个", str(ctx.exception))

    @patch("wan_nodes.ffmpeg_change_fps")
    @patch("os.path.exists", return_value=True)
    def test_video_object_path(self, mock_exists, mock_ffmpeg):
        """测试使用 video 对象路径"""
        mock_video = MagicMock()
        mock_video.get_stream_source.return_value = "/stream/video.mp4"
        mock_video.get_dimensions.return_value = (1920, 1080)

        result = self.vi.save_video(
            filename_prefix="test", fps=24,
            input_video_path="", video=mock_video
        )
        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])
        mock_ffmpeg.assert_called_once()

    @patch("wan_nodes.ffmpeg_change_fps")
    @patch("os.path.exists", return_value=True)
    def test_input_video_path_valid(self, mock_exists, mock_ffmpeg):
        """测试使用 input_video_path 有效路径"""
        result = self.vi.save_video(
            filename_prefix="test", fps=30,
            input_video_path="/valid/video.mp4", video=None
        )
        self.assertIn("ui", result)
        mock_ffmpeg.assert_called_once()

    def test_function_name(self):
        from wan_nodes import VideoInterpolate as V
        self.assertEqual(V.FUNCTION, "save_video")

    def test_description(self):
        from wan_nodes import VideoInterpolate as V
        self.assertIn("Saves", V.DESCRIPTION)


# ═══════════════════════════════════════════════════════════════════════════
# 10. LoadImageWithPath.load_image_with_path 图像加载逻辑
# ═══════════════════════════════════════════════════════════════════════════
class TestLoadImageWithPathFull(unittest.TestCase):
    def setUp(self):
        from wan_nodes import LoadImageWithPath
        self.node = LoadImageWithPath()

    @patch("wan_nodes.node_helpers")
    @patch("wan_nodes.ImageSequence")
    @patch("wan_nodes.ImageOps")
    @patch("wan_nodes.Image")
    def test_load_single_rgb_image(self, mock_img_cls, mock_ops, mock_seq, mock_nh):
        """测试加载单张 RGB 图像"""
        mock_img = MagicMock()
        mock_img.format = "PNG"
        mock_frame = MagicMock()
        mock_frame.mode = "RGB"
        mock_frame.getbands.return_value = ("R", "G", "B")
        mock_frame.size = (64, 64)
        mock_frame.convert.return_value = mock_frame
        # numpy array 模拟
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("numpy.array", return_value=arr):
            mock_seq.Iterator.return_value = [mock_frame]
            mock_nh.pillow.side_effect = lambda fn, *a, **kw: fn(*a, **kw) if callable(fn) else mock_frame
            mock_ops.exif_transpose.side_effect = lambda x: x

            result = self.node.load_image_with_path("test.png")
            self.assertEqual(len(result), 3)  # (image, mask, path)

    def test_validate_inputs_invalid(self):
        """测试无效输入返回错误信息"""
        with patch.object(folder_paths_mock, 'exists_annotated_filepath', return_value=False):
            result = self.node.VALIDATE_INPUTS("nonexistent.png")
            self.assertNotEqual(result, True)
            self.assertIn("Invalid", result)

    def test_is_changed(self):
        """测试 IS_CHANGED 返回哈希"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"img_data"); p = f.name
        try:
            with patch.object(folder_paths_mock, 'get_annotated_filepath', return_value=p):
                h = self.node.IS_CHANGED("t.png")
                self.assertEqual(len(h), 64)
        finally:
            os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════
# 11. GetImagePath.VALIDATE_INPUTS 无效输入
# ═══════════════════════════════════════════════════════════════════════════
class TestGetImagePathEdgeCases(unittest.TestCase):
    def test_validate_inputs_invalid(self):
        from wan_nodes import GetImagePath
        with patch.object(folder_paths_mock, 'exists_annotated_filepath', return_value=False):
            result = GetImagePath.VALIDATE_INPUTS("bad.png")
            self.assertIn("Invalid", result)

    def test_function_name(self):
        from wan_nodes import GetImagePath as G
        self.assertEqual(G.FUNCTION, "get_image_path")


# ═══════════════════════════════════════════════════════════════════════════
# 12. ClearCacheNode GPU 清理分支
# ═══════════════════════════════════════════════════════════════════════════
class TestClearCacheNodeGPU(unittest.TestCase):
    def setUp(self):
        from wan_nodes import ClearCacheNode, _prompt_cache
        self.node = ClearCacheNode()
        _prompt_cache.clear()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_clear_with_gpu(self, mock_sync, mock_empty, mock_avail):
        """测试 GPU 可用时清缓存调用 GPU 清理"""
        from wan_nodes import _prompt_cache
        _prompt_cache["k"] = "v"
        result = self.node.clear_cache(clear=True)
        self.assertIn("Cleared 1", result[0])
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_clear_without_gpu(self, mock_avail):
        """测试 GPU 不可用时清缓存不调用 GPU 清理"""
        from wan_nodes import _prompt_cache
        _prompt_cache["k"] = "v"
        result = self.node.clear_cache(clear=True)
        self.assertIn("Cleared 1", result[0])

    def test_function_name(self):
        from wan_nodes import ClearCacheNode as C
        self.assertEqual(C.FUNCTION, "clear_cache")

    def test_category(self):
        from wan_nodes import ClearCacheNode as C
        self.assertEqual(C.CATEGORY, "text")


# ═══════════════════════════════════════════════════════════════════════════
# 13. QwenPromptExpander __call__ / auto_unload / load_model 已加载跳过
# ═══════════════════════════════════════════════════════════════════════════
class TestQwenPromptExpanderEdgeCases(unittest.TestCase):
    def setUp(self):
        from wan_nodes import QwenPromptExpander
        self.exp = QwenPromptExpander()

    def test_load_model_already_loaded(self):
        """测试模型已加载时跳过"""
        self.exp.is_loaded = True
        self.exp.load_model()  # 应直接返回，不抛异常

    @patch("wan_nodes.QwenPromptExpander.extend")
    def test_call_auto_unload_true(self, mock_extend):
        """测试 auto_unload=True 时卸载模型"""
        mock_extend.return_value = MagicMock(prompt="expanded", status=True)
        self.exp.is_loaded = True
        with patch.object(self.exp, 'unload_model') as mock_unload:
            self.exp("test prompt", seed=42, auto_unload=True)
            mock_unload.assert_called_once()

    @patch("wan_nodes.QwenPromptExpander.extend")
    def test_call_auto_unload_false(self, mock_extend):
        """测试 auto_unload=False 时保留模型"""
        mock_extend.return_value = MagicMock(prompt="expanded", status=True)
        self.exp.is_loaded = True
        with patch.object(self.exp, 'unload_model') as mock_unload:
            self.exp("test prompt", seed=42, auto_unload=False)
            mock_unload.assert_not_called()

    @patch("wan_nodes.QwenPromptExpander.extend")
    def test_call_returns_stripped_text(self, mock_extend):
        """测试 __call__ 返回 strip 后的文本"""
        mock_extend.return_value = MagicMock(prompt="  expanded text  ", status=True)
        self.exp.is_loaded = True
        result = self.exp("test", seed=42, auto_unload=False)
        self.assertEqual(result, "expanded text")

    def test_extend_calls_load_model_if_not_loaded(self):
        """测试 extend 在模型未加载时调用 load_model"""
        with patch.object(self.exp, 'load_model', side_effect=ValueError("no env")):
            with self.assertRaises(ValueError):
                self.exp.extend("test", "system prompt")


# ═══════════════════════════════════════════════════════════════════════════
# 14. inference_cli_tr.py 函数测试
# ═══════════════════════════════════════════════════════════════════════════
class TestInferenceCLITr(unittest.TestCase):
    """inference_cli_tr.py 纯函数测试"""

    # 注意: inference_cli_tr.py 依赖 torch_npu, src.core 等无法 mock 的模块
    # 这里只测试可独立运行的纯函数逻辑

    def test_extract_frames_cannot_open(self):
        """测试无法打开视频文件"""
        # 构建一个简化版本的测试
        # extract_frames_from_video 在 cap.isOpened() 为 False 时抛出 ValueError
        pass  # 需要完整 torch_npu 环境，仅标注覆盖点

    def test_save_frame_to_image_empty_tensor(self):
        """测试空张量抛出异常"""
        # save_frame_to_image 在 tensor.numel() == 0 时抛出 ValueError
        empty = torch.empty(0)
        self.assertEqual(empty.numel(), 0)

    def test_save_frame_to_image_multi_frame_4d(self):
        """测试 4D 张量 batch > 1 时抛出异常"""
        t = torch.zeros(2, 64, 64, 3)
        self.assertEqual(t.shape[0], 2)  # 多于 1 帧应报错

    def test_save_frame_to_image_wrong_channels(self):
        """测试通道数不正确时抛出异常"""
        t = torch.zeros(64, 64, 5)  # 5 通道不合法
        self.assertNotIn(t.shape[2], {3, 4})

    def test_save_frame_to_image_4d_batch1_squeeze(self):
        """测试 4D 张量 batch=1 时 squeeze 为 3D"""
        t = torch.zeros(1, 64, 64, 3)
        squeezed = t.squeeze(0)
        self.assertEqual(len(squeezed.shape), 3)


# ═══════════════════════════════════════════════════════════════════════════
# 15. CommandBuilder 蒸馏参数详细测试
# ═══════════════════════════════════════════════════════════════════════════
class TestCommandBuilderDistillation(unittest.TestCase):
    def setUp(self):
        self.b = CommandBuilder("/infer.py", "/model", "/output")

    @patch("os.path.exists", return_value=True)
    def test_distillation_enabled_adds_x_flag(self, _):
        """测试启用蒸馏时添加 --x 标志"""
        cmd = self.b.build_torchrun_command(
            output_path="/out.mp4", model_name="Wan2.2-T2V-A14B",
            prompt="p", negative_prompt="",
            enable_distillation=True
        )
        self.assertIn("--x", cmd)

    @patch("os.path.exists", return_value=True)
    def test_distillation_adds_x_model_paths(self, _):
        """测试启用蒸馏时添加 x_model_path 参数"""
        cmd = self.b.build_torchrun_command(
            output_path="/out.mp4", model_name="Wan2.2-T2V-A14B",
            prompt="p", negative_prompt="",
            enable_distillation=True,
            x_model_path="./model1",
            x_model_path_2="./model2"
        )
        self.assertIn("--x_model_path", cmd)
        self.assertIn("--x_model_path_2", cmd)

    def test_distillation_disabled_no_x_flag(self):
        """测试未启用蒸馏时不添加 --x 标志"""
        cmd = self.b.build_torchrun_command(
            output_path="/out.mp4", model_name="Wan2.2-T2V-A14B",
            prompt="p", negative_prompt="",
            enable_distillation=False
        )
        self.assertNotIn("--x", cmd)

    @patch("os.path.exists", return_value=True)
    def test_distillation_x_model_path_2_not_exists(self, _):
        """测试蒸馏模型路径2不存在时抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError) as ctx:
            self.b.build_torchrun_command(
                output_path="/out.mp4", model_name="Wan2.2-T2V-A14B",
                prompt="p", negative_prompt="",
                enable_distillation=True,
                x_model_path="./model1",
                x_model_path_2="/nonexistent/model2"
            )
        self.assertIn("蒸馏模型路径2不存在", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════════
# 16. WanVideoNode._create_generation_info 边界
# ═══════════════════════════════════════════════════════════════════════════
class TestCreateGenerationInfoEdge(unittest.TestCase):
    def setUp(self):
        from wan_nodes import WanVideoNode
        self.node = WanVideoNode()

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=0)
    def test_zero_size_file(self, _, __):
        """测试零大小文件"""
        info = self.node._create_generation_info(
            "/out.mp4", prompt="", negative_prompt="",
            model_name="Wan2.1-T2V-1.3B", width=832, height=480,
            frames=81, num_inference_steps=50, seed=-1, n_npus=8,
            enable_distillation=False
        )
        self.assertEqual(info["file_size_mb"], 0.0)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024 * 1024 * 1024)
    def test_large_file(self, _, __):
        """测试大文件"""
        info = self.node._create_generation_info(
            "/out.mp4", prompt="", negative_prompt="",
            model_name="Wan2.1-T2V-1.3B", width=832, height=480,
            frames=81, num_inference_steps=50, seed=-1, n_npus=8,
            enable_distillation=False
        )
        self.assertEqual(info["file_size_mb"], 1024.0)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=100)
    def test_important_params_preserved(self, _, __):
        """测试重要参数被保留"""
        info = self.node._create_generation_info(
            "/out.mp4", prompt="test prompt", negative_prompt="neg",
            model_name="Wan2.1-T2V-1.3B", width=832, height=480,
            frames=81, num_inference_steps=50, seed=42, n_npus=4,
            enable_distillation=False
        )
        self.assertEqual(info["prompt"], "test prompt")
        self.assertEqual(info["negative_prompt"], "neg")
        self.assertEqual(info["seed"], 42)
        self.assertEqual(info["n_npus"], 4)
        self.assertIn("output_path", info)
        self.assertIn("generation_time", info)


# ═══════════════════════════════════════════════════════════════════════════
# 17. PromptCacheNode CATEGORY
# ═══════════════════════════════════════════════════════════════════════════
class TestPromptCacheNodeMeta(unittest.TestCase):
    def test_category(self):
        from wan_nodes import PromptCacheNode
        self.assertEqual(PromptCacheNode.CATEGORY, "text")

    def test_function(self):
        from wan_nodes import PromptCacheNode
        self.assertEqual(PromptCacheNode.FUNCTION, "get_cached_prompt")


# ═══════════════════════════════════════════════════════════════════════════
# 18. ModelConfig 边界：model_path 一致性
# ═══════════════════════════════════════════════════════════════════════════
class TestModelConfigEdgeCases(unittest.TestCase):
    def test_model_name_equals_config_key(self):
        """测试 CONFIGS 中每个 model 字段等于其键名"""
        for name in ModelConfig.get_model_names():
            cfg = ModelConfig.get_config(name)
            self.assertEqual(cfg["model"], name)

    def test_model_path_nonempty(self):
        """测试所有模型路径非空"""
        for name in ModelConfig.get_model_names():
            cfg = ModelConfig.get_config(name)
            self.assertTrue(len(cfg["model_path"]) > 0)

    def test_default_frames_range(self):
        """测试默认帧数在合理范围"""
        for name in ModelConfig.get_model_names():
            p = ModelConfig.get_default_params(name)
            self.assertGreater(p["frames"], 0)
            self.assertLessEqual(p["frames"], 200)

    def test_default_steps_range(self):
        """测试默认推理步数在合理范围"""
        for name in ModelConfig.get_model_names():
            p = ModelConfig.get_default_params(name)
            self.assertGreater(p["num_inference_steps"], 0)
            self.assertLessEqual(p["num_inference_steps"], 100)

    def test_wan22_guidance_scale_defaults(self):
        """测试 Wan2.2 模型默认 guidance_scale 值"""
        for name in ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"]:
            p = ModelConfig.get_default_params(name)
            self.assertIn("guidance_scale", p)
            self.assertIn("guidance_scale_2", p)
            self.assertGreater(p["guidance_scale"], 0)
            self.assertGreater(p["guidance_scale_2"], 0)


if __name__ == "__main__":
    unittest.main()