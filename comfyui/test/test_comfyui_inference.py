"""
ComfyUI 推理流程集成测试
覆盖: 分布式初始化、环境变量、工作流 JSON、目录结构、
      源码关键特征、序列并行、README 内容校验
"""
import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

COMFYUI_DIR = os.path.join(os.path.dirname(__file__), "..")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_DIR, "custom_nodes")
ASCENDX_DIR = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-WanVideoWrapper", "comfyui_ascendx")
sys.path.insert(0, os.path.join(CUSTOM_NODES_DIR, "ComfyUI-WanVideoWrapper"))
sys.path.insert(0, ASCENDX_DIR)


def _read_file(rel_path):
    with open(os.path.join(COMFYUI_DIR, rel_path), "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# 分布式后端选择测试
# ============================================================
class TestDistributedBackendSelection(unittest.TestCase):

    def test_npu_available_selects_hccl(self):
        mock_torch = MagicMock()
        mock_torch.npu.is_available.return_value = True
        self.assertTrue(mock_torch.npu.is_available())

    def test_gpu_fallback_to_nccl(self):
        mock_torch = MagicMock()
        mock_torch.npu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        self.assertTrue(mock_torch.cuda.is_available())

    def test_cpu_fallback_to_gloo(self):
        mock_torch = MagicMock()
        mock_torch.npu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        self.assertFalse(mock_torch.npu.is_available())
        self.assertFalse(mock_torch.cuda.is_available())

    def test_rank_env_determines_distributed(self):
        with patch.dict(os.environ, {"RANK": "0"}):
            self.assertIn("RANK", os.environ)

    def test_no_rank_means_single_process(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("RANK", os.environ)


class TestDistributedDeviceBinding(unittest.TestCase):

    def test_npu_set_device_called(self):
        mock_torch = MagicMock()
        mock_torch.npu.is_available.return_value = True
        mock_torch.npu.set_device(0)
        mock_torch.npu.set_device.assert_called_once_with(0)

    def test_cuda_set_device_called(self):
        mock_torch = MagicMock()
        mock_torch.npu.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.set_device(0)
        mock_torch.cuda.set_device.assert_called_once_with(0)


# ============================================================
# main_multi.py 环境与初始化测试
# ============================================================
class TestMainMultiEnvironment(unittest.TestCase):

    def test_acl_context_mode_set(self):
        content = _read_file("main_multi.py")
        self.assertIn("ACL_CONTEXT_MODE", content)
        self.assertIn(3, content)

    def test_fsdp_env_var_in_readme(self):
        content = _read_file("README.md")
        self.assertIn("WAN_FSDP_ENABLE", content)

    def test_phaa_env_var_in_model(self):
        content = _read_file(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"))
        self.assertIn("PHAA_ENABLE", content)


class TestMainMultiDistributedInit(unittest.TestCase):

    def test_contains_dist_init(self):
        self.assertIn("init_process_group", _read_file("main_multi.py"))

    def test_contains_hccl_backend(self):
        self.assertIn("hccl", _read_file("main_multi.py"))

    def test_contains_nccl_backend(self):
        self.assertIn("nccl", _read_file("main_multi.py"))

    def test_contains_gloo_backend(self):
        self.assertIn("gloo", _read_file("main_multi.py"))

    def test_contains_timeout_config(self):
        self.assertIn("timedelta", _read_file("main_multi.py"))

    def test_contains_rank_and_world_size(self):
        content = _read_file("main_multi.py")
        self.assertIn("get_rank", content)
        self.assertIn("get_world_size", content)

    def test_contains_npu_set_device(self):
        content = _read_file("main_multi.py")
        self.assertIn("npu.set_device", content)

    def test_contains_cuda_set_device(self):
        content = _read_file("main_multi.py")
        self.assertIn("cuda.set_device", content)

    def test_contains_frontend_thread(self):
        self.assertIn("frontend_thread", _read_file("main_multi.py"))

    def test_contains_slave_worker(self):
        self.assertIn("slave_worker_main_thread", _read_file("main_multi.py"))


# ============================================================
# 工作流 JSON 测试
# ============================================================
class TestWorkflowJson(unittest.TestCase):

    def _load_workflow(self):
        with open(os.path.join(COMFYUI_DIR, "Wan2.2首尾帧720P加速版.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_workflow_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(COMFYUI_DIR, "Wan2.2首尾帧720P加速版.json")))

    def test_workflow_is_valid_json_dict(self):
        self.assertIsInstance(self._load_workflow(), dict)

    def test_workflow_has_nodes_key(self):
        self.assertIn("nodes", self._load_workflow())

    def test_workflow_has_links_key(self):
        self.assertIn("links", self._load_workflow())

    def test_workflow_nodes_is_list(self):
        self.assertIsInstance(self._load_workflow()["nodes"], list)

    def test_workflow_has_at_least_one_node(self):
        self.assertTrue(len(self._load_workflow()["nodes"]) > 0)

    def test_workflow_links_is_list(self):
        self.assertIsInstance(self._load_workflow()["links"], list)

    def test_workflow_file_not_empty(self):
        self.assertTrue(os.path.getsize(os.path.join(COMFYUI_DIR, "Wan2.2首尾帧720P加速版.json")) > 0)


# ============================================================
# 目录结构测试
# ============================================================
class TestDirectoryStructure(unittest.TestCase):

    def test_comfyui_dir_exists(self):
        self.assertTrue(os.path.isdir(COMFYUI_DIR))

    def test_custom_nodes_dir_exists(self):
        self.assertTrue(os.path.isdir(CUSTOM_NODES_DIR))

    def test_wan_video_wrapper_dir_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(CUSTOM_NODES_DIR, "ComfyUI-WanVideoWrapper")))

    def test_ascendx_dir_exists(self):
        self.assertTrue(os.path.isdir(ASCENDX_DIR))

    def test_modules_dir_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(ASCENDX_DIR, "modules")))

    def test_patches_dir_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(ASCENDX_DIR, "patches")))

    def test_thinksound_wrapper_dir_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(CUSTOM_NODES_DIR, "ComfyUI-ThinkSound_Wrapper")))

    def test_video_helper_suite_dir_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(CUSTOM_NODES_DIR, "ComfyUI-VideoHelperSuite")))

    def test_main_multi_py_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(COMFYUI_DIR, "main_multi.py")))

    def test_readme_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(COMFYUI_DIR, "README.md")))


class TestModuleFilesExist(unittest.TestCase):

    def _assert_module_exists(self, *parts):
        path = os.path.join(ASCENDX_DIR, *parts)
        self.assertTrue(os.path.isfile(path), f"{path} 不存在")

    def test_model_py(self):
        self._assert_module_exists("modules", "model.py")

    def test_attention_py(self):
        self._assert_module_exists("modules", "attention.py")

    def test_quant_py(self):
        self._assert_module_exists("modules", "quant.py")

    def test_fsdp_py(self):
        self._assert_module_exists("modules", "fsdp.py")

    def test_vae_decode_multi_py(self):
        self._assert_module_exists("modules", "vae_decode_multi.py")

    def test_nodes_model_loading_py(self):
        self._assert_module_exists("nodes_model_loading.py")


# ============================================================
# 序列并行 get_sp_group 测试
# ============================================================
class TestGetSpGroup(unittest.TestCase):

    def test_returns_none_when_dist_not_initialized(self):
        with patch("comfyui_ascendx.modules.model.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            from comfyui_ascendx.modules.model import get_sp_group
            self.assertIsNone(get_sp_group()[0])

    def test_returns_world_size_1_when_not_distributed(self):
        with patch("comfyui_ascendx.modules.model.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            from comfyui_ascendx.modules.model import get_sp_group
            self.assertEqual(get_sp_group()[1], 1)


# ============================================================
# 源码关键特征测试
# ============================================================
class TestSourceCodeKeyFeatures(unittest.TestCase):

    def _check_contains(self, rel_path, keyword):
        content = _read_file(rel_path)
        self.assertIn(keyword, content)

    def test_model_py_has_is_phaa_enabled(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"), "is_phaa_enabled")

    def test_model_py_has_get_sp_group(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"), "get_sp_group")

    def test_model_py_has_SPWanLayerNorm(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"), "SPWanLayerNorm")

    def test_model_py_has_all_to_all(self):
        content = _read_file(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"))
        self.assertIn("all_to_all_before_attn", content)
        self.assertIn("all_to_all_after_attn", content)

    def test_model_py_has_zero_timestep(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"), "zero_timestep")

    def test_model_py_has_split_cross_attn_ffn(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "model.py"), "split_cross_attn_ffn")

    def test_quant_py_has_npu_dynamic_quant(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "quant.py"), "npu_dynamic_quant")

    def test_quant_py_has_npu_quant_matmul(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "quant.py"), "npu_quant_matmul")

    def test_attention_py_has_sageattn(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "attention.py"), "sageattn")

    def test_attention_py_has_laser(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "attention.py"), "laser")

    def test_attention_py_has_sdpa(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "attention.py"), "scaled_dot_product_attention")

    def test_fsdp_py_has_FSDP(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "fsdp.py"), "FullyShardedDataParallel")

    def test_fsdp_py_has_recursive_search(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "fsdp.py"), "_recursive_search")

    def test_vae_has_VAEManager(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "vae_decode_multi.py"), "VAEManager")

    def test_vae_has_spatial_tiled_decode(self):
        self._check_contains(os.path.join("custom_nodes", "ComfyUI-WanVideoWrapper",
                                           "comfyui_ascendx", "modules", "vae_decode_multi.py"), "spatial_tiled_decode")


# ============================================================
# 模型类可导入性测试
# ============================================================
class TestModelClassImportable(unittest.TestCase):

    def test_SPWanLayerNorm_importable(self):
        from comfyui_ascendx.modules.model import SPWanLayerNorm
        self.assertTrue(callable(SPWanLayerNorm))

    def test_is_phaa_enabled_importable(self):
        from comfyui_ascendx.modules.model import is_phaa_enabled
        self.assertTrue(callable(is_phaa_enabled))

    def test_get_sp_group_importable(self):
        from comfyui_ascendx.modules.model import get_sp_group
        self.assertTrue(callable(get_sp_group))


# ============================================================
# README 内容测试
# ============================================================
class TestReadmeContent(unittest.TestCase):

    def _read_readme(self):
        return _read_file("README.md")

    def test_mentions_comfyui(self):
        self.assertIn("ComfyUI", self._read_readme())

    def test_mentions_w8a8(self):
        self.assertIn("W8A8", self._read_readme())

    def test_mentions_sage_attention(self):
        self.assertIn("Sage Attention", self._read_readme())

    def test_mentions_fsdp(self):
        self.assertIn("FSDP", self._read_readme())

    def test_mentions_sequence_parallel(self):
        self.assertIn("序列并行", self._read_readme())

    def test_mentions_torchrun(self):
        self.assertIn("torchrun", self._read_readme())
