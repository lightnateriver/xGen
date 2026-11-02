"""
ComfyUI 自定义节点与模块单元测试
覆盖: NPUWanVideoModelLoader 配置、WeightQuantLinearModule 量化、
      attention 模式分发、FSDP 配置、SPWanLayerNorm、PHAA 开关、
      VAE 解码签名、Patch 文件完整性
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CUSTOM_NODES_DIR = os.path.join(os.path.dirname(__file__), "..", "custom_nodes")
ASCENDX_DIR = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-WanVideoWrapper", "comfyui_ascendx")
sys.path.insert(0, os.path.join(CUSTOM_NODES_DIR, "ComfyUI-WanVideoWrapper"))
sys.path.insert(0, ASCENDX_DIR)


def _get_loader():
    from comfyui_ascendx.nodes_model_loading import NPUWanVideoModelLoader
    return NPUWanVideoModelLoader


# ============================================================
# NPUWanVideoModelLoader 节点配置测试
# ============================================================
class TestNPUWanVideoModelLoaderRequired(unittest.TestCase):

    def test_input_types_has_required_and_optional(self):
        Loader = _get_loader()
        result = Loader.INPUT_TYPES()
        self.assertIn("required", result)
        self.assertIn("optional", result)

    def test_required_contains_all_fields(self):
        Loader = _get_loader()
        required = Loader.INPUT_TYPES()["required"]
        for field in ("model", "base_precision", "quantization", "load_device"):
            self.assertIn(field, required)

    def test_required_field_count_is_four(self):
        Loader = _get_loader()
        self.assertEqual(len(Loader.INPUT_TYPES()["required"]), 4)

    def test_model_is_tuple_with_tooltip(self):
        Loader = _get_loader()
        model_cfg = Loader.INPUT_TYPES()["required"]["model"]
        self.assertIsInstance(model_cfg, tuple)
        self.assertEqual(len(model_cfg), 2)


class TestNPUWanVideoModelLoaderPrecision(unittest.TestCase):

    def test_precision_options(self):
        Loader = _get_loader()
        options = Loader.INPUT_TYPES()["required"]["base_precision"][0]
        for p in ("fp32", "bf16", "fp16"):
            self.assertIn(p, options)

    def test_precision_default_is_bf16(self):
        Loader = _get_loader()
        default = Loader.INPUT_TYPES()["required"]["base_precision"][1]["default"]
        self.assertEqual(default, "bf16")

    def test_precision_has_three_options(self):
        Loader = _get_loader()
        self.assertEqual(len(Loader.INPUT_TYPES()["required"]["base_precision"][0]), 3)


class TestNPUWanVideoModelLoaderQuantization(unittest.TestCase):

    def test_quantization_options(self):
        Loader = _get_loader()
        options = Loader.INPUT_TYPES()["required"]["quantization"][0]
        self.assertIn("disabled", options)
        self.assertIn("w8a8", options)

    def test_quantization_default_is_disabled(self):
        Loader = _get_loader()
        self.assertEqual(Loader.INPUT_TYPES()["required"]["quantization"][1]["default"], "disabled")

    def test_quantization_tooltip_mentions_w8a8(self):
        Loader = _get_loader()
        self.assertIn("w8a8", Loader.INPUT_TYPES()["required"]["quantization"][1]["tooltip"])


class TestNPUWanVideoModelLoaderLoadDevice(unittest.TestCase):

    def test_load_device_options(self):
        Loader = _get_loader()
        options = Loader.INPUT_TYPES()["required"]["load_device"][0]
        self.assertIn("main_device", options)
        self.assertIn("offload_device", options)

    def test_load_device_default_is_offload_device(self):
        Loader = _get_loader()
        self.assertEqual(Loader.INPUT_TYPES()["required"]["load_device"][1]["default"], "offload_device")


class TestNPUWanVideoModelLoaderOptional(unittest.TestCase):

    def test_optional_fields_exist(self):
        Loader = _get_loader()
        optional = Loader.INPUT_TYPES()["optional"]
        for field in ("attention_mode", "compile_args", "block_swap_args",
                      "lora", "vram_management_args", "extra_model",
                      "rms_norm_function"):
            self.assertIn(field, optional)

    def test_attention_mode_options(self):
        Loader = _get_loader()
        options = Loader.INPUT_TYPES()["optional"]["attention_mode"][0]
        for mode in ("sdpa", "sageattn", "laserattn"):
            self.assertIn(mode, options)

    def test_attention_mode_default_is_sdpa(self):
        Loader = _get_loader()
        self.assertEqual(Loader.INPUT_TYPES()["optional"]["attention_mode"][1]["default"], "sdpa")

    def test_rms_norm_function_options_and_default(self):
        Loader = _get_loader()
        options = Loader.INPUT_TYPES()["optional"]["rms_norm_function"][0]
        self.assertIn("default", options)
        self.assertIn("pytorch", options)
        self.assertEqual(Loader.INPUT_TYPES()["optional"]["rms_norm_function"][1]["default"], "default")


# ============================================================
# WeightQuantLinearModule 量化测试
# ============================================================
class TestDynamicQuantSkippedCases(unittest.TestCase):

    def test_skip_cases_not_empty(self):
        from comfyui_ascendx.modules.quant import DYNAMIC_QUANT_SKIPPED_CASE
        self.assertTrue(len(DYNAMIC_QUANT_SKIPPED_CASE) > 0)

    def test_skip_cases_contains_all_keywords(self):
        from comfyui_ascendx.modules.quant import DYNAMIC_QUANT_SKIPPED_CASE
        for kw in ("time_embedding", "time_projection", "text_embedding", "head"):
            self.assertIn(kw, DYNAMIC_QUANT_SKIPPED_CASE)

    def test_skip_cases_has_four_entries(self):
        from comfyui_ascendx.modules.quant import DYNAMIC_QUANT_SKIPPED_CASE
        self.assertEqual(len(DYNAMIC_QUANT_SKIPPED_CASE), 4)

    def test_skip_layer_name_matches(self):
        from comfyui_ascendx.modules.quant import DYNAMIC_QUANT_SKIPPED_CASE
        self.assertTrue(any(s in "blocks.0.time_embedding.linear" for s in DYNAMIC_QUANT_SKIPPED_CASE))

    def test_normal_layer_name_not_matches(self):
        from comfyui_ascendx.modules.quant import DYNAMIC_QUANT_SKIPPED_CASE
        self.assertFalse(any(s in "blocks.3.ffn.linear" for s in DYNAMIC_QUANT_SKIPPED_CASE))


class TestWeightQuantLinearModuleMocked(unittest.TestCase):

    @patch("comfyui_ascendx.modules.quant.torch_npu", create=True)
    def test_forward_output_shape_2d(self, mock_npu):
        import torch
        import torch.nn as nn
        linear = nn.Linear(64, 32, bias=True)
        mock_npu.npu_dynamic_quant.return_value = (MagicMock(), MagicMock())
        mock_npu.npu_quant_matmul.return_value = torch.randn(4, 32)
        from comfyui_ascendx.modules.quant import WeightQuantLinearModule
        qmod = WeightQuantLinearModule(linear.weight, linear.bias)
        out = qmod(torch.randn(4, 64))
        self.assertEqual(out.shape, (4, 32))

    @patch("comfyui_ascendx.modules.quant.torch_npu", create=True)
    def test_forward_output_shape_3d(self, mock_npu):
        import torch
        import torch.nn as nn
        linear = nn.Linear(64, 32, bias=True)
        mock_npu.npu_dynamic_quant.return_value = (MagicMock(), MagicMock())
        mock_npu.npu_quant_matmul.return_value = torch.randn(16, 32)
        from comfyui_ascendx.modules.quant import WeightQuantLinearModule
        qmod = WeightQuantLinearModule(linear.weight, linear.bias)
        out = qmod(torch.randn(2, 8, 64))
        self.assertEqual(out.shape, (2, 8, 32))

    @patch("comfyui_ascendx.modules.quant.torch_npu", create=True)
    def test_forward_calls_npu_dynamic_quant(self, mock_npu):
        import torch
        import torch.nn as nn
        linear = nn.Linear(32, 16, bias=True)
        mock_npu.npu_dynamic_quant.return_value = (MagicMock(), MagicMock())
        mock_npu.npu_quant_matmul.return_value = torch.randn(4, 16)
        from comfyui_ascendx.modules.quant import WeightQuantLinearModule
        qmod = WeightQuantLinearModule(linear.weight, linear.bias)
        qmod(torch.randn(4, 32))
        self.assertTrue(mock_npu.npu_dynamic_quant.called)

    @patch("comfyui_ascendx.modules.quant.torch_npu", create=True)
    def test_forward_calls_npu_quant_matmul(self, mock_npu):
        import torch
        import torch.nn as nn
        linear = nn.Linear(32, 16, bias=True)
        mock_npu.npu_dynamic_quant.return_value = (MagicMock(), MagicMock())
        mock_npu.npu_quant_matmul.return_value = torch.randn(4, 16)
        from comfyui_ascendx.modules.quant import WeightQuantLinearModule
        qmod = WeightQuantLinearModule(linear.weight, linear.bias)
        qmod(torch.randn(4, 32))
        self.assertTrue(mock_npu.npu_quant_matmul.called)

    @patch("comfyui_ascendx.modules.quant.torch_npu", create=True)
    def test_forward_preserves_dtype(self, mock_npu):
        import torch
        import torch.nn as nn
        linear = nn.Linear(32, 16, bias=True)
        mock_npu.npu_dynamic_quant.return_value = (MagicMock(), MagicMock())
        mock_npu.npu_quant_matmul.return_value = torch.randn(4, 16, dtype=torch.bfloat16)
        from comfyui_ascendx.modules.quant import WeightQuantLinearModule
        qmod = WeightQuantLinearModule(linear.weight, linear.bias)
        out = qmod(torch.randn(4, 32, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)

    @patch("comfyui_ascendx.modules.quant.torch_npu", create=True)
    def test_init_registers_three_parameters(self, mock_npu):
        import torch.nn as nn
        linear = nn.Linear(32, 16, bias=True)
        mock_npu.npu_dynamic_quant.return_value = (MagicMock(), MagicMock())
        from comfyui_ascendx.modules.quant import WeightQuantLinearModule
        qmod = WeightQuantLinearModule(linear.weight, linear.bias)
        for name in ("weight", "antiquant_scale", "bias"):
            self.assertIn(name, qmod._parameters)


# ============================================================
# Attention 模式测试
# ============================================================
class TestAttentionFunction(unittest.TestCase):

    def test_attention_function_exists(self):
        from comfyui_ascendx.modules.attention import attention
        self.assertTrue(callable(attention))

    def test_attention_accepts_mode_param(self):
        import inspect
        from comfyui_ascendx.modules.attention import attention
        self.assertIn("attention_mode", inspect.signature(attention).parameters)

    def test_attention_default_mode_is_sdpa(self):
        import inspect
        from comfyui_ascendx.modules.attention import attention
        self.assertEqual(inspect.signature(attention).parameters["attention_mode"].default, "sdpa")

    def test_attention_signature_params(self):
        import inspect
        from comfyui_ascendx.modules.attention import attention
        params = inspect.signature(attention).parameters
        for p in ("q", "k", "v", "causal", "window_size", "dtype",
                  "q_lens", "k_lens", "softmax_scale", "deterministic", "q_scale"):
            self.assertIn(p, params)

    def test_mindie_available_is_bool(self):
        from comfyui_ascendx.modules.attention import MINDIE_AVAILABLE
        self.assertIsInstance(MINDIE_AVAILABLE, bool)


# ============================================================
# SPWanLayerNorm 测试
# ============================================================
class TestSPWanLayerNorm(unittest.TestCase):

    def test_forward_output_shape(self):
        import torch
        from comfyui_ascendx.modules.model import SPWanLayerNorm
        norm = SPWanLayerNorm(128, elementwise_affine=False)
        x = torch.randn(2, 16, 128)
        self.assertEqual(norm(x).shape, x.shape)

    def test_forward_long_seq(self):
        import torch
        from comfyui_ascendx.modules.model import SPWanLayerNorm
        norm = SPWanLayerNorm(256, elementwise_affine=False)
        x = torch.randn(4, 1024, 256)
        self.assertEqual(norm(x).shape, (4, 1024, 256))

    def test_inherits_from_layernorm(self):
        import torch.nn as nn
        from comfyui_ascendx.modules.model import SPWanLayerNorm
        self.assertTrue(issubclass(SPWanLayerNorm, nn.LayerNorm))

    def test_default_eps(self):
        from comfyui_ascendx.modules.model import SPWanLayerNorm
        self.assertAlmostEqual(SPWanLayerNorm(64, elementwise_affine=False).eps, 1e-6)

    def test_elementwise_affine_false_no_weight_bias(self):
        from comfyui_ascendx.modules.model import SPWanLayerNorm
        norm = SPWanLayerNorm(64, elementwise_affine=False)
        self.assertIsNone(norm.weight)
        self.assertIsNone(norm.bias)


# ============================================================
# PHAA 开关测试
# ============================================================
class TestIsPhaaEnabled(unittest.TestCase):

    def test_disabled_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertFalse(is_phaa_enabled())

    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "true"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertTrue(is_phaa_enabled())

    def test_enabled_with_1(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "1"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertTrue(is_phaa_enabled())

    def test_enabled_with_on(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "on"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertTrue(is_phaa_enabled())

    def test_disabled_with_false(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "false"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertFalse(is_phaa_enabled())

    def test_disabled_with_0(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "0"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertFalse(is_phaa_enabled())

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "TRUE"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertTrue(is_phaa_enabled())

    def test_whitespace_trimmed(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": " true "}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertTrue(is_phaa_enabled())

    def test_empty_string_disabled(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": ""}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertFalse(is_phaa_enabled())

    def test_random_string_disabled(self):
        with patch.dict(os.environ, {"PHAA_ENABLE": "yes"}):
            from comfyui_ascendx.modules.model import is_phaa_enabled
            self.assertFalse(is_phaa_enabled())


# ============================================================
# FSDP 函数签名测试
# ============================================================
class TestFsdpFunction(unittest.TestCase):

    def test_fsdp_function_exists(self):
        from comfyui_ascendx.modules.fsdp import fsdp
        self.assertTrue(callable(fsdp))

    def test_fsdp_signature_params(self):
        import inspect
        from comfyui_ascendx.modules.fsdp import fsdp
        params = inspect.signature(fsdp).parameters
        for p in ("model", "device_id", "param_dtype", "sharding_strategy",
                  "process_group", "sync_module_states"):
            self.assertIn(p, params)

    def test_fsdp_default_dtype_is_bfloat16(self):
        import inspect
        import torch
        from comfyui_ascendx.modules.fsdp import fsdp
        self.assertEqual(inspect.signature(fsdp).parameters["param_dtype"].default, torch.bfloat16)

    def test_fsdp_default_sync_module_states_is_true(self):
        import inspect
        from comfyui_ascendx.modules.fsdp import fsdp
        self.assertTrue(inspect.signature(fsdp).parameters["sync_module_states"].default)


# ============================================================
# VAE 解码签名测试
# ============================================================
class TestVaeDecodeSignature(unittest.TestCase):

    def test_spatial_tiled_decode_exists(self):
        from comfyui_ascendx.modules.vae_decode_multi import spatial_tiled_decode
        self.assertTrue(callable(spatial_tiled_decode))

    def test_signature_params(self):
        import inspect
        from comfyui_ascendx.modules.vae_decode_multi import spatial_tiled_decode
        params = inspect.signature(spatial_tiled_decode).parameters
        for p in ("hidden_state", "device", "decoder_wrapper", "pbar"):
            self.assertIn(p, params)

    def test_pbar_default_is_true(self):
        import inspect
        from comfyui_ascendx.modules.vae_decode_multi import spatial_tiled_decode
        self.assertTrue(inspect.signature(spatial_tiled_decode).parameters["pbar"].default)


# ============================================================
# Patch 文件完整性测试
# ============================================================
class TestPatchFiles(unittest.TestCase):

    def test_model_patch_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(ASCENDX_DIR, "patches", "model.patch")))

    def test_nodes_model_loading_patch_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(ASCENDX_DIR, "patches", "nodes_model_loading.patch")))

    def test_vae_decode_multi_patch_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(ASCENDX_DIR, "patches", "vae_decode_multi.patch")))

    def test_patch_apply_script_exists_and_not_empty(self):
        path = os.path.join(ASCENDX_DIR, "patch_apply.sh")
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(os.path.getsize(path) > 0)

    def test_thinksound_patch_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(CUSTOM_NODES_DIR, "ComfyUI-ThinkSound_Wrapper", "thinksound.patch")))

    def test_video_save_adaptor_patch_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(CUSTOM_NODES_DIR, "ComfyUI-VideoHelperSuite", "video_save_adaptor.patch")))

