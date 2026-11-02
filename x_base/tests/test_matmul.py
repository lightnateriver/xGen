"""
测试用例: x_base.operator.matmul 模块
覆盖:
  - WeightQuantLinearModule: 初始化、forward 形状保持、dtype 转换
  - WeightQuantLinearModule_w4a4: 初始化、next_power 计算、pad 逻辑、
    smoothquant 缩放、forward 首次/非首次调用
  - enable_weight_quant_linear: w8a8/w4a4 模式替换、白名单/黑名单过滤、
    嵌套模块替换、根层级替换、lora 重命名处理
  - QuantWanCausalConv3d: 初始化、transform_5d_to_7d、padding 计算
  - enable_conv3d_quant: 替换逻辑、形状条件过滤
  - MatmulManager: enable_online_dynamic_quant / enable_vae_conv3d_quant、
    自动推断配置、异常处理
  - QuantLayerConfig: should_quantize / should_use_w4a4 / _matches_patterns /
    from_dict / __post_init__
  - QuantConfigManager: 单例模式、get_config / detect_config_name / reload

运行: python -m pytest test_matmul.py -v
"""

import gc
import logging
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch
from torch import nn

# ── 路径设置 ──
SOURCE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "x_gen", "x_base"))
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

# ── Mock 外部依赖 ──
# Mock torch_npu
torch_npu_mock = MagicMock()

def _fake_npu_dynamic_quant(tensor, dst_type=None):
    """模拟 npu_dynamic_quant: 返回 int8 张量和 scale"""
    shape = tensor.shape if hasattr(tensor, 'shape') else (1, 1)
    quant = torch.randint(-127, 127, shape, dtype=torch.int8)
    scale = torch.ones(shape[0], dtype=torch.float32)
    return quant, scale

torch_npu_mock.npu_dynamic_quant = _fake_npu_dynamic_quant

def _fake_npu_quant_matmul(x, w, scale, bias=None, pertoken_scale=None, output_dtype=torch.float16):
    """模拟 npu_quant_matmul: 返回形状匹配的浮点张量"""
    out_features = w.shape[1] if len(w.shape) > 1 else w.shape[0]
    batch = x.shape[0]
    return torch.randn(batch, out_features, dtype=output_dtype)

torch_npu_mock.npu_quant_matmul = _fake_npu_quant_matmul

def _fake_npu_quantize(tensor, scales=None, zero_points=None, dtype=torch.qint8, axis=0, div_mode=True):
    """模拟 npu_quantize"""
    return tensor.to(torch.int8)

torch_npu_mock.npu_quantize = _fake_npu_quantize
sys.modules["torch_npu"] = torch_npu_mock

# Mock hadamard_transform
hadamard_mock = MagicMock(side_effect=lambda x: x)
sys.modules["hadamard_transform"] = MagicMock(hadamard_transform=hadamard_mock)

# Mock fast_hadamard_npu
sys.modules["fast_hadamard_npu"] = MagicMock()

# Mock turing_cloud_ops
turing_cloud_ops_mock = MagicMock()
def _fake_cloud_dynamic_quant(x, smooth_scales=None, group_index=None, dst_type=2,
                               is_symmetrical=True, quant_mode=1):
    shape = x.shape
    return x.to(torch.int8), torch.ones(shape[0], dtype=torch.float32)

turing_cloud_ops_mock.cloud_dynamic_quant = _fake_cloud_dynamic_quant

def _fake_cloud_conv3d_quant(x_int8, weight, bias, scale, stride, padding, dilations):
    # 返回与输入 batch 维度匹配的输出
    return torch.randn(x_int8.shape[0], weight.shape[0], x_int8.shape[2],
                       x_int8.shape[3], x_int8.shape[4], dtype=torch.float16)

turing_cloud_ops_mock.cloud_conv3d_quant = _fake_cloud_conv3d_quant
sys.modules["turing_cloud_ops"] = turing_cloud_ops_mock

# Mock x_base.config — 在 import matmul 之前
# 先确保 x_base 包可导入
x_base_init = os.path.join(SOURCE_ROOT, "x_base", "__init__.py")
if not os.path.exists(x_base_init):
    os.makedirs(os.path.dirname(x_base_init), exist_ok=True)
    with open(x_base_init, "w") as f:
        f.write("")

# Mock yaml (config_loader 依赖)
sys.modules["yaml"] = MagicMock()

# Mock torch.distributed.fsdp
fsdp_mock = MagicMock()
sys.modules["torch.distributed.fsdp"] = MagicMock()
sys.modules["torch.distributed.fsdp.fully_sharded_data_parallel"] = MagicMock(
    FullyShardedDataParallel=fsdp_mock)

# 现在导入
from x_base.config.config_loader import QuantLayerConfig, QuantConfigManager
from x_base.operator.matmul import (
    WeightQuantLinearModule,
    WeightQuantLinearModule_w4a4,
    QuantWanCausalConv3d,
    MatmulManager,
    matmul_manager,
    enable_weight_quant_linear,
    enable_conv3d_quant,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. WeightQuantLinearModule 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestWeightQuantLinearModule(unittest.TestCase):
    def _make_linear(self, in_f=64, out_f=32):
        return nn.Linear(in_f, out_f)

    def test_init_shapes(self):
        """测试初始化后形状正确"""
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        self.assertEqual(mod.out_features, 32)
        self.assertEqual(mod.in_features, 64)

    def test_init_weight_dtype(self):
        """测试权重参数为 int8"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        self.assertEqual(mod.weight.dtype, torch.int8)

    def test_init_has_antiquant_scale(self):
        """测试存在 antiquant_scale 参数"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        self.assertTrue(hasattr(mod, 'antiquant_scale'))

    def test_init_has_bias(self):
        """测试存在 bias 参数"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        self.assertTrue(hasattr(mod, 'bias'))

    def test_forward_2d_input(self):
        """测试 2D 输入 forward"""
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        x = torch.randn(4, 64, dtype=torch.float16)
        out = mod(x)
        self.assertEqual(out.shape, (4, 32))

    def test_forward_3d_input(self):
        """测试 3D 输入 forward 保持前导维度"""
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        x = torch.randn(2, 3, 64, dtype=torch.float16)
        out = mod(x)
        self.assertEqual(out.shape, (2, 3, 32))

    def test_forward_preserves_batch_dims(self):
        """测试 forward 保持除最后一维外的所有维度"""
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        x = torch.randn(2, 3, 5, 64, dtype=torch.float16)
        out = mod(x)
        self.assertEqual(out.shape[:-1], x.shape[:-1])
        self.assertEqual(out.shape[-1], 32)

    def test_forward_weight_dtype_conversion(self):
        """测试 forward 时权重 dtype 自动转 int8"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule(lin.weight, lin.bias)
        # 强制改为非 int8
        mod.weight.data = mod.weight.data.to(torch.float16)
        x = torch.randn(2, 64, dtype=torch.float16)
        out = mod(x)
        self.assertEqual(mod.weight.dtype, torch.int8)


# ═══════════════════════════════════════════════════════════════════════════
# 2. WeightQuantLinearModule_w4a4 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestWeightQuantLinearModuleW4A4(unittest.TestCase):
    def _make_linear(self, in_f=64, out_f=32):
        return nn.Linear(in_f, out_f)

    def test_init_next_power_of_2(self):
        """测试 next_power 计算正确（向上取 2 的幂）"""
        # in_features=64 → next_power=64 (已是2的幂)
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        self.assertEqual(mod.next_power, 64)
        self.assertEqual(mod.d1, 64)

    def test_init_next_power_non_power_of_2(self):
        """测试非2幂次 in_features 的 next_power 计算"""
        # in_features=48 → next_power=64
        lin = nn.Linear(48, 16)
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        self.assertEqual(mod.next_power, 64)
        self.assertEqual(mod.d1, 48)

    def test_init_pad_size(self):
        """测试 pad_size 计算正确"""
        lin = nn.Linear(48, 16)
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        self.assertEqual(mod.pad_size, 64 - 48)

    def test_init_is_first_init_true(self):
        """测试初始化时 is_first_init 为 True"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        self.assertTrue(mod.is_first_init)

    def test_init_dtype_bfloat16(self):
        """测试默认 dtype 为 bfloat16"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        self.assertEqual(mod.dtype, torch.bfloat16)

    def test_init_name_stored(self):
        """测试 name 被保存"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias, name="test_layer")
        self.assertEqual(mod.name, "test_layer")

    def test_smoothquant_sets_is_first_init_false(self):
        """测试 smoothquant 后 is_first_init 变为 False"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        x = torch.randn(2, 64, dtype=torch.float16)
        mod.smoothquant(x, mod.weight, torch.float16)
        self.assertFalse(mod.is_first_init)

    def test_smoothquant_creates_smooth_scale(self):
        """测试 smoothquant 创建 smooth_scale"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        x = torch.randn(2, 64, dtype=torch.float16)
        mod.smoothquant(x, mod.weight, torch.float16)
        self.assertTrue(hasattr(mod, 'smooth_scale'))

    def test_smoothquant_creates_weight_scale(self):
        """测试 smoothquant 创建 weight_scale"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        x = torch.randn(2, 64, dtype=torch.float16)
        mod.smoothquant(x, mod.weight, torch.float16)
        self.assertTrue(hasattr(mod, 'weight_scale'))

    def test_forward_triggers_smoothquant_on_first(self):
        """测试首次 forward 触发 smoothquant"""
        lin = self._make_linear()
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        self.assertTrue(mod.is_first_init)
        x = torch.randn(2, 64, dtype=torch.float16)
        with patch.object(mod, 'smoothquant', wraps=mod.smoothquant) as spy:
            mod(x)
            spy.assert_called_once()

    def test_forward_output_shape(self):
        """测试 forward 输出形状正确"""
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        x = torch.randn(2, 64, dtype=torch.float16)
        out = mod(x)
        self.assertEqual(out.shape, (2, 32))

    def test_forward_3d_output_shape(self):
        """测试 3D 输入 forward 输出形状"""
        lin = self._make_linear(64, 32)
        mod = WeightQuantLinearModule_w4a4(lin.weight, lin.bias)
        x = torch.randn(2, 3, 64, dtype=torch.float16)
        out = mod(x)
        self.assertEqual(out.shape, (2, 3, 32))


# ═══════════════════════════════════════════════════════════════════════════
# 3. enable_weight_quant_linear 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestEnableWeightQuantLinear(unittest.TestCase):
    def _make_model(self):
        """创建简单的测试模型"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 16)
                self.linear2 = nn.Linear(16, 8)
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.linear2(self.relu(self.linear1(x)))
        return TestModel()

    def test_replaces_linear_layers_w8a8(self):
        """测试 w8a8 模式替换 Linear 层"""
        model = self._make_model()
        config = QuantLayerConfig()  # 默认全量化
        result = enable_weight_quant_linear(model, config, dtype='w8a8')
        self.assertIsInstance(result.linear1, WeightQuantLinearModule)
        self.assertIsInstance(result.linear2, WeightQuantLinearModule)

    def test_replaces_linear_layers_w4a4(self):
        """测试 w4a4 模式替换匹配的层"""
        model = self._make_model()
        config = QuantLayerConfig(w4a4_patterns=["linear"], w4a4_block_patterns=["linear"])
        result = enable_weight_quant_linear(model, config, dtype='w4a4')
        # linear1 和 linear2 都匹配 w4a4 模式
        self.assertIsInstance(result.linear1, WeightQuantLinearModule_w4a4)
        self.assertIsInstance(result.linear2, WeightQuantLinearModule_w4a4)

    def test_skips_non_linear_layers(self):
        """测试非 Linear 层不被替换"""
        model = self._make_model()
        config = QuantLayerConfig()
        result = enable_weight_quant_linear(model, config)
        self.assertIsInstance(result.relu, nn.ReLU)

    def test_exclude_patterns(self):
        """测试黑名单排除层"""
        model = self._make_model()
        config = QuantLayerConfig(exclude_patterns=["linear2"])
        result = enable_weight_quant_linear(model, config)
        self.assertIsInstance(result.linear1, WeightQuantLinearModule)
        self.assertIsInstance(result.linear2, nn.Linear)  # 被排除

    def test_include_patterns(self):
        """测试白名单只量化匹配层"""
        model = self._make_model()
        config = QuantLayerConfig(include_patterns=["linear1"])
        result = enable_weight_quant_linear(model, config)
        self.assertIsInstance(result.linear1, WeightQuantLinearModule)
        self.assertIsInstance(result.linear2, nn.Linear)  # 不在白名单

    def test_nested_module_replacement(self):
        """测试嵌套模块替换"""
        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(32, 16)
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.outer = Outer()

        model = Root()
        config = QuantLayerConfig()
        result = enable_weight_quant_linear(model, config)
        self.assertIsInstance(result.outer.inner, WeightQuantLinearModule)

    def test_lora_base_layer_rename(self):
        """测试 LoRA base_layer 重命名处理"""
        class LoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)
                # 模拟 LoRA 重命名
                self.linear.base_layer = nn.Linear(32, 16)
        model = LoRAModel()
        config = QuantLayerConfig()
        result = enable_weight_quant_linear(model, config)
        # base_layer 应被替换
        self.assertIsInstance(result.linear.base_layer, WeightQuantLinearModule)

    def test_mixed_w4a4_and_w8a8(self):
        """测试混合 w4a4 和 w8a8 替换"""
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks_to_k = nn.Linear(32, 16)
                self.other_proj = nn.Linear(16, 8)
        model = MixedModel()
        config = QuantLayerConfig(
            w4a4_patterns=["to_k"],
            w4a4_block_patterns=["blocks"]
        )
        result = enable_weight_quant_linear(model, config, dtype='w4a4')
        # blocks_to_k 同时匹配 to_k 和 blocks → w4a4
        self.assertIsInstance(result.blocks_to_k, WeightQuantLinearModule_w4a4)
        # other_proj 不匹配 w4a4 模式 → w8a8
        self.assertIsInstance(result.other_proj, WeightQuantLinearModule)


# ═══════════════════════════════════════════════════════════════════════════
# 4. QuantWanCausalConv3d 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestQuantWanCausalConv3d(unittest.TestCase):
    def test_init_padding_override(self):
        """测试初始化时 padding 被设为 (0,0,0)"""
        conv = QuantWanCausalConv3d(32, 64, kernel_size=3, padding=1)
        self.assertEqual(conv.padding, (0, 0, 0))

    def test_init_custom_padding(self):
        """测试 _padding 计算正确"""
        conv = QuantWanCausalConv3d(32, 64, kernel_size=3, padding=1)
        expected = (1, 1, 1, 1, 2, 0)
        self.assertEqual(conv._padding, expected)

    def test_init_quant_weight_true(self):
        """测试初始化时 quant_weight 为 True"""
        conv = QuantWanCausalConv3d(32, 64, kernel_size=3)
        self.assertTrue(conv.quant_weight)

    def test_init_dilations(self):
        """测试 dilations 默认值"""
        conv = QuantWanCausalConv3d(32, 64, kernel_size=3)
        self.assertEqual(conv.dilations, [1, 1, 1])

    def test_transform_5d_to_7d(self):
        """测试 5D → 7D 变换"""
        conv = QuantWanCausalConv3d(32, 64, kernel_size=3)
        # n=32, c=64, d=3, h=3, w=3
        # n0=16, c0=32 → n1=2, c1=2
        weight_5d = torch.randn(32, 64, 3, 3, 3)
        result = conv.transform_5d_to_7d(weight_5d, n0=16, c0=32)
        self.assertEqual(result.shape, (3, 2, 3, 3, 2, 16, 32))

    def test_transform_5d_to_7d_different_sizes(self):
        """测试不同尺寸的 5D→7D 变换"""
        conv = QuantWanCausalConv3d(16, 32, kernel_size=3)
        weight_5d = torch.randn(16, 32, 1, 1, 1)
        result = conv.transform_5d_to_7d(weight_5d, n0=16, c0=32)
        self.assertEqual(result.shape, (1, 1, 1, 1, 1, 16, 32))


# ═══════════════════════════════════════════════════════════════════════════
# 5. enable_conv3d_quant 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestEnableConv3dQuant(unittest.TestCase):
    def test_replaces_wan_causal_conv3d(self):
        """测试替换 WanCausalConv3d 类型的层"""
        class WanCausalConv3d(nn.Conv3d):
            def __init__(self):
                super().__init__(32, 64, kernel_size=3, padding=1)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = WanCausalConv3d()

        model = Model()
        result = enable_conv3d_quant(model)
        self.assertIsInstance(result.conv, QuantWanCausalConv3d)

    def test_skips_small_channels(self):
        """测试小通道数的层不被替换"""
        class WanCausalConv3d(nn.Conv3d):
            def __init__(self):
                super().__init__(8, 8, kernel_size=3, padding=1)  # 太小

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = WanCausalConv3d()

        model = Model()
        result = enable_conv3d_quant(model)
        # out_features=8 ≤ 16 或 in_features=8 ≤ 32 → 不替换
        self.assertNotIsInstance(result.conv, QuantWanCausalConv3d)

    def test_skips_non_conv3d(self):
        """测试非 Conv3d 层不受影响"""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16)

        model = Model()
        result = enable_conv3d_quant(model)
        self.assertIsInstance(result.linear, nn.Linear)

    def test_preserves_padding(self):
        """测试替换时保留 _padding 属性"""
        class WanCausalConv3d(nn.Conv3d):
            def __init__(self):
                super().__init__(32, 64, kernel_size=3, padding=1)
                self._padding = (1, 1, 1, 1, 2, 0)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = WanCausalConv3d()

        model = Model()
        result = enable_conv3d_quant(model)
        self.assertEqual(result.conv._padding, (1, 1, 1, 1, 2, 0))


# ═══════════════════════════════════════════════════════════════════════════
# 6. MatmulManager 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestMatmulManager(unittest.TestCase):
    def setUp(self):
        self.mgr = MatmulManager()

    @patch("x_base.operator.matmul.enable_weight_quant_linear")
    @patch("x_base.operator.matmul.dist")
    def test_enable_online_dynamic_quant_success(self, mock_dist, mock_eqwl):
        """测试在线动态量化成功"""
        mock_dist.is_initialized.return_value = False
        mock_eqwl.return_value = MagicMock()

        model = nn.Linear(32, 16)
        config = QuantLayerConfig()
        result = self.mgr.enable_online_dynamic_quant(model, config, dtype='w8a8')
        mock_eqwl.assert_called_once()

    @patch("x_base.operator.matmul.enable_weight_quant_linear")
    @patch("x_base.operator.matmul.dist")
    def test_enable_online_dynamic_quant_auto_config(self, mock_dist, mock_eqwl):
        """测试自动推断量化配置"""
        mock_dist.is_initialized.return_value = False
        mock_dist.get_rank.return_value = 0
        mock_eqwl.return_value = MagicMock()

        model = MagicMock()
        result = self.mgr.enable_online_dynamic_quant(model, quant_config=None)
        # 应调用 detect_config_name
        mock_eqwl.assert_called_once()

    @patch("x_base.operator.matmul.enable_weight_quant_linear", side_effect=RuntimeError("fail"))
    @patch("x_base.operator.matmul.dist")
    def test_enable_online_dynamic_quant_failure(self, mock_dist, mock_eqwl):
        """测试在线动态量化失败抛出 ValueError"""
        mock_dist.is_initialized.return_value = False
        model = nn.Linear(32, 16)
        config = QuantLayerConfig()
        with self.assertRaises(ValueError) as ctx:
            self.mgr.enable_online_dynamic_quant(model, config)
        self.assertIn("Dynamic weight modify model failed", str(ctx.exception))

    @patch("x_base.operator.matmul.enable_conv3d_quant")
    @patch("x_base.operator.matmul.dist")
    def test_enable_vae_conv3d_quant_success(self, mock_dist, mock_ecq):
        """测试 VAE Conv3D 量化成功"""
        mock_dist.get_rank.return_value = 0
        mock_ecq.return_value = MagicMock()
        vae = MagicMock()
        result = self.mgr.enable_vae_conv3d_quant(vae)
        mock_ecq.assert_called_once()

    @patch("x_base.operator.matmul.enable_conv3d_quant", side_effect=RuntimeError("fail"))
    @patch("x_base.operator.matmul.dist")
    def test_enable_vae_conv3d_quant_failure(self, mock_dist, mock_ecq):
        """测试 VAE Conv3D 量化失败抛出 ValueError"""
        mock_dist.get_rank.return_value = 0
        vae = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            self.mgr.enable_vae_conv3d_quant(vae)
        self.assertIn("Dynamic vae conv3d modify model failed", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════════
# 7. QuantLayerConfig 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestQuantLayerConfig(unittest.TestCase):
    def test_default_values(self):
        """测试默认值"""
        cfg = QuantLayerConfig()
        self.assertIsNone(cfg.include_patterns)
        self.assertEqual(cfg.exclude_patterns, [])
        self.assertEqual(cfg.layer_types, ["Linear"])

    def test_should_quantize_linear_layer(self):
        """测试 Linear 层默认应量化"""
        cfg = QuantLayerConfig()
        self.assertTrue(cfg.should_quantize("linear1", nn.Linear(32, 16)))

    def test_should_quantize_non_linear_layer(self):
        """测试非 Linear 层默认不量化"""
        cfg = QuantLayerConfig()
        self.assertFalse(cfg.should_quantize("relu", nn.ReLU()))

    def test_should_quantize_exclude_overrides(self):
        """测试黑名单优先级高于白名单"""
        cfg = QuantLayerConfig(
            include_patterns=["linear"],
            exclude_patterns=["linear1"]
        )
        self.assertFalse(cfg.should_quantize("linear1", nn.Linear(32, 16)))

    def test_should_quantize_include_filter(self):
        """测试白名单过滤"""
        cfg = QuantLayerConfig(include_patterns=["linear1"])
        self.assertTrue(cfg.should_quantize("linear1", nn.Linear(32, 16)))
        self.assertFalse(cfg.should_quantize("linear2", nn.Linear(32, 16)))

    def test_should_quantize_no_include_quantizes_all(self):
        """测试无白名单时量化所有匹配类型"""
        cfg = QuantLayerConfig()
        self.assertTrue(cfg.should_quantize("any_name", nn.Linear(32, 16)))

    def test_should_use_w4a4_both_match(self):
        """测试 w4a4 需同时匹配 patterns 和 block_patterns"""
        cfg = QuantLayerConfig(
            w4a4_patterns=["to_k"],
            w4a4_block_patterns=["blocks"]
        )
        self.assertTrue(cfg.should_use_w4a4("blocks.0.attn.to_k"))
        self.assertFalse(cfg.should_use_w4a4("blocks.0.attn.to_q"))  # to_q 不在 w4a4_patterns
        self.assertFalse(cfg.should_use_w4a4("other.to_k"))  # 不匹配 block

    def test_should_use_w4a4_default_patterns(self):
        """测试默认 w4a4 patterns"""
        cfg = QuantLayerConfig()
        # 默认 w4a4_patterns=["to_k", "to_v", "to_q", "to_out", "proj"]
        # 默认 w4a4_block_patterns=["blocks"]
        self.assertTrue(cfg.should_use_w4a4("blocks.0.attn.to_k"))
        self.assertTrue(cfg.should_use_w4a4("blocks.5.ff.proj"))
        self.assertFalse(cfg.should_use_w4a4("blocks.0.ff.fc1"))  # fc1 不在 w4a4_patterns

    def test_matches_patterns_glob(self):
        """测试 glob 模式匹配"""
        self.assertTrue(QuantLayerConfig._matches_patterns("blocks.0.attn", ["blocks.*.attn"]))
        self.assertFalse(QuantLayerConfig._matches_patterns("blocks.0.ff", ["blocks.*.attn"]))

    def test_matches_patterns_substring(self):
        """测试子串匹配"""
        self.assertTrue(QuantLayerConfig._matches_patterns("my_linear_layer", ["linear"]))
        self.assertFalse(QuantLayerConfig._matches_patterns("my_conv_layer", ["linear"]))

    def test_matches_patterns_empty(self):
        """测试空模式列表"""
        self.assertFalse(QuantLayerConfig._matches_patterns("any", []))

    def test_from_dict(self):
        """测试从字典创建"""
        d = {
            "include_patterns": ["blocks"],
            "exclude_patterns": ["bias"],
            "layer_types": ["Linear"],
            "w4a4_patterns": ["to_k"],
            "w4a4_block_patterns": ["blocks"]
        }
        cfg = QuantLayerConfig.from_dict(d)
        self.assertEqual(cfg.include_patterns, ["blocks"])
        self.assertEqual(cfg.exclude_patterns, ["bias"])
        self.assertEqual(cfg.w4a4_patterns, ["to_k"])

    def test_from_dict_defaults(self):
        """测试从空字典创建使用默认值"""
        cfg = QuantLayerConfig.from_dict({})
        self.assertIsNone(cfg.include_patterns)
        self.assertEqual(cfg.exclude_patterns, [])

    def test_post_init_parses_layer_types(self):
        """测试 __post_init__ 解析层类型"""
        cfg = QuantLayerConfig(layer_types=["Linear", "Conv2d"])
        self.assertIn(nn.Linear, cfg._parsed_layer_types)
        self.assertIn(nn.Conv2d, cfg._parsed_layer_types)

    def test_repr(self):
        """测试 __repr__"""
        cfg = QuantLayerConfig()
        r = repr(cfg)
        self.assertIn("QuantLayerConfig", r)

    def test_conv2d_layer_type(self):
        """测试 Conv2d 层类型"""
        cfg = QuantLayerConfig(layer_types=["Conv2d"])
        self.assertTrue(cfg.should_quantize("conv1", nn.Conv2d(3, 16, 3)))
        self.assertFalse(cfg.should_quantize("linear1", nn.Linear(32, 16)))


# ═══════════════════════════════════════════════════════════════════════════
# 8. QuantConfigManager 测试
# ═══════════════════════════════════════════════════════════════════════════
class TestQuantConfigManager(unittest.TestCase):
    def setUp(self):
        # 重置单例
        QuantConfigManager._instance = None

    @patch("x_base.config.config_loader.load_quant_config")
    def test_singleton(self, mock_load):
        """测试单例模式"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        m1 = QuantConfigManager()
        m2 = QuantConfigManager()
        self.assertIs(m1, m2)

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_exact_match(self, mock_load):
        """测试类名精确匹配"""
        mock_load.return_value = {
            "model_type_mapping": {"MyTransformer": "my_config"},
            "default": {}
        }
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "MyTransformer"
        model.__class__.__mro__ = (model.__class__, object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "my_config")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_base_class(self, mock_load):
        """测试基类链匹配"""
        mock_load.return_value = {
            "model_type_mapping": {"BaseTransformer": "base_config"},
            "default": {}
        }
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "DerivedTransformer"
        model.__class__.__mro__ = (model.__class__, type("BaseTransformer", (), {}), object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "base_config")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_fuzzy_qwen_image(self, mock_load):
        """测试模糊匹配 qwen_image"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "QwenImageTransformer"
        model.__class__.__mro__ = (model.__class__, object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "qwen_image")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_fuzzy_wan(self, mock_load):
        """测试模糊匹配 wan"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "WanTransformer3DModel"
        model.__class__.__mro__ = (model.__class__, object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "wan")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_fuzzy_hunyuan(self, mock_load):
        """测试模糊匹配 hunyuan"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "HunyuanVideoTransformer"
        model.__class__.__mro__ = (model.__class__, object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "hunyuan")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_fuzzy_cogvideo(self, mock_load):
        """测试模糊匹配 cogvideox"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "CogVideoXTransformer3DModel"
        model.__class__.__mro__ = (model.__class__, object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "cogvideox")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_detect_config_name_default(self, mock_load):
        """测试无匹配时返回 default"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        mgr = QuantConfigManager()
        model = MagicMock()
        model.__class__.__name__ = "UnknownModel"
        model.__class__.__mro__ = (model.__class__, object)
        result = mgr.detect_config_name(model)
        self.assertEqual(result, "default")

    @patch("x_base.config.config_loader.load_quant_config")
    def test_get_config_caches(self, mock_load):
        """测试配置缓存"""
        mock_load.return_value = {
            "model_type_mapping": {},
            "my_config": {"include_patterns": ["blocks"]},
            "default": {}
        }
        mgr = QuantConfigManager()
        c1 = mgr.get_config("my_config")
        c2 = mgr.get_config("my_config")
        self.assertIs(c1, c2)

    @patch("x_base.config.config_loader.load_quant_config")
    def test_get_config_fallback_to_default(self, mock_load):
        """测试配置不存在时回退到 default"""
        mock_load.return_value = {
            "model_type_mapping": {},
            "default": {"exclude_patterns": ["bias"]}
        }
        mgr = QuantConfigManager()
        cfg = mgr.get_config("nonexistent_config")
        self.assertIsInstance(cfg, QuantLayerConfig)

    @patch("x_base.config.config_loader.load_quant_config")
    @patch("x_base.config.config_loader.get_config")
    def test_reload(self, mock_get, mock_load):
        """测试 reload 清除缓存"""
        mock_load.return_value = {"model_type_mapping": {}, "default": {}}
        mock_get.return_value = {"model_type_mapping": {"new": "config"}, "default": {}}
        mgr = QuantConfigManager()
        mgr._configs["old"] = MagicMock()
        mgr.reload()
        self.assertEqual(len(mgr._configs), 0)


# ═══════════════════════════════════════════════════════════════════════════
# 9. matmul_manager 全局单例测试
# ═══════════════════════════════════════════════════════════════════════════
class TestMatmulManagerGlobal(unittest.TestCase):
    def test_global_instance(self):
        """测试全局 matmul_manager 实例"""
        self.assertIsInstance(matmul_manager, MatmulManager)


if __name__ == "__main__":
    unittest.main()
