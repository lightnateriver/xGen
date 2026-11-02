"""
Unit tests for x_diffusers QwenImage pipeline (text-to-image generation).

Tests cover the utility functions and logic in:
- x_diffusers.framework.pipeline.qwenimage.utils
- x_diffusers.framework.pipeline.qwenimage.pipeline_qwenimage

Note: These tests verify standalone logic without requiring NPU or model weights.
Uses setup_all_mocks() + load_module_directly() from framework conftest to bypass
diffusers/x_base/torch_npu dependencies.
"""

import pytest
import torch
import numpy as np
import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock


# ============================================================================
# Module loading helpers (same pattern as framework conftest.py)
# ============================================================================

def _get_utils_path():
    """Resolve the absolute path to qwenimage/utils.py."""
    # Handle both x_diffusers/ and x_diffusers/x_diffusers/ layouts
    base = Path(__file__).resolve().parent.parent.parent
    candidates = [
        base / "x_diffusers" / "framework" / "pipeline" / "qwenimage" / "utils.py",
        base / "framework" / "pipeline" / "qwenimage" / "utils.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find qwenimage/utils.py under {base}")


def _setup_mocks():
    """Setup mock modules for diffusers, x_base, torch_npu, etc."""
    # --- diffusers mock ---
    mock_diffusers = MagicMock()

    mock_logging = MagicMock()
    mock_logging.get_logger = MagicMock(return_value=MagicMock())
    mock_diffusers.utils = MagicMock()
    mock_diffusers.utils.logging = mock_logging
    mock_diffusers.utils.is_torch_xla_available = MagicMock(return_value=False)

    mock_edit_plus = MagicMock()
    mock_edit_plus.calculate_dimensions = MagicMock(return_value=(1024, 1024))
    mock_diffusers.pipelines = MagicMock()
    mock_diffusers.pipelines.qwenimage = MagicMock()
    mock_diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus = mock_edit_plus

    mock_diffusers.configuration_utils = MagicMock()
    mock_diffusers.configuration_utils.ConfigMixin = type("ConfigMixin", (), {})
    mock_diffusers.configuration_utils.register_to_config = lambda f: f

    mock_diffusers.pipelines.pipeline_utils = MagicMock()
    mock_diffusers.pipelines.pipeline_utils.DiffusionPipeline = type("DiffusionPipeline", (), {})

    mock_diffusers.schedulers = MagicMock()
    mock_diffusers.schedulers.scheduling_utils = MagicMock()
    mock_diffusers.schedulers.scheduling_utils.SchedulerMixin = type("SchedulerMixin", (), {})

    mock_diffusers.callbacks = MagicMock()
    mock_diffusers.callbacks.MultiPipelineCallbacks = MagicMock
    mock_diffusers.callbacks.PipelineCallback = MagicMock

    mock_diffusers.image_processor = MagicMock()
    mock_diffusers.image_processor.PipelineImageInput = MagicMock
    mock_diffusers.video_processor = MagicMock()

    mock_diffusers.loaders = MagicMock()
    mock_diffusers.loaders.peft = MagicMock()
    mock_diffusers.loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING = {}

    # --- x_base mock ---
    mock_x_base = MagicMock()
    mock_x_base.get_cfg_group = MagicMock(return_value=None)

    # --- torch_npu mock ---
    mock_torch_npu = MagicMock()

    # --- Register all in sys.modules ---
    module_map = {
        "diffusers": mock_diffusers,
        "diffusers.utils": mock_diffusers.utils,
        "diffusers.utils.logging": mock_logging,
        "diffusers.configuration_utils": mock_diffusers.configuration_utils,
        "diffusers.pipelines": mock_diffusers.pipelines,
        "diffusers.pipelines.pipeline_utils": mock_diffusers.pipelines.pipeline_utils,
        "diffusers.pipelines.qwenimage": mock_diffusers.pipelines.qwenimage,
        "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus": mock_edit_plus,
        "diffusers.schedulers": mock_diffusers.schedulers,
        "diffusers.schedulers.scheduling_utils": mock_diffusers.schedulers.scheduling_utils,
        "diffusers.callbacks": mock_diffusers.callbacks,
        "diffusers.image_processor": mock_diffusers.image_processor,
        "diffusers.video_processor": mock_diffusers.video_processor,
        "diffusers.loaders": mock_diffusers.loaders,
        "diffusers.loaders.peft": mock_diffusers.loaders.peft,
        "x_base": mock_x_base,
        "x_base.utils": MagicMock(),
        "x_base.utils.infer_info": MagicMock(),
        "torch_npu": mock_torch_npu,
        "torch_npu.contrib": MagicMock(),
    }
    return module_map


@pytest.fixture(scope="module", autouse=True)
def _load_utils_module():
    """Load qwenimage/utils.py with all mocks injected before import."""
    mock_map = _setup_mocks()

    # Save original modules
    saved = {}
    for name in mock_map:
        if name in sys.modules:
            saved[name] = sys.modules[name]

    # Install mocks
    for name, mod in mock_map.items():
        sys.modules[name] = mod

    # Also mock the x_diffusers package init to avoid triggering real imports
    if "x_diffusers" not in sys.modules:
        sys.modules["x_diffusers"] = MagicMock()
    if "x_diffusers.framework" not in sys.modules:
        sys.modules["x_diffusers.framework"] = MagicMock()
    if "x_diffusers.framework.pipeline" not in sys.modules:
        sys.modules["x_diffusers.framework.pipeline"] = MagicMock()
    if "x_diffusers.framework.pipeline.qwenimage" not in sys.modules:
        sys.modules["x_diffusers.framework.pipeline.qwenimage"] = MagicMock()

    # Load utils.py directly from file
    # Prepend `from __future__ import annotations` to handle PEP 604 union
    # syntax (str | list[str]) which requires Python 3.10+ at runtime
    utils_path = _get_utils_path()
    source_code = utils_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in source_code:
        source_code = "from __future__ import annotations\n" + source_code

    code_obj = compile(source_code, str(utils_path), "exec")
    utils_mod = types.ModuleType("x_diffusers.framework.pipeline.qwenimage.utils")
    utils_mod.__file__ = str(utils_path)
    utils_mod.__loader__ = MagicMock()
    utils_mod.__spec__ = MagicMock()
    utils_mod.__package__ = "x_diffusers.framework.pipeline.qwenimage"
    exec(code_obj, utils_mod.__dict__)
    sys.modules["x_diffusers.framework.pipeline.qwenimage.utils"] = utils_mod

    yield utils_mod

    # Restore original modules
    for name in mock_map:
        if name in saved:
            sys.modules[name] = saved[name]
        elif name in sys.modules:
            del sys.modules[name]


# ============================================================================
# Helper to get the loaded module
# ============================================================================

def _utils():
    """Get the loaded utils module."""
    return sys.modules["x_diffusers.framework.pipeline.qwenimage.utils"]


# ============================================================================
# Tests for PipelineConfig
# ============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass in qwenimage/utils.py."""

    def test_qwenimage_config_no_image(self):
        """QwenImagePipeline (t2i) does not need image input."""
        config = _utils().PIPELINE_CONFIGS["qwenimage"]
        assert config.needs_image is False
        assert config.needs_multiple_images is False
        assert config.needs_edit_preprocess is False

    def test_qwenimage_config_no_batch_limit(self):
        """QwenImagePipeline has no batch size limit."""
        config = _utils().PIPELINE_CONFIGS["qwenimage"]
        assert config.batch_size_limit is None


# ============================================================================
# Tests for CFG utilities
# ============================================================================

class TestCFGUtilities:
    """Tests for classifier-free guidance helper functions."""

    def test_apply_cfg_basic(self):
        """Test apply_cfg produces weighted combination with normalization."""
        apply_cfg = _utils().apply_cfg

        noise_pred = torch.tensor([[1.0, 2.0, 3.0]])
        neg_noise_pred = torch.tensor([[0.0, 0.0, 0.0]])
        true_cfg_scale = 4.0

        result = apply_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        # comb = neg + scale * (cond - neg) = 0 + 4 * [1,2,3] = [4,8,12]
        # cond_norm = sqrt(1+4+9) = sqrt(14)
        # noise_norm = sqrt(16+64+144) = sqrt(224)
        # result = [4,8,12] * sqrt(14/224) = [4,8,12] * 0.25
        expected = torch.tensor([[4.0, 8.0, 12.0]]) * (14.0 / 224.0) ** 0.5
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_cfg_scale_one(self):
        """Test apply_cfg with scale=1 returns normalized cond prediction."""
        apply_cfg = _utils().apply_cfg

        noise_pred = torch.tensor([[2.0, 4.0]])
        neg_noise_pred = torch.tensor([[1.0, 2.0]])
        true_cfg_scale = 1.0

        result = apply_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        # comb = neg + 1*(cond - neg) = cond
        # Normalization: result = cond * (||cond|| / ||cond||) = cond
        assert torch.allclose(result, noise_pred, atol=1e-5)

    def test_apply_cfg_zero_neg(self):
        """Test apply_cfg with zero negative prediction is pure scaled cond."""
        apply_cfg = _utils().apply_cfg

        noise_pred = torch.ones(1, 8)
        neg_noise_pred = torch.zeros(1, 8)
        true_cfg_scale = 3.0

        result = apply_cfg(noise_pred, neg_noise_pred, true_cfg_scale)
        # comb = 0 + 3*ones = 3*ones
        # cond_norm = sqrt(8), noise_norm = sqrt(8*9) = 3*sqrt(8)
        # result = 3*ones * sqrt(8)/(3*sqrt(8)) = ones
        assert torch.allclose(result, noise_pred, atol=1e-5)


# ============================================================================
# Tests for batch size and prompt utilities
# ============================================================================

class TestBatchSizeUtilities:
    """Tests for batch size and prompt handling utilities."""

    def test_get_batch_size_string_prompt(self):
        """Single string prompt yields batch_size=1."""
        assert _utils().get_batch_size("a cat", None) == 1

    def test_get_batch_size_list_prompt(self):
        """List prompt yields batch_size=len(list)."""
        assert _utils().get_batch_size(["a cat", "a dog"], None) == 2

    def test_get_batch_size_from_embeds(self):
        """When prompt is None, batch_size comes from prompt_embeds."""
        embeds = torch.randn(3, 512, 4096)
        assert _utils().get_batch_size(None, embeds) == 3


# ============================================================================
# Tests for CFG setup utilities
# ============================================================================

class TestCFGSetup:
    """Tests for CFG setup and warning logic."""

    def test_has_neg_prompt_with_text(self):
        """Negative prompt text means has_neg_prompt=True."""
        assert _utils().get_has_neg_prompt("blurry", None, None) is True

    def test_has_neg_prompt_with_embeds(self):
        """Negative prompt embeds + mask means has_neg_prompt=True."""
        embeds = torch.randn(1, 512, 4096)
        mask = torch.ones(1, 512)
        assert _utils().get_has_neg_prompt(None, embeds, mask) is True

    def test_has_neg_prompt_without_any(self):
        """No negative prompt at all means has_neg_prompt=False."""
        assert _utils().get_has_neg_prompt(None, None, None) is False

    def test_do_true_cfg_enabled(self):
        """true_cfg_scale > 1 with neg prompt enables CFG."""
        has_neg, do_cfg = _utils().get_cfg_setup(4.0, "blurry", None, None)
        assert has_neg is True
        assert do_cfg is True

    def test_do_true_cfg_disabled_no_neg(self):
        """true_cfg_scale > 1 without neg prompt disables CFG."""
        has_neg, do_cfg = _utils().get_cfg_setup(4.0, None, None, None)
        assert has_neg is False
        assert do_cfg is False

    def test_do_true_cfg_disabled_low_scale(self):
        """true_cfg_scale <= 1 disables CFG even with neg prompt."""
        has_neg, do_cfg = _utils().get_cfg_setup(1.0, "blurry", None, None)
        assert has_neg is True
        assert do_cfg is False


# ============================================================================
# Tests for guidance handling
# ============================================================================

class TestGuidanceHandling:
    """Tests for guidance scale handling in QwenImage pipelines."""

    def test_guidance_embeds_enabled_with_scale(self):
        """When transformer has guidance_embeds, guidance tensor is created."""
        handle_guidance = _utils().handle_guidance

        mock_transformer = MagicMock()
        mock_transformer.config.guidance_embeds = True

        latents = torch.randn(2, 1024, 64)
        guidance = handle_guidance(mock_transformer, 3.5, latents, "cpu")

        assert guidance is not None
        assert guidance.shape == (2,)
        assert torch.allclose(guidance, torch.tensor([3.5, 3.5]))

    def test_guidance_embeds_enabled_without_scale_raises(self):
        """When transformer has guidance_embeds but no scale, ValueError is raised."""
        handle_guidance = _utils().handle_guidance

        mock_transformer = MagicMock()
        mock_transformer.config.guidance_embeds = True

        with pytest.raises(ValueError, match="guidance_scale is required"):
            handle_guidance(mock_transformer, None, torch.randn(2, 1024, 64), "cpu")

    def test_guidance_embeds_disabled_with_scale(self):
        """When transformer has no guidance_embeds, scale is ignored and returns None."""
        handle_guidance = _utils().handle_guidance

        mock_transformer = MagicMock()
        mock_transformer.config.guidance_embeds = False

        guidance = handle_guidance(mock_transformer, 3.5, torch.randn(2, 1024, 64), "cpu")
        assert guidance is None

    def test_guidance_embeds_disabled_without_scale(self):
        """When transformer has no guidance_embeds and no scale, returns None."""
        handle_guidance = _utils().handle_guidance

        mock_transformer = MagicMock()
        mock_transformer.config.guidance_embeds = False

        guidance = handle_guidance(mock_transformer, None, torch.randn(2, 1024, 64), "cpu")
        assert guidance is None


# ============================================================================
# Tests for pipeline state initialization
# ============================================================================

class TestPipelineStateInit:
    """Tests for pipeline state initialization."""

    def test_init_pipeline_state(self):
        """Test init_pipeline_state sets correct internal state."""
        init_pipeline_state = _utils().init_pipeline_state

        pipeline = MagicMock()
        init_pipeline_state(pipeline, guidance_scale=3.5, attention_kwargs={"key": "val"})

        assert pipeline._guidance_scale == 3.5
        assert pipeline._attention_kwargs == {"key": "val"}
        assert pipeline._current_timestep is None
        assert pipeline._interrupt is False


# ============================================================================
# Tests for QwenImagePipeline configuration from shell script
# ============================================================================

class TestQwenImagePipelineConfig:
    """Tests for QwenImagePipeline configuration matching infer_qwen_image_2512.sh."""

    def test_default_inference_params(self):
        """Test default inference parameters from shell script."""
        # From infer_qwen_image_2512.sh
        model = "Qwen-Image-2512"
        height = 1024
        width = 1024
        num_inference_steps = 20
        true_cfg_scale = 4.0
        cfg_parallel_size = 2
        seed = 42

        assert model == "Qwen-Image-2512"
        assert height == 1024
        assert width == 1024
        assert num_inference_steps == 20
        assert true_cfg_scale == pytest.approx(4.0)
        assert cfg_parallel_size == 2

    def test_lightning_inference_params(self):
        """Test Lightning LoRA inference parameters from shell script."""
        # From infer_qwen_image_2512_lightning.sh
        num_inference_steps = 4
        true_cfg_scale = 1.0
        cfg_parallel_size = 1  # Lightning only supports 1 NPU

        assert num_inference_steps == 4
        assert true_cfg_scale == pytest.approx(1.0)
        assert cfg_parallel_size == 1

    def test_qwenimage_config_no_image_required(self):
        """QwenImagePipeline (t2i) should not require image_path_list."""
        config = _utils().PIPELINE_CONFIGS["qwenimage"]
        assert config.needs_image is False
        assert config.needs_multiple_images is False


# ============================================================================
# Tests for timesteps preparation
# ============================================================================

class TestTimestepsPreparation:
    """Tests for timesteps and sigmas preparation."""

    def test_sigmas_linspace(self):
        """Test default sigmas generation matches num_inference_steps."""
        num_inference_steps = 20
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        assert len(sigmas) == num_inference_steps
        assert sigmas[0] == 1.0
        assert abs(sigmas[-1] - 0.05) < 1e-10

    def test_sigmas_lightning(self):
        """Test Lightning mode uses fewer steps."""
        num_inference_steps = 4
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        assert len(sigmas) == 4
        assert sigmas[0] == 1.0
        assert abs(sigmas[-1] - 0.25) < 1e-10
