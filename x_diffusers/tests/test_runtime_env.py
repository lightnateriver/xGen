import pytest
from unittest.mock import MagicMock, patch
from types import ModuleType
import importlib
import sys


@pytest.fixture(autouse=True)
def npu_runtime_contract(monkeypatch):
    # Exercise the existing NPU startup branch without loading vendor packages
    # or creating a real process group. Function-level patches supply devices.
    cfg = ModuleType('x_base.cfg_parallelism.utils')
    cfg.init_CFG_Parallel = MagicMock()
    from x_diffusers import adaptor
    previous = getattr(adaptor, 'runtime_env', None)
    try:
        with (patch.dict(sys.modules, {'x_base.cfg_parallelism.utils': cfg}),
              patch('x_diffusers.framework.initialize_framework')):
            runtime_env = importlib.import_module('x_diffusers.adaptor.runtime_env')
            monkeypatch.setattr(runtime_env, 'device_type', lambda: 'npu')
            monkeypatch.setattr(runtime_env, 'torch_npu', MagicMock(), raising=False)
            yield
    finally:
        if previous is None:
            delattr(adaptor, 'runtime_env')
        else:
            adaptor.runtime_env = previous


class TestInitEnv:
    @patch("x_diffusers.adaptor.runtime_env.dist")
    @patch("x_diffusers.adaptor.runtime_env.torch_npu")
    @patch("x_diffusers.adaptor.runtime_env.torch")
    @patch("x_diffusers.adaptor.runtime_env.os")
    def test_init_env_default_env(self, mock_os, mock_torch, mock_torch_npu, mock_dist):
        mock_os.getenv.side_effect = lambda key, default=None: {"RANK": "0", "WORLD_SIZE": "1"}.get(key, default)

        from x_diffusers.adaptor.runtime_env import init_env
        init_env()

        mock_torch.set_num_threads.assert_called_with(24)
        mock_dist.init_process_group.assert_called_once()
        mock_torch_npu.npu.set_device.assert_called_with(0)

    @patch("x_diffusers.adaptor.runtime_env.dist")
    @patch("x_diffusers.adaptor.runtime_env.torch_npu")
    @patch("x_diffusers.adaptor.runtime_env.torch")
    @patch("x_diffusers.adaptor.runtime_env.os")
    def test_init_env_custom_rank(self, mock_os, mock_torch, mock_torch_npu, mock_dist):
        mock_os.getenv.side_effect = lambda key, default=None: {"RANK": "3", "WORLD_SIZE": "8"}.get(key, default)

        from x_diffusers.adaptor.runtime_env import init_env
        init_env()

        mock_dist.init_process_group.assert_called_with(
            backend="nccl", init_method="env://", world_size=8, rank=3,
        )
        mock_torch_npu.npu.set_device.assert_called_with(3)


class TestInitCfgEnv:
    @patch("x_diffusers.adaptor.runtime_env.init_CFG_Parallel")
    @patch("x_diffusers.adaptor.runtime_env.init_env")
    def test_init_cfg_env(self, mock_init_env, mock_init_cfg):
        from x_diffusers.adaptor.runtime_env import init_cfg_env
        init_cfg_env(2)

        mock_init_env.assert_called_once()
        mock_init_cfg.assert_called_with(2)

    @patch("x_diffusers.adaptor.runtime_env.init_CFG_Parallel")
    @patch("x_diffusers.adaptor.runtime_env.init_env")
    def test_init_cfg_env_size_4(self, mock_init_env, mock_init_cfg):
        from x_diffusers.adaptor.runtime_env import init_cfg_env
        init_cfg_env(4)

        mock_init_cfg.assert_called_with(4)
