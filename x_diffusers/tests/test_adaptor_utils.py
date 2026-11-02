import pytest
from unittest.mock import MagicMock, patch


class TestInitEnv:
    @patch("x_diffusers.adaptor.utils.dist")
    @patch("x_diffusers.adaptor.utils.torch_npu")
    @patch("x_diffusers.adaptor.utils.torch")
    @patch("x_diffusers.adaptor.utils.os")
    def test_init_env_default_env(self, mock_os, mock_torch, mock_torch_npu, mock_dist):
        mock_os.getenv.side_effect = lambda key, default=None: {"RANK": "0", "WORLD_SIZE": "1"}.get(key, default)

        from x_diffusers.adaptor.utils import init_env
        init_env()

        mock_torch.set_num_threads.assert_called_with(24)
        mock_dist.init_process_group.assert_called_once()
        mock_torch_npu.npu.set_device.assert_called_with(0)

    @patch("x_diffusers.adaptor.utils.dist")
    @patch("x_diffusers.adaptor.utils.torch_npu")
    @patch("x_diffusers.adaptor.utils.torch")
    @patch("x_diffusers.adaptor.utils.os")
    def test_init_env_custom_rank(self, mock_os, mock_torch, mock_torch_npu, mock_dist):
        mock_os.getenv.side_effect = lambda key, default=None: {"RANK": "3", "WORLD_SIZE": "8"}.get(key, default)

        from x_diffusers.adaptor.utils import init_env
        init_env()

        mock_dist.init_process_group.assert_called_with(
            backend="nccl", init_method="env://", world_size=8, rank=3,
        )
        mock_torch_npu.npu.set_device.assert_called_with(3)


class TestInitCfgEnv:
    @patch("x_diffusers.adaptor.utils.init_CFG_Parallel")
    @patch("x_diffusers.adaptor.utils.init_env")
    def test_init_cfg_env(self, mock_init_env, mock_init_cfg):
        from x_diffusers.adaptor.utils import init_cfg_env
        init_cfg_env(2)

        mock_init_env.assert_called_once()
        mock_init_cfg.assert_called_with(2)

    @patch("x_diffusers.adaptor.utils.init_CFG_Parallel")
    @patch("x_diffusers.adaptor.utils.init_env")
    def test_init_cfg_env_size_4(self, mock_init_env, mock_init_cfg):
        from x_diffusers.adaptor.utils import init_cfg_env
        init_cfg_env(4)

        mock_init_cfg.assert_called_with(4)
