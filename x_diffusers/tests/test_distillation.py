"""
Unit tests for x_diffusers/adaptor/distillation.py.

Covers:
- is_arm: arm/aarch64 detection, x86 negative
- generate_hardware_key: returns valid Fernet key
- encrypt_model_weights / decrypt_worker: round-trip encrypt-decrypt
- load_non_encryption_distillation_weights: non-Wan model returns None, file path, dir path, invalid path
- load_distillation_weights: dispatches by ENCRYPTION_WEIGHT flag
- update_wan_distillation_pipe_params: no x flag, VACE raises, Wan2.2 loads transformer_2
"""
import io
import os
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_diffusers.adaptor.distillation import (
    is_arm,
    generate_hardware_key,
    encrypt_model_weights,
    decrypt_worker,
    load_non_encryption_distillation_weights,
    load_distillation_weights,
    update_wan_distillation_pipe_params,
)


class TestIsArm:
    def test_arm64(self):
        with patch("x_diffusers.adaptor.distillation.platform") as mock_p:
            mock_p.machine.return_value = "aarch64"
            assert is_arm() is True

    def test_x86(self):
        with patch("x_diffusers.adaptor.distillation.platform") as mock_p:
            mock_p.machine.return_value = "x86_64"
            assert is_arm() is False


class TestGenerateHardwareKey:
    def test_returns_valid_fernet_key(self):
        key = generate_hardware_key()
        assert isinstance(key, bytes)
        assert len(key) == 44

    def test_deterministic(self):
        k1 = generate_hardware_key()
        k2 = generate_hardware_key()
        assert k1 == k2


class TestEncryptDecryptRoundTrip:
    def test_encrypt_and_decrypt(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        key = generate_hardware_key()
        output_dir = str(tmp_path / "encrypted")
        encrypt_model_weights(model, key, output_dir)
        assert os.path.isdir(output_dir)
        bin_files = [f for f in os.listdir(output_dir) if f.endswith(".bin")]
        assert len(bin_files) > 0
        for bf in bin_files:
            name, b = decrypt_worker(os.path.join(output_dir, bf), key)
            assert isinstance(name, str)
            assert isinstance(b, io.BytesIO)

    def test_decrypt_produces_valid_tensor(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        key = generate_hardware_key()
        output_dir = str(tmp_path / "enc")
        encrypt_model_weights(model, key, output_dir)
        first_bin = [f for f in os.listdir(output_dir) if f.endswith(".bin")][0]
        name, b = decrypt_worker(os.path.join(output_dir, first_bin), key)
        tensor = torch.load(b, weights_only=True)
        assert isinstance(tensor, torch.Tensor)


class TestLoadNonEncryptionDistillationWeights:
    def test_non_wan_returns_none(self):
        result = load_non_encryption_distillation_weights("CogVideoX", "/fake", {})
        assert result is None

    @patch("x_diffusers.adaptor.distillation.WanTransformer3DModel")
    @patch("x_diffusers.adaptor.distillation.os.path.isfile", return_value=True)
    def test_file_path_loads_state_dict(self, mock_isfile, mock_cls):
        mock_transformer = MagicMock()
        mock_cls.from_pretrained.return_value = mock_transformer
        with patch("x_diffusers.adaptor.distillation.load_file", return_value={}):
            result = load_non_encryption_distillation_weights("Wan2.1", "/fake/model.safetensors", {})
        assert result == mock_transformer

    @patch("x_diffusers.adaptor.distillation.WanTransformer3DModel")
    @patch("x_diffusers.adaptor.distillation.os.path.isfile", return_value=False)
    @patch("x_diffusers.adaptor.distillation.os.path.isdir", return_value=True)
    def test_dir_path_loads_from_pretrained(self, mock_isdir, mock_isfile, mock_cls):
        mock_cls.from_pretrained.return_value = MagicMock()
        result = load_non_encryption_distillation_weights("Wan2.1", "/fake/dir", {})
        assert mock_cls.from_pretrained.call_count == 1

    @patch("x_diffusers.adaptor.distillation.os.path.isfile", return_value=False)
    @patch("x_diffusers.adaptor.distillation.os.path.isdir", return_value=False)
    def test_invalid_path_returns_none(self, mock_isdir, mock_isfile):
        result = load_non_encryption_distillation_weights("Wan2.1", "/nonexistent", {})
        assert result is None


class TestLoadDistillationWeights:
    @patch("x_diffusers.adaptor.distillation.load_encryption_distillation_weights", return_value="enc_result")
    def test_encryption_path(self, mock_enc):
        with patch("x_diffusers.adaptor.distillation.ENCRYPTION_WEIGHT", True):
            result = load_distillation_weights("Wan2.1", "/path", {})
        assert result == "enc_result"

    @patch("x_diffusers.adaptor.distillation.load_non_encryption_distillation_weights", return_value="nonenc_result")
    def test_non_encryption_path(self, mock_nonenc):
        with patch("x_diffusers.adaptor.distillation.ENCRYPTION_WEIGHT", False):
            result = load_distillation_weights("Wan2.1", "/path", {})
        assert result == "nonenc_result"


class TestUpdateWanDistillationPipeParams:
    def test_no_x_flag_returns_unchanged(self):
        args = MagicMock()
        args.x = None
        params = {"key": "value"}
        result = update_wan_distillation_pipe_params(params, args)
        assert result == params

    @patch("x_diffusers.adaptor.distillation.load_distillation_model", return_value=MagicMock())
    def test_vace_model_raises(self, mock_load):
        args = MagicMock()
        args.x = "/path"
        args.model = "Wan2.1-VACE"
        with pytest.raises(ValueError, match="does not yet support"):
            update_wan_distillation_pipe_params({}, args)

    @patch("x_diffusers.adaptor.distillation.load_distillation_model", return_value=MagicMock())
    def test_wan22_loads_transformer_2(self, mock_load):
        args = MagicMock()
        args.x = "/path"
        args.model = "Wan2.2-T2V-14B"
        args.pretrained_model_name_or_path = "/model"
        args.x_model_path = "/x"
        args.x_model_path_2 = "/x2"
        params = {}
        result = update_wan_distillation_pipe_params(params, args)
        assert "transformer" in result
        assert mock_load.call_count == 2
