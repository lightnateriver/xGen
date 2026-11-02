"""
Unit tests for fsdp/fsdp.py.

Covers:
- transformer_fsdp: is_need_dtype conversion, transformer_2 wrapping
- transformer_lambda_fsdp: is_need_dtype, WeightQuantLinearModule ignored, FSDP wrapping
- text_encoder_fsdp: t5/llama model_type, unsupported type raises
- fsdp_init: dist not available raises, transformer/all/text_encoder dispatch, Wan model branch
"""
from unittest.mock import patch, MagicMock, call

import pytest
import torch

from x_base.fsdp.fsdp import (
    transformer_fsdp,
    transformer_lambda_fsdp,
    text_encoder_fsdp,
    fsdp_init,
)


class TestTransformerFsdp:
    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    @patch("x_base.fsdp.fsdp.torch.npu.current_device", return_value=0)
    def test_wraps_transformer(self, mock_dev, mock_fsdp):
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        transformer_fsdp(pipe, is_need_dtype=False)
        mock_fsdp.assert_called_once()
        assert pipe.transformer == mock_fsdp.return_value

    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    @patch("x_base.fsdp.fsdp.torch.npu.current_device", return_value=0)
    def test_is_need_dtype_converts(self, mock_dev, mock_fsdp):
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        transformer_fsdp(pipe, is_need_dtype=True)
        pipe.transformer.to.assert_called_once_with(torch.bfloat16)

    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    @patch("x_base.fsdp.fsdp.torch.npu.current_device", return_value=0)
    def test_transformer_2_wrapped(self, mock_dev, mock_fsdp):
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        pipe.transformer_2 = MagicMock()
        transformer_fsdp(pipe, is_need_dtype=False, transformer_2=True)
        assert mock_fsdp.call_count == 2

    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    @patch("x_base.fsdp.fsdp.torch.npu.current_device", return_value=0)
    def test_transformer_2_not_wrapped_by_default(self, mock_dev, mock_fsdp):
        pipe = MagicMock()
        pipe.transformer = MagicMock()
        transformer_fsdp(pipe, is_need_dtype=False)
        assert mock_fsdp.call_count == 1


class TestTransformerLambdaFsdp:
    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    def test_wraps_model(self, mock_fsdp):
        model = MagicMock()
        model.blocks = [MagicMock()]
        model.named_children.return_value = []
        result = transformer_lambda_fsdp(model, device_id=0, is_need_dtype=False)
        mock_fsdp.assert_called_once()
        assert result == mock_fsdp.return_value

    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    def test_is_need_dtype_converts(self, mock_fsdp):
        model = MagicMock()
        model.blocks = [MagicMock()]
        model.named_children.return_value = []
        transformer_lambda_fsdp(model, device_id=0, is_need_dtype=True)
        model.to.assert_called_once()

    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    def test_finds_weight_quant_layers(self, mock_fsdp):
        from x_base.operator.matmul import WeightQuantLinearModule
        wq = MagicMock(spec=WeightQuantLinearModule)
        child = MagicMock()
        model = MagicMock()
        model.blocks = [MagicMock()]
        model.named_children.return_value = [("wq", wq)]
        wq.named_children.return_value = []
        transformer_lambda_fsdp(model, device_id=0, is_need_dtype=False)
        call_kwargs = mock_fsdp.call_args[1]
        assert wq in call_kwargs["ignored_modules"]


class TestTextEncoderFsdp:
    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    @patch("x_base.fsdp.fsdp.torch.npu.current_device", return_value=0)
    def test_t5_converts_dtype(self, mock_dev, mock_fsdp):
        pipe = MagicMock()
        text_encoder_fsdp(pipe, "t5")
        pipe.text_encoder.to.assert_called_once_with(torch.bfloat16)

    @patch("x_base.fsdp.fsdp.FSDP", return_value=MagicMock())
    @patch("x_base.fsdp.fsdp.torch.npu.current_device", return_value=0)
    def test_llama_no_dtype_convert(self, mock_dev, mock_fsdp):
        pipe = MagicMock()
        text_encoder_fsdp(pipe, "llama")
        pipe.text_encoder.to.assert_not_called()

    def test_unsupported_model_type_raises(self):
        pipe = MagicMock()
        with pytest.raises(ValueError, match="Unsupported"):
            text_encoder_fsdp(pipe, "gpt")


class TestFsdpInit:
    def test_dist_not_available_raises(self):
        with patch("x_base.fsdp.fsdp.dist") as mock_dist:
            mock_dist.is_available.return_value = False
            with pytest.raises(RuntimeError, match="not available"):
                fsdp_init(MagicMock(), MagicMock(), False, "t5")

    @patch("x_base.fsdp.fsdp.transformer_fsdp")
    @patch("x_base.fsdp.fsdp.dist")
    def test_transformer_dispatch(self, mock_dist, mock_tfsdp):
        mock_dist.is_available.return_value = True
        pipe = MagicMock()
        args = MagicMock()
        args.fsdp = "transformer"
        args.model = "CogVideoX"
        fsdp_init(pipe, args, False, "t5")
        mock_tfsdp.assert_called_once()

    @patch("x_base.fsdp.fsdp.transformer_lambda_fsdp")
    @patch("x_base.fsdp.fsdp.dist")
    def test_wan_model_uses_lambda_fsdp(self, mock_dist, mock_lambda):
        mock_dist.is_available.return_value = True
        pipe = MagicMock()
        args = MagicMock()
        args.fsdp = "transformer"
        args.model = "Wan2.1"
        args.joint = False
        fsdp_init(pipe, args, False, "t5")
        mock_lambda.assert_called_once()

    @patch("x_base.fsdp.fsdp.text_encoder_fsdp")
    @patch("x_base.fsdp.fsdp.dist")
    def test_text_encoder_dispatch(self, mock_dist, mock_tefsdp):
        mock_dist.is_available.return_value = True
        pipe = MagicMock()
        args = MagicMock()
        args.fsdp = "text_encoder"
        fsdp_init(pipe, args, False, "t5")
        mock_tefsdp.assert_called_once()

    @patch("x_base.fsdp.fsdp.transformer_fsdp")
    @patch("x_base.fsdp.fsdp.text_encoder_fsdp")
    @patch("x_base.fsdp.fsdp.dist")
    def test_all_dispatches_both(self, mock_dist, mock_tefsdp, mock_tfsdp):
        mock_dist.is_available.return_value = True
        pipe = MagicMock()
        args = MagicMock()
        args.fsdp = "all"
        args.model = "CogVideoX"
        fsdp_init(pipe, args, False, "t5")
        mock_tfsdp.assert_called_once()
        mock_tefsdp.assert_called_once()

    @patch("x_base.fsdp.fsdp.transformer_lambda_fsdp")
    @patch("x_base.fsdp.fsdp.dist")
    def test_wan_joint_wraps_transformer_3(self, mock_dist, mock_lambda):
        mock_dist.is_available.return_value = True
        pipe = MagicMock()
        args = MagicMock()
        args.fsdp = "transformer"
        args.model = "Wan2.1"
        args.joint = True
        fsdp_init(pipe, args, False, "t5", transformer_2=True)
        assert mock_lambda.call_count == 3
