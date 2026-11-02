import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from x_diffusers.framework.pipeline.longcat_image.base_longcat_image import (
    BaseLongCatImagePipelineMixin,
)


def _make_mixin(**kwargs):
    mixin = BaseLongCatImagePipelineMixin()
    for k, v in kwargs.items():
        setattr(mixin, k, v)
    return mixin


def _make_scheduler(order=1):
    scheduler = MagicMock()
    scheduler.order = order
    scheduler.config = MagicMock()
    scheduler.config.get = lambda key, default=None: {
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.15,
    }.get(key, default)
    return scheduler


class TestPrepareTimesteps:
    def test_default_sigmas(self):
        mixin = _make_mixin(scheduler=_make_scheduler())
        latents = torch.randn(1, 64, 4)
        device = torch.device("cpu")

        mock_timesteps = [torch.tensor(0.9), torch.tensor(0.1)]
        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.retrieve_timesteps",
            return_value=(mock_timesteps, 2),
        ):
            timesteps, num_steps, warmup = mixin._prepare_timesteps(2, None, latents, device)

        assert num_steps == 2
        assert warmup == max(len(mock_timesteps) - 2 * 1, 0)
        assert mixin._num_timesteps == len(mock_timesteps)

    def test_custom_sigmas(self):
        mixin = _make_mixin(scheduler=_make_scheduler())
        latents = torch.randn(1, 64, 4)
        device = torch.device("cpu")
        sigmas = [1.0, 0.5, 0.0]

        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.retrieve_timesteps",
            return_value=([torch.tensor(0.5)], 1),
        ):
            timesteps, num_steps, warmup = mixin._prepare_timesteps(3, sigmas, latents, device)

        assert num_steps == 1

    def test_warmup_steps_zero(self):
        mixin = _make_mixin(scheduler=_make_scheduler(order=10))
        latents = torch.randn(1, 64, 4)
        device = torch.device("cpu")

        ts = [torch.tensor(0.9)]
        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.retrieve_timesteps",
            return_value=(ts, 1),
        ):
            _, _, warmup = mixin._prepare_timesteps(1, None, latents, device)

        assert warmup == max(1 - 1 * 10, 0)


class TestDenoiseStepNonParallel:
    def test_no_cfg(self):
        mixin = _make_mixin(do_classifier_free_guidance=False)
        transformer = MagicMock()
        transformer.return_value = (torch.randn(1, 10, 4),)
        mixin.transformer = transformer

        latent_model_input = torch.randn(1, 10, 4)
        timestep = torch.tensor([0.5])
        prompt_embeds = torch.randn(1, 5, 4)
        text_ids = torch.randn(5, 3)
        latent_image_ids = torch.randn(10, 3)

        result = mixin._denoise_step_non_parallel(
            latent_model_input, timestep, prompt_embeds, None,
            text_ids, None, latent_image_ids, False, 0.0, None,
        )

        assert result.shape == (1, 10, 4)
        transformer.assert_called_once()

    def test_with_cfg(self):
        mixin = _make_mixin(
            do_classifier_free_guidance=True,
            guidance_scale=3.0,
        )
        transformer = MagicMock()
        transformer.return_value = (torch.randn(1, 10, 4),)
        mixin.transformer = transformer

        latent_model_input = torch.randn(1, 10, 4)
        timestep = torch.tensor([0.5])
        prompt_embeds = torch.randn(1, 5, 4)
        neg_prompt_embeds = torch.randn(1, 5, 4)
        text_ids = torch.randn(5, 3)
        neg_text_ids = torch.randn(5, 3)
        latent_image_ids = torch.randn(10, 3)

        result = mixin._denoise_step_non_parallel(
            latent_model_input, timestep, prompt_embeds, neg_prompt_embeds,
            text_ids, neg_text_ids, latent_image_ids, False, 0.0, None,
        )

        assert result.shape == (1, 10, 4)
        assert transformer.call_count == 2

    def test_with_image_seq_len(self):
        mixin = _make_mixin(do_classifier_free_guidance=False)
        transformer = MagicMock()
        transformer.return_value = (torch.randn(1, 20, 4),)
        mixin.transformer = transformer

        result = mixin._denoise_step_non_parallel(
            torch.randn(1, 20, 4), torch.tensor([0.5]),
            torch.randn(1, 5, 4), None,
            torch.randn(5, 3), None, torch.randn(20, 3),
            False, 0.0, 10,
        )

        assert result.shape == (1, 10, 4)

    def test_cfg_renorm(self):
        mixin = _make_mixin(
            do_classifier_free_guidance=True,
            guidance_scale=2.0,
        )
        transformer = MagicMock()
        transformer.return_value = (torch.ones(1, 10, 4),)
        mixin.transformer = transformer

        result = mixin._denoise_step_non_parallel(
            torch.randn(1, 10, 4), torch.tensor([0.5]),
            torch.randn(1, 5, 4), torch.randn(1, 5, 4),
            torch.randn(5, 3), torch.randn(5, 3), torch.randn(10, 3),
            True, 0.0, None,
        )

        assert result.shape == (1, 10, 4)


class TestDenoiseStepParallel:
    @patch("x_diffusers.framework.pipeline.longcat_image.base_longcat_image.get_cfg_group")
    def test_parallel_cond_rank(self, mock_get_cfg_group):
        mock_group = MagicMock()
        cond_pred = torch.randn(1, 10, 4)
        uncond_pred = torch.randn(1, 10, 4)
        mock_group.all_gather.return_value = torch.stack([cond_pred, uncond_pred])
        mock_get_cfg_group.return_value = mock_group

        mixin = _make_mixin(guidance_scale=3.0)
        transformer = MagicMock()
        transformer.return_value = (torch.randn(1, 10, 4),)
        mixin.transformer = transformer

        with patch("torch.cuda.synchronize"):
            result = mixin._denoise_step_parallel(
                torch.randn(1, 10, 4), torch.tensor([0.5]),
                torch.randn(1, 5, 4), torch.randn(5, 3), torch.randn(10, 3),
                0, False, 0.0, None,
            )

        assert result.shape == (1, 10, 4)

    @patch("x_diffusers.framework.pipeline.longcat_image.base_longcat_image.get_cfg_group")
    def test_parallel_uncond_rank(self, mock_get_cfg_group):
        mock_group = MagicMock()
        mock_group.all_gather.return_value = torch.stack([
            torch.randn(1, 10, 4), torch.randn(1, 10, 4)
        ])
        mock_get_cfg_group.return_value = mock_group

        mixin = _make_mixin(guidance_scale=3.0)
        transformer = MagicMock()
        transformer.return_value = (torch.randn(1, 10, 4),)
        mixin.transformer = transformer

        with patch("torch.cuda.synchronize"):
            result = mixin._denoise_step_parallel(
                torch.randn(1, 10, 4), torch.tensor([0.5]),
                torch.randn(1, 5, 4), torch.randn(5, 3), torch.randn(10, 3),
                1, False, 0.0, None,
            )

        assert result.shape == (1, 10, 4)

    @patch("x_diffusers.framework.pipeline.longcat_image.base_longcat_image.get_cfg_group")
    def test_parallel_with_renorm(self, mock_get_cfg_group):
        mock_group = MagicMock()
        mock_group.all_gather.return_value = torch.stack([
            torch.ones(1, 10, 4), torch.ones(1, 10, 4)
        ])
        mock_get_cfg_group.return_value = mock_group

        mixin = _make_mixin(guidance_scale=2.0)
        transformer = MagicMock()
        transformer.return_value = (torch.ones(1, 10, 4),)
        mixin.transformer = transformer

        with patch("torch.cuda.synchronize"):
            result = mixin._denoise_step_parallel(
                torch.randn(1, 10, 4), torch.tensor([0.5]),
                torch.randn(1, 5, 4), torch.randn(5, 3), torch.randn(10, 3),
                0, True, 0.0, None,
            )

        assert result.shape == (1, 10, 4)

    @patch("x_diffusers.framework.pipeline.longcat_image.base_longcat_image.get_cfg_group")
    def test_parallel_with_image_seq_len(self, mock_get_cfg_group):
        mock_group = MagicMock()
        mock_group.all_gather.return_value = torch.stack([
            torch.randn(1, 10, 4), torch.randn(1, 10, 4)
        ])
        mock_get_cfg_group.return_value = mock_group

        mixin = _make_mixin(guidance_scale=2.0)
        transformer = MagicMock()
        transformer.return_value = (torch.randn(1, 20, 4),)
        mixin.transformer = transformer

        with patch("torch.cuda.synchronize"):
            result = mixin._denoise_step_parallel(
                torch.randn(1, 20, 4), torch.tensor([0.5]),
                torch.randn(1, 5, 4), torch.randn(5, 3), torch.randn(20, 3),
                0, False, 0.0, 10,
            )

        assert result.shape == (1, 10, 4)


class TestDecodeLatents:
    def test_output_type_latent(self):
        mixin = _make_mixin()
        latents = torch.randn(1, 4, 64, 64)
        result = mixin._decode_latents(latents, 512, 512, "latent")
        assert torch.equal(result, latents)

    def test_decode_calls_unpack_and_vae(self):
        mixin = _make_mixin()
        mixin._unpack_latents = MagicMock(return_value=torch.randn(1, 4, 64, 64))
        mixin.vae = MagicMock()
        mixin.vae.config.scaling_factor = 0.1
        mixin.vae.config.shift_factor = 0.5
        mixin.vae.dtype = torch.float32
        mixin.vae.decode.return_value = (torch.randn(1, 3, 512, 512),)
        mixin.image_processor = MagicMock()
        mixin.image_processor.postprocess.return_value = "postprocessed"
        mixin.vae_scale_factor = 8

        result = mixin._decode_latents(torch.randn(1, 4, 64, 64), 512, 512, "pil")

        mixin._unpack_latents.assert_called_once()
        mixin.vae.decode.assert_called_once()
        mixin.image_processor.postprocess.assert_called_once()
        assert result == "postprocessed"

    def test_decode_dtype_mismatch(self):
        mixin = _make_mixin()
        mixin._unpack_latents = MagicMock(return_value=torch.randn(1, 4, 64, 64, dtype=torch.float16))
        mixin.vae = MagicMock()
        mixin.vae.config.scaling_factor = 0.1
        mixin.vae.config.shift_factor = 0.5
        mixin.vae.dtype = torch.float32
        mixin.vae.decode.return_value = (torch.randn(1, 3, 512, 512),)
        mixin.image_processor = MagicMock()
        mixin.image_processor.postprocess.return_value = "result"
        mixin.vae_scale_factor = 8

        mixin._decode_latents(torch.randn(1, 4, 64, 64, dtype=torch.float16), 512, 512, "pil")

        call_args = mixin.vae.decode.call_args
        decoded_latents = call_args[0][0]
        assert decoded_latents.dtype == torch.float32


class TestFinalizeOutput:
    def test_return_dict_false(self):
        mixin = _make_mixin()
        mixin.maybe_free_model_hooks = MagicMock()
        result = mixin._finalize_output("image_data", False)
        mixin.maybe_free_model_hooks.assert_called_once()
        assert result == ("image_data",)

    def test_return_dict_true(self):
        mixin = _make_mixin()
        mixin.maybe_free_model_hooks = MagicMock()
        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.LongCatImagePipelineOutput",
        ) as mock_output:
            mock_output.return_value = "output_obj"
            result = mixin._finalize_output("image_data", True)

        assert result == "output_obj"


class TestRunDenoisingLoop:
    def test_non_parallel_path(self):
        mixin = _make_mixin(
            joint_attention_kwargs=None,
            interrupt=False,
            guidance_scale=2.0,
        )
        mixin._denoise_step_non_parallel = MagicMock(return_value=torch.randn(1, 10, 4))
        mixin._denoise_step_parallel = MagicMock(return_value=torch.randn(1, 10, 4))

        scheduler = MagicMock()
        scheduler.order = 1
        scheduler.step.return_value = (torch.randn(1, 10, 4),)
        mixin.scheduler = scheduler

        progress_bar = MagicMock()
        progress_bar.__enter__ = MagicMock(return_value=progress_bar)
        progress_bar.__exit__ = MagicMock(return_value=False)
        mixin.progress_bar = MagicMock(return_value=progress_bar)

        latents = torch.randn(1, 10, 4)
        timesteps = [torch.tensor(0.9), torch.tensor(0.1)]
        prompt_embeds = torch.randn(1, 5, 4)
        text_ids = torch.randn(5, 3)
        latent_image_ids = torch.randn(10, 3)

        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.XLA_AVAILABLE", False
        ):
            result = mixin._run_denoising_loop(
                latents, timesteps, 2, 0,
                prompt_embeds, None, text_ids, None, latent_image_ids,
                cfg_parallel_size=1,
            )

        assert mixin._denoise_step_non_parallel.call_count == 2
        assert mixin._denoise_step_parallel.call_count == 0
        assert mixin._current_timestep is None

    def test_parallel_path(self):
        mixin = _make_mixin(
            joint_attention_kwargs=None,
            interrupt=False,
            guidance_scale=2.0,
        )
        mixin._denoise_step_non_parallel = MagicMock(return_value=torch.randn(1, 10, 4))
        mixin._denoise_step_parallel = MagicMock(return_value=torch.randn(1, 10, 4))

        scheduler = MagicMock()
        scheduler.order = 1
        scheduler.step.return_value = (torch.randn(1, 10, 4),)
        mixin.scheduler = scheduler

        progress_bar = MagicMock()
        progress_bar.__enter__ = MagicMock(return_value=progress_bar)
        progress_bar.__exit__ = MagicMock(return_value=False)
        mixin.progress_bar = MagicMock(return_value=progress_bar)

        with patch("x_diffusers.framework.pipeline.longcat_image.base_longcat_image.dist") as mock_dist, \
             patch(
                "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.XLA_AVAILABLE", False
            ):
            mock_dist.get_rank.return_value = 0
            result = mixin._run_denoising_loop(
                torch.randn(1, 10, 4), [torch.tensor(0.5)], 1, 0,
                torch.randn(1, 5, 4), None, torch.randn(5, 3), None, torch.randn(10, 3),
                cfg_parallel_size=2,
            )

        assert mixin._denoise_step_parallel.call_count == 1

    def test_interrupt_skips_denoise(self):
        mixin = _make_mixin(
            joint_attention_kwargs=None,
            guidance_scale=2.0,
            interrupt=True,
        )
        mixin._denoise_step_non_parallel = MagicMock(return_value=torch.randn(1, 10, 4))

        scheduler = MagicMock()
        scheduler.order = 1
        scheduler.step.return_value = (torch.randn(1, 10, 4),)
        mixin.scheduler = scheduler

        progress_bar = MagicMock()
        progress_bar.__enter__ = MagicMock(return_value=progress_bar)
        progress_bar.__exit__ = MagicMock(return_value=False)
        mixin.progress_bar = MagicMock(return_value=progress_bar)

        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.XLA_AVAILABLE", False
        ):
            mixin._run_denoising_loop(
                torch.randn(1, 10, 4), [torch.tensor(0.5)], 1, 0,
                torch.randn(1, 5, 4), None, torch.randn(5, 3), None, torch.randn(10, 3),
                cfg_parallel_size=1,
            )

        assert mixin._denoise_step_non_parallel.call_count == 0

    def test_with_image_latents(self):
        mixin = _make_mixin(
            joint_attention_kwargs=None,
            interrupt=False,
            guidance_scale=2.0,
        )
        mixin._denoise_step_non_parallel = MagicMock(return_value=torch.randn(1, 10, 4))

        scheduler = MagicMock()
        scheduler.order = 1
        scheduler.step.return_value = (torch.randn(1, 10, 4),)
        mixin.scheduler = scheduler

        progress_bar = MagicMock()
        progress_bar.__enter__ = MagicMock(return_value=progress_bar)
        progress_bar.__exit__ = MagicMock(return_value=False)
        mixin.progress_bar = MagicMock(return_value=progress_bar)

        with patch(
            "x_diffusers.framework.pipeline.longcat_image.base_longcat_image.XLA_AVAILABLE", False
        ):
            mixin._run_denoising_loop(
                torch.randn(1, 10, 4), [torch.tensor(0.5)], 1, 0,
                torch.randn(1, 5, 4), None, torch.randn(5, 3), None, torch.randn(10, 3),
                cfg_parallel_size=1, image_latents=torch.randn(1, 5, 4),
            )

        call_args = mixin._denoise_step_non_parallel.call_args
        latent_input = call_args[0][0]
        assert latent_input.shape[1] == 15
