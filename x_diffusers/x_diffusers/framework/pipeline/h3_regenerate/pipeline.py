"""Experimental latent SR backend for XGEN's existing managers (no request loop)."""
import functools
import logging
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import torch.distributed as dist

from x_base.device import device_type
from x_base.infinite_vram.immutable_weights import CpuWeights
from x_base.utils.stage_metrics import Telemetry
from .assets import prepared_cache, validate_base_schedule
from .vendor import geometry, upscaler, vae_fast
from .vendor.sigmas import SCHEDULES

logger = logging.getLogger(__name__)


def prepare_vae_precision(vae):
    """Override from_pretrained's keep-in-FP32 Conv policy before BF16 tile inputs."""
    vae.to(dtype=torch.bfloat16)
    for module in vae.decoder.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.RMSNorm)):
            module.float()
    return vae


class H3RefinerPipeline:
    """Borrow the manager's Base; own only Refiner components and stage residency."""
    def __init__(self, base, args):
        from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel
        from diffusers.models.attention_dispatch import AttentionBackendName
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
        from x_base.adaln_cache import install_adaln_cache, release_adaln_weights, validate_stripped_checkpoint
        from x_base.phaa.utils import inject_phaa_for_minimax_h3
        self.base, self.args = base, args
        self.backend = device_type()
        self.accelerator = torch.get_device_module(self.backend)
        self.rank = dist.get_rank()
        self.failed = False
        self.completed = 0
        self.startup_seconds = None
        self.metrics = Telemetry()
        turbo8 = getattr(args, "h3_turbo8", False)
        self.base_steps = validate_base_schedule(args.num_inference_steps, turbo8=turbo8)
        if args.adaln_cache_file:
            from safetensors import safe_open
            with safe_open(args.adaln_cache_file, framework='pt') as cache:
                validate_base_schedule(args.num_inference_steps, cache.metadata() or {}, turbo8=turbo8)
        weights, prepared = Path(args.h3_refiner_weights), Path(args.h3_refiner_prepared)
        self.base.audio_vae.float()
        self.base_weights = CpuWeights(base.transformer, 'h3.weights', self.metrics)
        self.base_vae = CpuWeights(base.vae, 'h3.reference_vae', self.metrics)
        self.base_audio = CpuWeights(base.audio_vae, 'h3.audio_vae', self.metrics)
        self.calls = dict(h3=0, refiner=0)
        self.original_forwards = []
        self._instrument(base.transformer, 'h3')
        self.refiner = MiniMaxH3Transformer3DModel.from_pretrained(
            str(prepared/'transformer'), torch_dtype=torch.bfloat16, local_files_only=True).eval()
        validate_stripped_checkpoint(self.refiner, str(prepared/'transformer'))
        install_adaln_cache(SimpleNamespace(transformer=self.refiner), str(prepared_cache(prepared)), 'none')
        release_adaln_weights(self.refiner)
        self.refiner.enable_parallel(dp_size=1, sp_size=8)
        inject_phaa_for_minimax_h3(self.refiner, 2)
        self.ref_weights = CpuWeights(self.refiner, 'refiner.weights', self.metrics)
        self.ref_weights.stream_blocks(args.h3_refiner_loading)
        self._instrument(self.refiner, 'refiner')
        self.text = MiniMaxH3TextEncoderStep().init_pipeline(str(weights/'base_model'))
        self.text.update_components(text_encoder=base.text_encoder, tokenizer=base.tokenizer, processor=base.processor)
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(str(weights/'base_model'), subfolder='vae',
            torch_dtype=torch.bfloat16, local_files_only=True).eval()
        prepare_vae_precision(self.vae)
        if self.backend == 'npu':
            self.vae.set_attention_backend(AttentionBackendName._NATIVE_NPU)
        self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                              tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
        vae_fast.install(self.vae, batch=1, distributed=True)
        for field in ('latents_mean', 'latents_std'):
            if list(getattr(base.vae.config, field)) != list(getattr(self.vae.config, field)):
                raise ValueError('H3/Refiner VAE normalization differs: '+field)
        self.vae_weights = CpuWeights(self.vae, 'refiner.vae_weights', self.metrics)
        self.node = upscaler.load(weights/'latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
        if self.backend == 'npu':
            resolve = self.node._resolve_device
            self.node._resolve_device = lambda backend: torch.device('npu') if backend == 'cuda' else resolve(backend)
        self.node.load_model('upscaler', torch.device('cpu'), 'bf16')
        self.upscale_model = next(iter(self.node.MODEL_CACHE.values()))
        self.upscale_weights = CpuWeights(self.upscale_model, 'upscaler.weights', self.metrics)
        self.node.load_model = lambda *unused: self.upscale_model
        self.base_weights.onload()
        self.identity = self.identities()
        logger.info('H3_REFINER_ASSETS_READY cpu_master_bytes=%s refiner_dtype=bf16 refiner_steps=3',
                    sum(h.nbytes for h in self.holders()))

    def holders(self):
        return (self.base_weights, self.base_vae, self.base_audio, self.ref_weights,
                self.vae_weights, self.upscale_weights)

    def identities(self):
        return {h.name: h.fingerprint() for h in self.holders()}

    def _instrument(self, model, name):
        forward = model.forward
        self.original_forwards.append((model, forward))
        @functools.wraps(forward)
        def measured(*args, **kwargs):
            self.calls[name] += 1
            with self.metrics.span(name+'.dit', sync=False):
                return forward(*args, **kwargs)
        model.forward = measured

    def close(self):
        """Release only this backend's holders/hooks; changing backend requires restart."""
        for holder in self.holders():
            holder.close()
        for model, forward in self.original_forwards:
            model.forward = forward
        self.failed = True

    @torch.inference_mode()
    def _draft(self, args, spec):
        from diffusers.utils import load_image
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
        from x_diffusers.adaptor.h3_refiner_config import native_padding
        self.base_weights.onload()
        self.base_vae.onload()
        params = dict(prompt=args.prompt, width=1344, height=768, num_frames=spec['generation_frames'],
                      num_inference_steps=args.num_inference_steps,
                      generator=torch.Generator().manual_seed(args.seed), output=['latents', 'audio_latents'])
        if args.task_type == 'ref2va':
            params['references'] = [MiniMaxH3ImageReference.from_file(p) for p in args.reference_paths]
        elif args.task_type == 'fl2va':
            params['image'] = load_image(args.keyframe_image)
        original = self.base._blocks
        blocks = self.base.blocks
        blocks.sub_blocks.pop('decode')
        self.base._blocks = blocks
        try:
            with native_padding(self.base, spec), self.metrics.span('h3.pipeline'):
                result = self.base(**params)
        finally:
            self.base._blocks = original
        video, audio = result['latents'].float().cpu(), result['audio_latents'].float().cpu()
        if tuple(video.shape) != (1,24,spec['latent_frames'],48,84) or not torch.isfinite(video).all():
            raise ValueError('Unexpected or nonfinite normalized H3 video latent')
        if tuple(audio.shape) != (2,32,spec['audio_frames']) or not torch.isfinite(audio).all():
            raise ValueError('Unexpected or nonfinite H3 audio latent')
        self.base_vae.offload()
        self.base_weights.offload()
        return video[0], audio

    @torch.inference_mode()
    def _conditions(self, args, spec, native):
        self.base_audio.onload()
        av = self.base.audio_vae
        with self.metrics.span('refiner.silence_encode'):
            silence = av.encode(torch.zeros(2,1,round(spec['generation_frames']/24*32000),device=self.backend),
                                return_dict=False)[0].mode()[...,:spec['audio_frames']]
            if silence.shape[-1] != spec['audio_frames']:
                raise ValueError('Short silent conditioning')
            mean = silence.new_tensor(av.config.latents_mean).view(1,-1,1)
            std = silence.new_tensor(av.config.latents_std).view(1,-1,1)
            audio = ((silence-mean)/std).bfloat16().cpu()
        self.base_audio.offload()
        with self.metrics.span('refiner.text_encode'):
            text = self.text(prompt=args.prompt, output=['prompt_embeds','text_token_tags'])
            text = dict(prompt_embeds=text['prompt_embeds'][0].cpu(),
                        text_token_tags=text['text_token_tags'].flatten().cpu())
        self.upscale_weights.onload()
        with self.metrics.span('upscaler.compute'):
            low = upscaler.upscale(self.node, native, (24,spec['latent_frames'],90,160)).cpu()
        self.upscale_weights.offload()
        return dict(upscaled=low, audio=audio, **text)

    @torch.inference_mode()
    def _refine(self, args, spec, conditions):
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        self.ref_weights.onload(self.ref_weights.nonblocks, 'refiner.initial_h2d_wait')
        audio = geometry.pack_audio_latents(conditions['audio'][None]).to(self.backend)
        shape = geometry.MiniMaxH3Geometry(spec['latent_frames'],90,160,conditions['audio'].shape[-1])
        layout = geometry.build_t2va_layout(conditions['text_token_tags'],shape).to(self.backend)
        prompt = conditions['prompt_embeds'][None].to(self.backend)
        low = patchify_video_latents(conditions['upscaled'][None].to(self.backend),shape.patch_size).reshape(
            1,shape.num_video_rows,shape.video_row_dim).float()
        state = (1-SCHEDULES[3][0])*low+SCHEDULES[3][0]*torch.randn(low.shape,device=self.backend,
            generator=torch.Generator(device=self.backend).manual_seed(args.seed),dtype=torch.float32)
        for index,(sigma,following) in enumerate(zip(SCHEDULES[3][:-1],SCHEDULES[3][1:])):
            times,indices = geometry.build_row_timestep_plan(layout,video_sigma=sigma,audio_sigma=0.,keyframe_noise_aug=1.)
            with self.metrics.span('refiner.step',step=index):
                if self.args.h3_refiner_loading == 'prefetch':
                    self.ref_weights.prefetch.begin_step()
                velocity,_ = self.refiner(hidden_states=state.bfloat16(),audio_hidden_states=audio,
                    encoder_hidden_states=prompt,timestep=times,timestep_indices=indices,
                    token_tags=layout.token_tags,position_ids=layout.position_ids,
                    video_indices=layout.video_indices,audio_indices=layout.audio_indices,
                    text_indices=layout.text_indices,attention_kwargs={},return_dict=False)
                state = state+(sigma-following)*velocity.float()
                if self.args.h3_refiner_loading == 'prefetch':
                    self.ref_weights.prefetch.end_step()
        final = geometry.unpatchify_video_rows(state,shape)[0].cpu()
        if not torch.isfinite(final).all():
            raise ValueError('Nonfinite Refiner output')
        if self.args.h3_refiner_loading == 'prefetch':
            self.ref_weights.prefetch.release()
        self.ref_weights.offload()
        return final

    @torch.inference_mode()
    def _decode(self, args, spec, final, audio):
        from ...vae import minimax_h3 as native_vae
        from ...vae.h3_refiner_output import RefinerOutput
        waveform = None
        if getattr(args, 'enable_audio', True):
            self.base_audio.onload()
            with self.metrics.span('audio.decode'):
                latent = audio.to(self.backend)
                av = self.base.audio_vae
                mean = latent.new_tensor(av.config.latents_mean).view(1,-1,1)
                std = latent.new_tensor(av.config.latents_std).view(1,-1,1)
                waveform = av.decode(latent*std+mean,return_dict=False)[0].float().permute(1,0,2).cpu()
                if not torch.isfinite(waveform).all():
                    raise ValueError('Nonfinite decoded H3 audio')
            self.base_audio.offload()
        sink = RefinerOutput(args.save_path, spec['output_frames'], self.metrics, waveform) if self.rank == 0 else None
        self.vae_weights.onload()
        try:
            with self.metrics.span('vae.decode'):
                latent = final[None].to(self.backend).float()
                mean = latent.new_tensor(self.vae.config.latents_mean).view(1,24,1,1,1)
                std = latent.new_tensor(self.vae.config.latents_std).view(1,24,1,1,1)
                pixels = native_vae._decode_chunks_streaming(self.vae,(latent*std+mean).bfloat16(),None,sink,
                                                            frame_callback=sink.enqueue if sink else None)
                if pixels.shape[2] < spec['output_frames']:
                    raise ValueError('Decoded video is shorter than requested')
                del pixels, latent
            if sink:
                sink.finish()
        finally:
            if sink:
                sink.close()
            self.vae_weights.offload()

    def __call__(self, args):
        from x_base import rope_manager
        from x_base.phaa import get_phaa_split_num, set_phaa_split_num
        from x_base.utils.infer_info import infer_info
        from x_diffusers.adaptor.h3_refiner_config import profile
        from .state import atomic_json
        if self.failed:
            raise RuntimeError('H3 Refiner worker is failed; restart required')
        spec = profile(args.h3_refiner_seconds)
        old_rope, old_phaa = rope_manager.is_rope_fused, get_phaa_split_num()
        old_lightning = infer_info.vae_lightning
        started = time.perf_counter()
        self.metrics.reset()
        self.calls = dict(h3=0,refiner=0)
        self.accelerator.reset_peak_memory_stats()
        try:
            infer_info.update_info(args)
            infer_info.vae_lightning = 'none'
            torch.manual_seed(args.seed)
            self.accelerator.manual_seed(args.seed)
            set_phaa_split_num(args.phaa_num)
            native, audio = self._draft(args,spec)
            rope_manager.is_rope_fused = False
            conditions = self._conditions(args,spec,native)
            set_phaa_split_num(2)
            final = self._refine(args,spec,conditions)
            self._decode(args,spec,final,audio)
            self.base_weights.onload(label='h3.restore_for_next_request_h2d_wait')
            if self.calls != dict(h3=self.base_steps,refiner=3):
                raise RuntimeError('Unexpected actual DiT counts: '+str(self.calls))
            if self.identities() != self.identity:
                raise RuntimeError('Resident model/CPU master identity changed')
            record = dict(experimental=True, startup_seconds=self.startup_seconds,
                request_seconds=time.perf_counter()-started, actual_dit=dict(self.calls),
                profile=spec, phases=self.metrics.resolve(), completed_requests=self.completed+1,
                peak_allocated_bytes=self.accelerator.max_memory_allocated(),peak_reserved_bytes=self.accelerator.max_memory_reserved(),
                allocated_after=self.accelerator.memory_allocated(),reserved_after=self.accelerator.memory_reserved(),
                model_and_cpu_master_reused=True)
            atomic_json(Path(args.save_path).with_suffix(f'.metrics.rank{self.rank}.json'),record)
            dist.barrier()
            self.completed += 1
            logger.info('H3_REFINER_COMPLETE rank=%s seconds=%.3f base_dit=%s refiner_dit=3',
                        self.rank,record['request_seconds'],self.calls['h3'])
            return dict(videos=None,audio=None,sampling_rate=32000,saved_path=str(Path(args.save_path).resolve()),
                        h3_refiner_metrics=record)
        except Exception:
            self.failed = True
            logger.exception('H3_REFINER_FAILED rank=%s retry=false restart_required=true',self.rank)
            raise
        finally:
            self.metrics.active = False
            rope_manager.is_rope_fused = old_rope
            set_phaa_split_num(old_phaa)
            infer_info.vae_lightning = old_lightning
