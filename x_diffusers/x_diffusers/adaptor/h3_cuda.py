"""Install optional CUDA optimizations on the existing H3 pipeline."""

import logging
import os
from types import MethodType

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


_DIT_FUSION_FLAGS = (
    "XGEN_H3_FUSED_INDEXED_AFFINE",
    "XGEN_H3_FUSED_PARTIAL_ROPE",
    "XGEN_H3_NVFP4_FUSED_AMAX",
    "XGEN_H3_FUSED_SWIGLU",
    "XGEN_H3_FUSED_QK_RMS",
    "XGEN_H3_FUSED_FF_AMAX",
)


def get_dit_fusions(packed_resident, packed_compile):
    enabled = frozenset(flag for flag in _DIT_FUSION_FLAGS if os.environ.get(flag) == "1")
    if enabled and (not packed_resident or packed_compile):
        raise ValueError(f"{', '.join(sorted(enabled))} requires resident CUDA NVFP4 without block compile")
    return enabled


def install_dit_fusions(transformer, enabled):
    if "XGEN_H3_FUSED_OUTPUT_ADALN" in enabled:
        transformer.norm_out._h3_fused_indexed_affine = True
        logger.info("Experimental CUDA H3 output AdaLN indexed affine fusion enabled")
    if "XGEN_H3_FUSED_INDEXED_AFFINE" in enabled:
        from x_base.operator.h3_indexed_affine import fused_h3_block_forward

        for block in transformer.transformer_blocks:
            block._h3_eager_forward = block.forward
            block._h3_fused_indexed_affine = True
            block.forward = MethodType(fused_h3_block_forward, block)
        logger.info("Experimental CUDA H3 indexed affine fusion enabled")
    if "XGEN_H3_FUSED_PARTIAL_ROPE" in enabled:
        for block in transformer.transformer_blocks:
            block.attn._h3_fused_partial_rope = True
        logger.info("Experimental CUDA H3 partial RoPE fusion enabled")
    if "XGEN_H3_FUSED_QK_RMS" in enabled:
        for block in transformer.transformer_blocks:
            block.attn._h3_fused_qk_rms = True
        logger.info("Experimental CUDA H3 Q/K RMS fusion enabled")
    if "XGEN_H3_NVFP4_FUSED_AMAX" in enabled:
        from x_base.operator.h3_nvfp4 import H3NVFP4Linear

        for module in transformer.modules():
            if isinstance(module, H3NVFP4Linear):
                module.fused_amax = True
        logger.info("Experimental CUDA H3 NVFP4 fused amax enabled")
    if "XGEN_H3_FUSED_SWIGLU" in enabled:
        from x_base.operator.h3_nvfp4 import H3GateFirstSwiGLU

        for module in transformer.modules():
            if isinstance(module, H3GateFirstSwiGLU):
                module.fused_swiglu = True
        logger.info("Experimental CUDA H3 SwiGLU fusion enabled")
    if "XGEN_H3_FUSED_FF_AMAX" in enabled:
        from x_base.operator.h3_nvfp4 import H3GateFirstSwiGLU, H3NVFP4Linear, fused_h3_ff_forward

        for block in transformer.transformer_blocks:
            ff = block.ff
            if (
                len(ff.net) != 3 or not isinstance(ff.net[0], H3GateFirstSwiGLU)
                or not isinstance(ff.net[1], torch.nn.Dropout)
                or not isinstance(ff.net[2], H3NVFP4Linear)
            ):
                raise ValueError("H3 fused FFN amax requires gate-first NVFP4 FeedForward")
            ff._h3_eager_ff_forward = ff.forward
            ff._h3_fused_ff_amax = True
            ff.forward = MethodType(fused_h3_ff_forward, ff)
        logger.info("Experimental CUDA H3 SwiGLU with activation amax enabled")


def configure_parallelism(pipe, args):
    cuda_vae_tiles = os.environ.get("XGEN_H3_CUDA_VAE_TILES") == "1"
    if getattr(args, "h3_cuda_precision", "none") in ("fp8-bf16", "fp8-sage"):
        cuda_vae_tiles = True
    if cuda_vae_tiles and not getattr(args, "h3_cuda_vae_sp", False):
        raise ValueError("XGEN_H3_CUDA_VAE_TILES requires --h3_cuda_vae_sp")
    if getattr(args, "h3_cuda_vae_sp", False):
        manager = pipe.transformer.parallel_manager
        heads = pipe.vae.decoder.transformer_blocks[0].attn.heads
        if heads % manager.sp_size:
            raise ValueError("CUDA H3 VAE attention heads must divide evenly across SP ranks")
        pipe.vae._h3_cuda_vae_parallel = manager
        if cuda_vae_tiles:
            from ..framework.vae.h3_cuda_tiles import decode_clip_tiles

            pipe.vae._h3_original_decode_clip = pipe.vae._decode_clip
            pipe.vae._decode_clip = MethodType(decode_clip_tiles, pipe.vae)
            pipe.vae._h3_cuda_vae_tiles = True
            logger.info("Experimental CUDA H3 spatial VAE tile parallelism enabled")
        logger.info("Experimental CUDA H3 VAE SP enabled, sp_size=%s", manager.sp_size)

    fp8_phaa = os.environ.get("XGEN_PHAA_FP8") == "1"
    async_exchange = os.environ.get("XGEN_PHAA_ASYNC_EXCHANGE") == "1"
    direct_return = os.environ.get("XGEN_PHAA_DIRECT_RETURN") == "1"
    fp8_row_tiling = os.environ.get("XGEN_PHAA_FP8_ROW_TILING") == "1"
    fused_return_pack = os.environ.get("XGEN_PHAA_FUSED_RETURN_PACK") == "1"
    direct_gather = os.environ.get("XGEN_PHAA_DIRECT_GATHER") == "1"
    if direct_return and not async_exchange:
        raise ValueError("XGEN_PHAA_DIRECT_RETURN requires XGEN_PHAA_ASYNC_EXCHANGE=1")
    if fused_return_pack and not direct_return:
        raise ValueError("XGEN_PHAA_FUSED_RETURN_PACK requires XGEN_PHAA_DIRECT_RETURN=1")
    if fp8_row_tiling and not async_exchange:
        raise ValueError("XGEN_PHAA_FP8_ROW_TILING requires XGEN_PHAA_ASYNC_EXCHANGE=1")
    if direct_gather and not async_exchange:
        raise ValueError("XGEN_PHAA_DIRECT_GATHER requires XGEN_PHAA_ASYNC_EXCHANGE=1")
    if async_exchange and not fp8_phaa:
        raise ValueError("XGEN_PHAA_ASYNC_EXCHANGE requires CUDA and XGEN_PHAA_FP8=1")
    if fp8_phaa:
        local_heads = pipe.transformer.config.num_attention_heads // args.sp
        if args.sp < 2 or not 1 <= args.phaa_num <= local_heads:
            raise ValueError("XGEN_PHAA_FP8=1 requires H3, SP > 1, and a valid --phaa_num")
        chunk_order = os.environ.get("XGEN_PHAA_CHUNK_ORDER", "natural")
        if chunk_order not in ("natural", "tail-first"):
            raise ValueError("XGEN_PHAA_CHUNK_ORDER must be natural or tail-first")
        if chunk_order == "tail-first" and args.phaa_num != 4:
            raise ValueError("PHAA tail-first requires --phaa_num 4")
        gather_priority = os.environ.get("XGEN_PHAA_GATHER_PRIORITY") == "1"
        dual_comm = os.environ.get("XGEN_PHAA_DUAL_COMM") == "1"
        device_join = os.environ.get("XGEN_PHAA_DEVICE_JOIN") == "1"
        if async_exchange and not device_join:
            raise ValueError("XGEN_PHAA_ASYNC_EXCHANGE requires XGEN_PHAA_DEVICE_JOIN=1")
        if async_exchange and not callable(getattr(dist.Work, "block_current_stream", None)):
            raise RuntimeError("XGEN_PHAA_ASYNC_EXCHANGE requires distributed Work.block_current_stream")
        if device_join and dual_comm:
            raise ValueError("XGEN_PHAA_DEVICE_JOIN requires XGEN_PHAA_DUAL_COMM to be disabled")
        scatter_group = scatter_stream = None
        if dual_comm:
            from x_base.phaa.utils import phaa_cuda_scatter_group

            sp_group = pipe.transformer.parallel_manager.sp_group
            if dist.get_process_group_ranks(sp_group) != list(range(dist.get_world_size())):
                raise ValueError("PHAA dual communication requires a world-spanning SP group")
            scatter_group = phaa_cuda_scatter_group(dist.group.WORLD)
            scatter_stream = torch.cuda.Stream()
        async_options = None
        if async_exchange:
            from x_base.phaa.fp8_async_transport import AsyncFP8Options

            async_options = AsyncFP8Options(
                direct_return=direct_return, row_tiling=fp8_row_tiling,
                fused_return_pack=fused_return_pack, direct_gather=direct_gather,
            )
        for block in pipe.transformer.transformer_blocks:
            block.attn._h3_phaa_fp8 = True
            block.attn._h3_phaa_gather_priority = gather_priority
            block.attn._h3_phaa_tail_first = chunk_order == "tail-first"
            block.attn._h3_phaa_device_join = device_join
            block.attn._h3_phaa_async_options = async_options
            block.attn._h3_phaa_scatter_group = scatter_group
            block.attn._h3_phaa_scatter_stream = scatter_stream
        logger.info("Experimental H3 PHAA E4M3 transport enabled, chunks=%s, order=%s, gather_priority=%s, dual_comm=%s, device_join=%s, async_exchange=%s, NCCL_MAX_CTAS=%s",
                    args.phaa_num, chunk_order, gather_priority, dual_comm, device_join,
                    async_exchange, os.environ.get("NCCL_MAX_CTAS"))
        if direct_return:
            logger.info("Experimental H3 PHAA direct return enabled")
        if fp8_row_tiling:
            logger.info("Experimental H3 PHAA FP8 row tiling enabled (4 rows/program)")
        if fused_return_pack:
            logger.info("Experimental H3 PHAA fused return packing enabled")
        if direct_gather:
            logger.info("Experimental H3 PHAA direct gather enabled")
