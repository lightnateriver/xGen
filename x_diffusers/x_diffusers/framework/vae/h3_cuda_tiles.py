# SPDX-License-Identifier: Apache-2.0
"""Distribute existing H3 spatial decode tiles across the CUDA SP group."""

import torch
import torch.distributed as dist


def decode_clip_tiles(vae, z):
    if not vae._h3_cuda_vae_tiles or not vae.use_tiling:
        return vae._h3_original_decode_clip(z)

    ratio = vae.spatial_compression_ratio
    y, heights, y_overlap = vae._split_tiles(
        z.shape[-2] * ratio, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height,
    )
    x, widths, x_overlap = vae._split_tiles(
        z.shape[-1] * ratio, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width,
    )
    group = vae._h3_cuda_vae_parallel.sp_group
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    num_tiles = len(y) * len(x)
    tiles = []
    for start in range(0, num_tiles, size):
        tile_id = min(start + rank, num_tiles - 1)
        i, j = divmod(tile_id, len(x))
        top, left = y[i] // ratio, x[j] // ratio
        tile = z[..., top:top + heights[i] // ratio, left:left + widths[j] // ratio]
        local = vae.decoder(vae.post_quant_conv(tile)).contiguous()
        gathered = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(gathered, local, group=group)
        tiles.extend(gathered[:num_tiles - start])

    rows = [tiles[start:start + len(x)] for start in range(0, num_tiles, len(x))]
    return vae._stitch_tiles(rows, y_overlap, x_overlap)
