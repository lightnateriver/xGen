from __future__ import annotations

import math
from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from .save_video_stream import SaveVideoStream
from .vae_mgr import blend_v, blend_h


def blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = (a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) +
                            b[:, :, x, :, :] * (x / blend_extent))
    return b


def merge_spatial_tiles_cog(spatial_rows, blend_extent, row_limit):
    """合并空间分块（与原实现保持一致）"""
    result_rows = []
    for i, row in enumerate(spatial_rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(spatial_rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def parallel_data_generator(gathered_results, gathered_dim_metadata):
    global_idx = 0
    for i, per_rank_metadata in enumerate(gathered_dim_metadata):
        _start_shape = 0
        for shape in per_rank_metadata:
            mul_shape = math.prod(shape)
            yield (gathered_results[i, _start_shape:_start_shape + mul_shape].reshape(shape), global_idx)
            _start_shape += mul_shape
            global_idx += 1


def parallel_spatial_tiled_decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        use_conv_cache: bool = True,
) -> Union[DecoderOutput, torch.FloatTensor]:
    """
    仅基于空间维度分块的并行解码，时间维度保持完整
    """
    from diffusers.models.autoencoders.vae import DecoderOutput

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    B, C, T, H, W = z.shape

    # 仅计算空间重叠参数
    s_overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
    s_blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
    s_row_limit = self.tile_sample_min_size - s_blend_extent

    # 计算空间分块数量
    num_h_tiles = (H + s_overlap_size - 1) // s_overlap_size
    num_w_tiles = (W + s_overlap_size - 1) // s_overlap_size
    total_spatial_tiles = num_h_tiles * num_w_tiles

    # 分配各GPU处理的分块范围
    tiles_per_rank = total_spatial_tiles // world_size
    extra_rank = total_spatial_tiles % world_size
    if rank < extra_rank:
        start_tile_idx = rank * (tiles_per_rank + 1)
        end_tile_idx = start_tile_idx + tiles_per_rank + 1
    else:
        start_tile_idx = extra_rank * (tiles_per_rank + 1) + (rank - extra_rank) * tiles_per_rank
        end_tile_idx = start_tile_idx + tiles_per_rank

    local_results = []
    local_dim_metadata = []
    # 处理分配的分块
    for global_idx in range(start_tile_idx, end_tile_idx):
        conv_cache = None
        h_idx = global_idx // num_w_tiles
        w_idx = global_idx % num_w_tiles

        # 空间位置计算
        h_start = h_idx * s_overlap_size
        w_start = w_idx * s_overlap_size

        # 提取完整时间维度的分块
        tile = z[:, :, :,  # 时间维度全保留
               h_start:h_start + self.tile_latent_min_size,
               w_start:w_start + self.tile_latent_min_size]

        # 处理分块
        if self.post_quant_conv is not None:
            tile = self.post_quant_conv(tile)
        if use_conv_cache:
            decoded, conv_cache = self.decoder(tile, conv_cache)
        else:
            decoded = self.decoder(tile)

        # 保存元数据和结果
        local_results.append(decoded.reshape(-1))
        local_dim_metadata.append(decoded.shape)

    # 结果收集与同步逻辑（与原函数保持一致）
    results = torch.cat(local_results, dim=0).contiguous()

    local_size = torch.tensor([results.size(0)], device=results.device, dtype=torch.int64)
    all_sizes = [torch.zeros(1, device=results.device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(size.item() for size in all_sizes)

    padded_results = torch.zeros(max_size, device=results.device)
    padded_results[:results.size(0)] = results

    gathered_dim_metadata = [None] * world_size
    gathered_results = torch.zeros_like(padded_results).repeat(world_size, 1).contiguous()
    dist.all_gather_into_tensor(gathered_results, padded_results)
    torch.cuda.set_device(rank)
    dist.all_gather_object(gathered_dim_metadata, local_dim_metadata)

    # 重组分块数据
    data = [[None for _ in range(num_w_tiles)] for _ in range(num_h_tiles)]
    for current_data, global_idx in parallel_data_generator(gathered_results, gathered_dim_metadata):
        h_idx = global_idx // num_w_tiles
        w_idx = global_idx % num_w_tiles
        data[h_idx][w_idx] = current_data

    # 合并空间分块
    merged = merge_spatial_tiles_cog(data, s_blend_extent, s_row_limit)

    if not return_dict:
        return (merged,)
    return DecoderOutput(sample=merged)


# vae3
def temporal_tiled_decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        sample_shape=None,
        save_path: str = None,
        use_conv_cache: bool = True,
) -> Union[DecoderOutput, torch.FloatTensor]:
    from diffusers.models.autoencoders.vae import DecoderOutput

    B, C, T, H, W = z.shape
    overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
    blend_extent_default = int(self.tile_sample_min_tsize * self.tile_overlap_factor)

    if dist.get_rank() == 0:
        save_video_stream = SaveVideoStream()

    last_decoded = None
    total_out = 0
    out_chunks = []

    for i in range(0, T, overlap_size):
        latent_end = min(i + self.tile_latent_min_tsize + 1, T)
        tile = z[:, :, i:latent_end, :, :]
        decoded = self.parallel_spatial_tiled_decode(
            tile, return_dict=True, use_conv_cache=use_conv_cache).sample

        actual_len = decoded.shape[2]

        curr_blend = min(blend_extent_default, actual_len - 1)
        is_first = (i == 0)
        is_last = (latent_end == T)

        if not is_first:
            decoded = decoded[:, :, 1:, :, :]
            actual_len -= 1
            decoded = blend_t(last_decoded, decoded, curr_blend)

        if is_last:
            video_chunk = decoded
        else:
            keep = actual_len - curr_blend
            video_chunk = decoded[:, :, :keep, :, :]

        out_len = video_chunk.shape[2]
        total_out += out_len
        if dist.get_rank() == 0:
            save_video_stream.save(video_chunk)

        out_chunks.append(video_chunk)
        last_decoded = decoded[:, :, -curr_blend:, :, :]

    if dist.get_rank() == 0:
        save_video_stream.close()

        # NOTE: This function works correctly for HunyuanVideo,
        #       where the total frame count always satisfies frames = 4 * k + 1.
        # TODO: Verify that the same logic holds for CogVideo,
        #       whose temporal-upsampling scheme may differ.
        exp_total = sample_shape[0] if sample_shape is not None else None
        if exp_total is not None and total_out != exp_total:
            print(f"[decode] FINISHED: produced {total_out} frames (expected {exp_total})")

    dec = torch.cat(out_chunks, dim=2)

    if not return_dict:
        return (dec,)
    return DecoderOutput(sample=dec)


def tiled_decode_parallel(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        use_conv_cache: bool = True,
) -> Union[DecoderOutput, torch.FloatTensor]:
    """
    Parallel version of tiled_decode that distributes both temporal and spatial computation across GPUs
    """
    from diffusers.models.autoencoders.vae import DecoderOutput

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    B, C, T, H, W = z.shape
    # Calculate parameters
    t_overlap_size = int(self.tile_latent_min_tsize *
                         (1 - self.tile_overlap_factor))
    t_blend_extent = int(self.tile_sample_min_tsize *
                         self.tile_overlap_factor)
    t_limit = self.tile_sample_min_tsize - t_blend_extent
    s_overlap_size = int(self.tile_latent_min_size *
                         (1 - self.tile_overlap_factor))
    s_blend_extent = int(self.tile_sample_min_size *
                         self.tile_overlap_factor)
    s_row_limit = self.tile_sample_min_size - s_blend_extent
    # Calculate tile dimensions
    num_t_tiles = (T + t_overlap_size - 1) // t_overlap_size
    num_h_tiles = (H + s_overlap_size - 1) // s_overlap_size
    num_w_tiles = (W + s_overlap_size - 1) // s_overlap_size
    total_spatial_tiles = num_h_tiles * num_w_tiles
    total_tiles = num_t_tiles * total_spatial_tiles
    # Calculate tiles per rank and padding
    tiles_per_rank = (total_tiles + world_size - 1) // world_size
    start_tile_idx = rank * tiles_per_rank
    end_tile_idx = min((rank + 1) * tiles_per_rank, total_tiles)
    local_results = []
    local_dim_metadata = []
    # Process assigned tiles
    for local_idx, global_idx in enumerate(range(start_tile_idx, end_tile_idx)):
        conv_cache = None
        # Convert flat index to 3D indices
        t_idx = global_idx // total_spatial_tiles
        spatial_idx = global_idx % total_spatial_tiles
        h_idx = spatial_idx // num_w_tiles
        w_idx = spatial_idx % num_w_tiles
        # Calculate positions
        t_start = t_idx * t_overlap_size
        h_start = h_idx * s_overlap_size
        w_start = w_idx * s_overlap_size
        # Extract and process tile
        tile = z[:, :, t_start:t_start + self.tile_latent_min_tsize + 1,
               h_start:h_start + self.tile_latent_min_size,
               w_start:w_start + self.tile_latent_min_size]
        # Process tile
        if self.post_quant_conv is not None:
            tile = self.post_quant_conv(tile)
        if use_conv_cache:
            decoded, conv_cache = self.decoder(tile, conv_cache)
        else:
            decoded = self.decoder(tile)

        if t_start > 0:
            decoded = decoded[:, :, 1:, :, :]
        # Store metadata
        shape = decoded.shape
        # Store decoded data (flattened)
        decoded_flat = decoded.reshape(-1)
        local_results.append(decoded_flat)
        local_dim_metadata.append(shape)

    results = torch.cat(local_results, dim=0).contiguous()
    # first gather size to pad the results
    local_size = torch.tensor([results.size(0)],
                              device=results.device,
                              dtype=torch.int64)
    all_sizes = [
        torch.zeros(1, device=results.device, dtype=torch.int64)
        for _ in range(world_size)
    ]
    dist.all_gather(all_sizes, local_size)
    max_size = max(size.item() for size in all_sizes)
    padded_results = torch.zeros(max_size, device=results.device)
    padded_results[:results.size(0)] = results
    # Gather all results
    gathered_dim_metadata = [None] * world_size
    gathered_results = torch.zeros_like(padded_results).repeat(
        world_size, *[1] * len(padded_results.shape)
    ).contiguous()  # use contiguous to make sure it won't copy data in the following operations
    dist.all_gather_into_tensor(gathered_results, padded_results)
    torch.cuda.set_device(rank)
    dist.all_gather_object(gathered_dim_metadata, local_dim_metadata)
    # Process gathered results
    data = [[[[] for _ in range(num_w_tiles)] for _ in range(num_h_tiles)]
            for _ in range(num_t_tiles)]
    for current_data, global_idx in parallel_data_generator(gathered_results, gathered_dim_metadata):
        t_idx = global_idx // total_spatial_tiles
        spatial_idx = global_idx % total_spatial_tiles
        h_idx = spatial_idx // num_w_tiles
        w_idx = spatial_idx % num_w_tiles
        data[t_idx][h_idx][w_idx] = current_data
    # Merge results
    result_slices = []
    last_slice_data = None
    for i, tem_data in enumerate(data):
        slice_data = merge_spatial_tiles_cog(tem_data, s_blend_extent,
                                             s_row_limit)
        if i > 0:
            slice_data = blend_t(last_slice_data, slice_data, t_blend_extent)
            result_slices.append(slice_data[:, :, :t_limit, :, :])
        else:
            result_slices.append(slice_data[:, :, :t_limit + 1, :, :])
        last_slice_data = slice_data
    dec = torch.cat(result_slices, dim=2)
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample=dec)


def padded_tensor(tensor, target_shape):
    """Pad tensor to target shape and return the padding sizes."""
    padding = []
    for dim in range(len(tensor.shape)):
        pad_size = target_shape[dim] - tensor.shape[dim]
        if pad_size < 0:
            raise ValueError(
                f"Target shape {target_shape} is smaller than tensor shape {tensor.shape} in dimension {dim}.")

        # Ensure padding is split evenly
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left  # Ensure total padding is equal
        padding.append((pad_left, pad_right))

    # Flatten padding for the pad function
    padding_flat = [p for pair in padding[::-1] for p in pair]  # Reverse and flatten

    padded = torch.nn.functional.pad(tensor, padding_flat)  # Apply padding
    padding_tensor = torch.tensor(padding, device=tensor.device)
    return padded, padding_tensor  # Return both padded tensor and padding sizes


def depadded_tensor(padded, padding):
    """Remove padding from the padded tensor based on the recorded padding sizes."""
    padding_flat = [p for pair in padding for p in pair]  # Flatten padding for the unpad function
    # Check tensor dimensions

    # Create slices to remove padding
    slices = []
    for i in range(0, len(padding_flat), 2):
        start = padding_flat[i]
        end = padded.shape[i // 2] - padding_flat[i + 1]
        if end < start:
            raise ValueError(f"Invalid slice: start {start} should not be greater than end {end}.")
        slices.append(slice(start, end))

    return padded[tuple(slices)]


def tiled_encode_parallel(
        self,
        x: torch.Tensor,
        *,
        overlap_height: int,
        overlap_width: int,
        blend_extent_height: int,
        blend_extent_width: int,
        row_limit_height: int,
        row_limit_width: int,
        num_tiles_height: int,
        num_tiles_width: int,
        use_conv_cache: bool = True,
) -> torch.Tensor:
    """
    Encode a 5-D video/latent tensor in a memory-friendly, tile-wise fashion
    and seamlessly stitch the individual tiles back together.

    Parameters
    ----------
    x
        Input tensor of shape ``(B, C, T, H, W)``.
    overlap_height / overlap_width
        Stride between adjacent tiles (in spatial pixels); determines
        the *sample*-space overlap.
    blend_extent_height / blend_extent_width
        Width (in latent pixels) of the vertical / horizontal band used
        to alpha-blend neighbouring tiles.
    row_limit_height / row_limit_width
        Portion of each tile that survives after blending
        (``tile_latent_size – blend_extent``).
    num_tiles_height / num_tiles_width
        Grid dimensions computed for the sample-space tiling.

    Returns
    -------
    torch.Tensor
        The full latent tensor (`B × C × T × latent_H × latent_W`)
        reconstructed from the tiled encoding.
    """

    # ------------------------------------------------------------------
    # 1) Distributed bookkeeping
    # ------------------------------------------------------------------
    batch_size, num_channels, num_frames, height, width = x.shape
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Flatten the 2-D tile grid into a list and shard it across ranks
    tile_indices = [(i, j)
                    for i in range(num_tiles_height)
                    for j in range(num_tiles_width)]
    total_tiles = len(tile_indices)
    tiles_per_rank = total_tiles // world_size
    start_index = rank * tiles_per_rank
    end_index = start_index + tiles_per_rank if rank != world_size - 1 else total_tiles

    # ------------------------------------------------------------------
    # 2) Encode the tiles assigned to this rank
    # ------------------------------------------------------------------
    decoded_tiles: List[torch.Tensor] = []
    padding_records: List[torch.Tensor] = []

    for index in range(start_index, end_index):
        i, j = tile_indices[index]

        # Sample-space boundaries of the current tile
        start_h = i * overlap_height
        end_h = min(start_h + self.tile_sample_min_height, height)
        start_w = j * overlap_width
        end_w = min(start_w + self.tile_sample_min_width, width)
        tile = x[:, :, :, start_h:end_h, start_w:end_w].to(f"cuda:{rank}")

        # Encode tile → latent, then (optionally) post-process
        conv_cache = None
        if use_conv_cache:
            enc_tile, conv_cache = self.encoder(tile, conv_cache=conv_cache)
        else:
            enc_tile = self.encoder(tile)
        if self.quant_conv is not None:
            enc_tile = self.quant_conv(enc_tile)

        # Right-pad to the uniform latent size expected by collective ops
        padded_tile, padding = padded_tensor(enc_tile, self.max_shape_encode)
        decoded_tiles.append(padded_tile)
        padding_records.append(padding)

    # ------------------------------------------------------------------
    # 3) All-gather paddings and tiles so every rank holds the full batch
    # ------------------------------------------------------------------
    # Gather padding records
    gathered_pads = [
        torch.empty_like(torch.cat(padding_records, dim=0))
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_pads, torch.cat(padding_records, dim=0).contiguous())
    gathered_paddings_tensor = torch.cat(gathered_pads, dim=0)

    # Gather decoded tiles
    gathered_tiles = [
        torch.empty_like(torch.cat(decoded_tiles, dim=0))
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_tiles, torch.cat(decoded_tiles, dim=0).contiguous())

    # ------------------------------------------------------------------
    # 4) Build a convenient structure for depadding
    # ------------------------------------------------------------------
    gathered_paddings1 = gathered_paddings_tensor.view(
        num_tiles_height, num_tiles_width, 5, *gathered_paddings_tensor.shape[1:]
    )

    gathered_paddings: List[List[List[Tuple[int, int]]]] = []
    for i in range(num_tiles_height):
        row: List[List[Tuple[int, int]]] = []
        for j in range(num_tiles_width):
            padding = gathered_paddings1[i, j]
            # `p` is shaped (2,) → convert to Python tuple
            padding_list = [tuple(p.cpu().numpy()) for p in padding]
            row.append(padding_list)
        gathered_paddings.append(row)

    # ------------------------------------------------------------------
    # 5) Depad every tile and place them in a 2-D latent grid
    # ------------------------------------------------------------------
    all_tiles = torch.cat(gathered_tiles, dim=0)
    latent_grid = all_tiles.view(
        num_tiles_height, num_tiles_width, *all_tiles.shape[1:]
    )

    rows: List[List[torch.Tensor]] = [
        [
            depadded_tensor(latent_grid[i, j].unsqueeze(0),
                            gathered_paddings[i][j])
            for j in range(num_tiles_width)
        ]
        for i in range(num_tiles_height)
    ]

    # ------------------------------------------------------------------
    # 6) Blend overlaps to avoid seams
    # ------------------------------------------------------------------
    blended_rows: List[torch.Tensor] = []
    for i, row in enumerate(rows):
        blended_row: List[torch.Tensor] = []
        for j, tile in enumerate(row):
            # Blend with tile above / left whenever those exist
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent_width)

            # Retain only the valid interior (non-overlapping strip)
            tile = tile[:, :, :, :row_limit_height, :row_limit_width]
            blended_row.append(tile)

        # Concatenate tiles horizontally
        blended_rows.append(torch.cat(blended_row, dim=4))

    # ------------------------------------------------------------------
    # 7) Concatenate the rows vertically to obtain the full latent
    # ------------------------------------------------------------------
    enc = torch.cat(blended_rows, dim=3)

    return enc


def merge_spatial_tiles_blend(
        spatial_rows: List[List[torch.Tensor]],
        h_blend_extent: int,
        w_blend_extent: int,
):
    n_rows = len(spatial_rows)
    result_rows = []
    for i, row in enumerate(spatial_rows):
        result_row = []
        n_cols = len(row)
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(spatial_rows[i - 1][j], tile, h_blend_extent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, w_blend_extent)

            # Decide how much of the blending padding to crop away.
            # sample_pad_size = h_blend_extent // 2  (same logic applies to width).
            #
            # – Interior tiles (not on the outer video border):
            #   • We discard the full padding on the *bottom* and *right* sides
            #     (h_blend_extent == 2 * sample_pad_size), because those pixels
            #     have already been blended with the neighbouring tiles.
            #   • We keep the padding on the *top* and *left* sides so the
            #     overlap used for blending is preserved for the tiles that come
            #     before them.
            #
            # – Border tiles:
            #   • First row / first column: crop only the outer half-padding
            #     (sample_pad_size) on the *top* / *left* edge so the frame keeps
            #     its blend with the interior while removing the extra canvas
            #     outside the video.
            #   • Last row / last column: same idea for the *bottom* / *right*
            #     edge.
            #
            # The variables below encode this logic.
            top_crop = h_blend_extent // 2 if i == 0 else 0
            bottom_crop = h_blend_extent // 2 if i == n_rows - 1 else h_blend_extent
            left_crop = w_blend_extent // 2 if j == 0 else 0
            right_crop = w_blend_extent // 2 if j == n_cols - 1 else w_blend_extent

            result_row.append(tile[..., top_crop: -bottom_crop, left_crop: -right_crop])

        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def merge_spatial_tiles(
        spatial_rows: List[List[torch.Tensor]],
        sample_pad_size: int,
):
    result_rows = []
    for i, row in enumerate(spatial_rows):
        result_row = []
        for j, tile in enumerate(row):
            result_row.append(tile[:, :, :, sample_pad_size:-sample_pad_size, sample_pad_size:-sample_pad_size])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def floor_to_multiple(x: int, m: int) -> int:
    return (x // m) * m


def get_patch_hw(patch_size):
    if isinstance(patch_size, int):
        return patch_size, patch_size
    ps = list(patch_size)
    if len(ps) == 1:
        return int(ps[0]), int(ps[0])
    return int(ps[-2]), int(ps[-1])
