"""
Unit tests for vae_mgr.py — VAEManager and helper functions.

Covers:
- split_rectangle: optimal grid splitting
- VAEManager: init validation, tile extraction, align_out, merge
- blend_v / blend_h / blend_t: blending functions
- padded_tensor / depadded_tensor: padding round-trip
- get_patch_hw / lcm / floor_to_multiple: utility functions
"""
import math
from unittest.mock import patch, MagicMock

import pytest
import torch

from x_base.vae_parallelism.vae_mgr import (
    VAEManager,
    split_rectangle,
    blend_v,
    blend_h,
    blend_t,
    padded_tensor,
    depadded_tensor,
    get_patch_hw,
    lcm,
    floor_to_multiple,
    merge_spatial_tiles_cog,
    merge_spatial_tiles_blend,
    merge_spatial_tiles,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def _mock_infer_info():
    with patch("x_base.vae_parallelism.vae_mgr.infer_info") as mock_info:
        mock_info.vae_pad_latent_size = 8
        mock_info.get_tile_world_size = lambda ws: ws
        mock_info.task_type = "t2v"
        yield mock_info


@pytest.fixture(autouse=True)
def _mock_dist():
    with patch("x_base.vae_parallelism.vae_mgr.dist") as mock_dist:
        mock_dist.is_initialized.return_value = False
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1
        yield mock_dist


def _make_mgr(**kwargs):
    defaults = dict(
        input_shape=(1, 3, 1, 256, 256),
        factor=8,
        is_encode=False,
        pad_mode="all_sides_valid",
        pad_content="constant",
        use_blend=True,
    )
    defaults.update(kwargs)
    return VAEManager(**defaults)


# ============================================================
# split_rectangle
# ============================================================

class TestSplitRectangle:
    def test_square_4_cards(self):
        r, c = split_rectangle(512, 512, 4)
        assert r * c == 4
        assert r == 2 and c == 2

    def test_wide_image_prefers_more_cols(self):
        r, c = split_rectangle(256, 1024, 4)
        assert r * c == 4
        assert c >= r

    def test_tall_image_prefers_more_rows(self):
        r, c = split_rectangle(1024, 256, 4)
        assert r * c == 4
        assert r >= c

    def test_single_card(self):
        r, c = split_rectangle(512, 512, 1)
        assert (r, c) == (1, 1)

    def test_prime_world_size(self):
        r, c = split_rectangle(512, 512, 7)
        assert r * c == 7

    def test_returns_factor_pair(self):
        r, c = split_rectangle(100, 200, 6)
        assert r * c == 6


# ============================================================
# VAEManager.__init__
# ============================================================

class TestVAEManagerInit:
    def test_invalid_input_shape_dims(self):
        with pytest.raises(ValueError, match="5 dimensions"):
            VAEManager(input_shape=(1, 3, 256))

    def test_decode_mode_attributes(self):
        mgr = _make_mgr(is_encode=False)
        assert mgr.is_encode is False
        assert mgr.in_pad_size == 8
        assert mgr.out_pad_size == 64

    def test_encode_mode_attributes(self):
        mgr = _make_mgr(is_encode=True)
        assert mgr.is_encode is True
        assert mgr.in_pad_size == 64
        assert mgr.out_pad_size == 8

    def test_tail_only_stride_and_window(self):
        mgr = _make_mgr(pad_mode="tail_only", is_encode=False)
        assert mgr.in_stride_h > 0
        assert mgr.in_window_h == mgr.in_stride_h + mgr.in_pad_size

    def test_all_sides_stride_and_window(self):
        mgr = _make_mgr(pad_mode="all_sides", is_encode=False)
        assert mgr.in_window_h == mgr.in_stride_h + 2 * mgr.in_pad_size

    def test_all_sides_valid_stride_and_window(self):
        mgr = _make_mgr(pad_mode="all_sides_valid", is_encode=False)
        assert mgr.in_window_h == mgr.in_stride_h + 2 * mgr.in_pad_size

    def test_blend_sizes(self):
        mgr = _make_mgr(is_encode=False)
        assert mgr.blend_h == mgr.out_window_h - mgr.out_stride_h
        assert mgr.blend_w == mgr.out_window_w - mgr.out_stride_w

    def test_encode_output_dimensions(self):
        mgr = _make_mgr(is_encode=True, factor=8)
        assert mgr.out_stride_h == int(mgr.in_stride_h / 8)
        assert mgr.out_window_h == int(mgr.in_window_h / 8)

    def test_decode_output_dimensions(self):
        mgr = _make_mgr(is_encode=False, factor=8)
        assert mgr.out_stride_h == mgr.in_stride_h * 8
        assert mgr.out_window_h == mgr.in_window_h * 8

    def test_tile_world_size_mapping(self):
        with patch("x_base.vae_parallelism.vae_mgr.infer_info") as mock_info:
            mock_info.vae_pad_latent_size = 8
            mock_info.get_tile_world_size = lambda ws: 8 if ws == 16 else ws
            mock_info.task_type = "t2v"
            with patch("x_base.vae_parallelism.vae_mgr.dist") as mock_dist:
                mock_dist.is_initialized.return_value = True
                mock_dist.get_rank.return_value = 10
                mock_dist.get_world_size.return_value = 16
                mgr = VAEManager(input_shape=(1, 3, 1, 512, 512), factor=8, is_encode=False)
                assert mgr.tile_world_size == 8
                assert mgr.tile_rank == 10 % 8


# ============================================================
# VAEManager.get_tile_from_x — all_sides_valid
# ============================================================

class TestGetTileFromXValidMode:
    def test_center_tile_no_pad(self):
        mgr = _make_mgr(input_shape=(1, 3, 1, 256, 256), is_encode=False, pad_mode="all_sides_valid")
        x = torch.randn(1, 3, 1, 256, 256)
        tile = mgr.get_tile_from_x(x)
        assert tile.shape[0] == 1
        assert tile.shape[1] == 3

    def test_corner_tile_records_pad_info(self):
        mgr = _make_mgr(input_shape=(1, 3, 1, 128, 128), is_encode=False, pad_mode="all_sides_valid")
        x = torch.randn(1, 3, 1, 128, 128)
        mgr.get_tile_from_x(x)
        pl, pr, pt, pb = mgr._curr_pad_info
        assert isinstance(pl, int) and isinstance(pt, int)

    def test_large_image_no_boundary_pad(self):
        mgr = _make_mgr(input_shape=(1, 3, 1, 1024, 1024), is_encode=False, pad_mode="all_sides_valid")
        x = torch.randn(1, 3, 1, 1024, 1024)
        tile = mgr.get_tile_from_x(x)
        pl, pr, pt, pb = mgr._curr_pad_info
        assert pl == 0 and pt == 0


# ============================================================
# VAEManager.get_tile_from_x — tail_only / all_sides
# ============================================================

class TestGetTileFromXTailAndAllSides:
    def test_tail_only_tile_shape(self):
        mgr = _make_mgr(pad_mode="tail_only", is_encode=False)
        x = torch.randn(1, 3, 1, 256, 256)
        tile = mgr.get_tile_from_x(x)
        assert tile.shape[2] == 1
        assert tile.shape[3] == mgr.in_window_h
        assert tile.shape[4] == mgr.in_window_w

    def test_all_sides_tile_shape(self):
        mgr = _make_mgr(pad_mode="all_sides", is_encode=False)
        x = torch.randn(1, 3, 1, 256, 256)
        tile = mgr.get_tile_from_x(x)
        assert tile.shape[3] == mgr.in_window_h
        assert tile.shape[4] == mgr.in_window_w

    def test_all_sides_reflect_pad(self):
        mgr = _make_mgr(pad_mode="all_sides", pad_content="reflect", is_encode=False)
        x = torch.randn(1, 3, 1, 256, 256)
        tile = mgr.get_tile_from_x(x)
        assert tile.shape[3] == mgr.in_window_h

    def test_all_sides_replicate_pad_bf16(self):
        mgr = _make_mgr(pad_mode="all_sides", pad_content="replicate", is_encode=False)
        x = torch.randn(1, 3, 1, 256, 256, dtype=torch.bfloat16)
        tile = mgr.get_tile_from_x(x)
        assert tile.dtype == torch.bfloat16
        assert tile.shape[3] == mgr.in_window_h


# ============================================================
# VAEManager.align_out
# ============================================================

class TestAlignOut:
    def test_valid_mode_pads_missing_pixels(self):
        mgr = _make_mgr(pad_mode="all_sides_valid", is_encode=False)
        x = torch.randn(1, 3, 1, 256, 256)
        mgr.get_tile_from_x(x)
        pl, pr, pt, pb = mgr._curr_pad_info
        out = torch.randn(1, 3, 1, mgr.out_window_h - pb * mgr.factor - pt * mgr.factor,
                          mgr.out_window_w - pr * mgr.factor - pl * mgr.factor)
        aligned = mgr.align_out(out)
        assert aligned.shape[-2] == mgr.out_window_h
        assert aligned.shape[-1] == mgr.out_window_w

    def test_valid_mode_encode_divides_by_factor(self):
        mgr = _make_mgr(pad_mode="all_sides_valid", is_encode=True)
        x = torch.randn(1, 3, 1, 256, 256)
        mgr.get_tile_from_x(x)
        pl, pr, pt, pb = mgr._curr_pad_info
        out_h = mgr.out_window_h - (pt // mgr.factor + pb // mgr.factor)
        out_w = mgr.out_window_w - (pl // mgr.factor + pr // mgr.factor)
        out = torch.randn(1, 3, 1, out_h, out_w)
        aligned = mgr.align_out(out)
        assert aligned.shape[-2] == mgr.out_window_h

    def test_valid_mode_no_pad_needed(self):
        mgr = _make_mgr(pad_mode="all_sides_valid", is_encode=False)
        x = torch.randn(1, 3, 1, 1024, 1024)
        mgr.get_tile_from_x(x)
        out = torch.randn(1, 3, 1, mgr.out_window_h, mgr.out_window_w)
        aligned = mgr.align_out(out)
        assert torch.equal(aligned, out)

    def test_tail_only_pads_right_bottom(self):
        mgr = _make_mgr(pad_mode="tail_only", is_encode=False)
        smaller_h = mgr.out_window_h - 10
        smaller_w = mgr.out_window_w - 10
        out = torch.randn(1, 3, 1, smaller_h, smaller_w)
        aligned = mgr.align_out(out)
        assert aligned.shape[-2] == mgr.out_window_h
        assert aligned.shape[-1] == mgr.out_window_w


# ============================================================
# VAEManager._blend_tile / _crop_tile / _merge_tiles
# ============================================================

class TestMergeTiles:
    def _make_grid(self, mgr, n_rows, n_cols):
        grid = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                tile = torch.ones(1, 3, 1, mgr.out_window_h, mgr.out_window_w) * (i * n_cols + j + 1)
                row.append(tile)
            grid.append(row)
        return grid

    def test_blend_tile_no_blend(self):
        mgr = _make_mgr(use_blend=False)
        tile = torch.ones(1, 3, 1, 32, 32)
        grid = [[torch.ones(1, 3, 1, 32, 32) * 2]]
        result = mgr._blend_tile(0, 0, grid, tile)
        assert torch.equal(result, tile)

    def test_blend_tile_with_above(self):
        mgr = _make_mgr(use_blend=True)
        tile = torch.ones(1, 3, 1, mgr.out_window_h, mgr.out_window_w)
        grid = [[torch.ones(1, 3, 1, mgr.out_window_h, mgr.out_window_w) * 2]]
        mgr.blend_h = mgr.blend_w = 0
        result = mgr._blend_tile(1, 0, grid, tile)
        assert result.shape == tile.shape

    def test_crop_tile_tail_only_interior(self):
        mgr = _make_mgr(pad_mode="tail_only", use_blend=False)
        tile = torch.ones(1, 3, 1, mgr.out_window_h, mgr.out_window_w)
        cropped = mgr._crop_tile(0, 0, 2, 2, tile)
        assert cropped.shape[-2] == mgr.out_stride_h
        assert cropped.shape[-1] == mgr.out_stride_w

    def test_crop_tile_tail_only_last(self):
        mgr = _make_mgr(pad_mode="tail_only", use_blend=False)
        tile = torch.ones(1, 3, 1, mgr.out_window_h, mgr.out_window_w)
        cropped = mgr._crop_tile(1, 1, 2, 2, tile)
        assert cropped.shape[-2] == mgr.out_window_h
        assert cropped.shape[-1] == mgr.out_window_w

    def test_crop_tile_all_sides_no_blend(self):
        mgr = _make_mgr(pad_mode="all_sides", use_blend=False)
        tile = torch.ones(1, 3, 1, mgr.out_window_h, mgr.out_window_w)
        cropped = mgr._crop_tile(0, 0, 2, 2, tile)
        expected_h = mgr.out_window_h - 2 * mgr.out_pad_size
        expected_w = mgr.out_window_w - 2 * mgr.out_pad_size
        assert cropped.shape[-2] == expected_h
        assert cropped.shape[-1] == expected_w

    def test_merge_tiles_shape(self):
        mgr = _make_mgr(pad_mode="all_sides_valid", use_blend=True)
        n_rows, n_cols = mgr.num_h_tiles, mgr.num_w_tiles
        grid = self._make_grid(mgr, n_rows, n_cols)
        with patch.object(mgr, "_blend_tile", side_effect=lambda i, j, g, t: t):
            with patch.object(mgr, "_crop_tile", side_effect=lambda i, j, nr, nc, t: t):
                result = mgr._merge_tiles(grid)
        assert result.ndim == 5


# ============================================================
# blend_v / blend_h / blend_t
# ============================================================

class TestBlendFunctions:
    def test_blend_v_shape_preserved(self):
        a = torch.randn(1, 2, 3, 10, 8)
        b = torch.randn(1, 2, 3, 10, 8)
        result = blend_v(a, b, blend_extent=4)
        assert result.shape == b.shape

    def test_blend_v_transition(self):
        a = torch.zeros(1, 1, 1, 10, 8)
        b = torch.ones(1, 1, 1, 10, 8)
        result = blend_v(a, b, blend_extent=4)
        assert result.shape == b.shape
        assert (result[..., :4, :] >= 0).all()
        assert (result[..., :4, :] <= 1).all()

    def test_blend_h_shape_preserved(self):
        a = torch.randn(1, 2, 3, 8, 10)
        b = torch.randn(1, 2, 3, 8, 10)
        result = blend_h(a, b, blend_extent=4)
        assert result.shape == b.shape

    def test_blend_h_transition(self):
        a = torch.zeros(1, 1, 1, 8, 10)
        b = torch.ones(1, 1, 1, 8, 10)
        result = blend_h(a, b, blend_extent=4)
        assert (result[..., :4] >= 0).all()
        assert (result[..., :4] <= 1).all()

    def test_blend_t_shape_preserved(self):
        a = torch.randn(2, 3, 10, 8, 8)
        b = torch.randn(2, 3, 10, 8, 8)
        result = blend_t(a, b, blend_extent=4)
        assert result.shape == b.shape

    def test_blend_t_transition(self):
        a = torch.zeros(2, 3, 10, 8, 8)
        b = torch.ones(2, 3, 10, 8, 8)
        result = blend_t(a, b, blend_extent=4)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_blend_v_extent_exceeds_dim(self):
        a = torch.randn(1, 1, 1, 2, 8)
        b = torch.randn(1, 1, 1, 2, 8)
        result = blend_v(a, b, blend_extent=100)
        assert result.shape == b.shape

    def test_blend_h_extent_exceeds_dim(self):
        a = torch.randn(1, 1, 1, 8, 2)
        b = torch.randn(1, 1, 1, 8, 2)
        result = blend_h(a, b, blend_extent=100)
        assert result.shape == b.shape


# ============================================================
# padded_tensor / depadded_tensor round-trip
# ============================================================

class TestPaddedDepaddedTensor:
    def test_round_trip_1d(self):
        t = torch.randn(10)
        padded, padding = padded_tensor(t, (16,))
        assert padded.shape == (16,)
        recovered = depadded_tensor(padded, padding)
        assert torch.allclose(recovered, t)

    def test_round_trip_2d(self):
        t = torch.randn(3, 5)
        padded, padding = padded_tensor(t, (6, 8))
        assert padded.shape == (6, 8)
        recovered = depadded_tensor(padded, padding)
        assert torch.allclose(recovered, t)

    def test_round_trip_5d(self):
        t = torch.randn(1, 3, 1, 64, 64)
        padded, padding = padded_tensor(t, (1, 3, 1, 80, 80))
        recovered = depadded_tensor(padded, padding)
        assert torch.allclose(recovered, t)

    def test_no_padding_needed(self):
        t = torch.randn(4, 4)
        padded, padding = padded_tensor(t, (4, 4))
        assert padded.shape == (4, 4)
        recovered = depadded_tensor(padded, padding)
        assert torch.allclose(recovered, t)

    def test_target_smaller_raises(self):
        t = torch.randn(10)
        with pytest.raises(ValueError, match="smaller"):
            padded_tensor(t, (5,))


# ============================================================
# Utility functions
# ============================================================

class TestGetPatchHw:
    def test_int_input(self):
        assert get_patch_hw(4) == (4, 4)

    def test_single_element_list(self):
        assert get_patch_hw([4]) == (4, 4)

    def test_two_element_list(self):
        assert get_patch_hw([2, 4]) == (2, 4)

    def test_tuple_input(self):
        assert get_patch_hw((1, 2, 3)) == (2, 3)


class TestLcm:
    def test_basic(self):
        assert lcm(4, 6) == 12

    def test_coprime(self):
        assert lcm(7, 5) == 35

    def test_same(self):
        assert lcm(8, 8) == 8

    def test_one(self):
        assert lcm(1, 9) == 9


class TestFloorToMultiple:
    def test_exact(self):
        assert floor_to_multiple(16, 8) == 16

    def test_non_exact(self):
        assert floor_to_multiple(17, 8) == 16

    def test_zero(self):
        assert floor_to_multiple(0, 8) == 0

    def test_smaller_than_multiple(self):
        assert floor_to_multiple(3, 8) == 0


# ============================================================
# merge_spatial_tiles_cog
# ============================================================

class TestMergeSpatialTilesCog:
    def test_single_tile(self):
        tile = torch.randn(1, 3, 1, 32, 32)
        grid = [[tile]]
        result = merge_spatial_tiles_cog(grid, blend_extent=4, row_limit=28)
        assert result.shape[-2] == 28
        assert result.shape[-1] == 28

    def test_2x2_grid(self):
        grid = [[torch.randn(1, 3, 1, 32, 32) for _ in range(2)] for _ in range(2)]
        result = merge_spatial_tiles_cog(grid, blend_extent=4, row_limit=28)
        assert result.ndim == 5


# ============================================================
# merge_spatial_tiles_blend
# ============================================================

class TestMergeSpatialTilesBlend:
    def test_single_tile(self):
        tile = torch.randn(1, 3, 1, 32, 32)
        grid = [[tile]]
        result = merge_spatial_tiles_blend(grid, h_blend_extent=8, w_blend_extent=8)
        assert result.ndim == 5

    def test_2x1_grid(self):
        grid = [[torch.randn(1, 3, 1, 32, 32), torch.randn(1, 3, 1, 32, 32)]]
        result = merge_spatial_tiles_blend(grid, h_blend_extent=8, w_blend_extent=8)
        assert result.ndim == 5


# ============================================================
# merge_spatial_tiles (no blend)
# ============================================================

class TestMergeSpatialTiles:
    def test_1x2_grid(self):
        grid = [[torch.randn(1, 3, 1, 32, 32), torch.randn(1, 3, 1, 32, 32)]]
        result = merge_spatial_tiles(grid, sample_pad_size=4)
        assert result.ndim == 5
        assert result.shape[-2] == 32 - 2 * 4
