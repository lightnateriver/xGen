"""
Unit tests for preprocess/vace_preprocess.py.

Covers:
- i2v_prepare_video_and_mask: first frame is image, rest are gray, mask pattern
- v2lf_prepare_video_and_mask: last frame is image, rest are gray, mask pattern
- flf2v_prepare_video_and_mask: first+last frames are images, middle gray, mask pattern
- random2v_prepare_video_and_mask: images placed at specified indices
- inpaint_prepare_video_and_mask: center strip masked, frames composited
- outpaint_prepare_video_and_mask: directional expansion
- iwri_prepare_video_and_mask: center strip masked (same as inpaint logic)
- prepare_video_and_mask: dispatch by vace_task
"""
from unittest.mock import patch, MagicMock

import pytest
import PIL.Image

from x_base.preprocess.vace_preprocess import (
    i2v_prepare_video_and_mask,
    v2lf_prepare_video_and_mask,
    flf2v_prepare_video_and_mask,
    random2v_prepare_video_and_mask,
    inpaint_prepare_video_and_mask,
    outpaint_prepare_video_and_mask,
    iwri_prepare_video_and_mask,
    prepare_video_and_mask,
    FRAME_COLOR,
    BLACK_COLOR,
    WHITE_COLOR,
)


def _make_img(w=64, h=64, color=(255, 0, 0)):
    return PIL.Image.new("RGB", (w, h), color)


class TestI2v:
    def test_first_frame_is_image(self):
        img = _make_img()
        frames, mask = i2v_prepare_video_and_mask(img, 64, 64, 5)
        assert len(frames) == 5
        assert len(mask) == 5
        assert frames[0] is img

    def test_remaining_frames_are_gray(self):
        img = _make_img()
        frames, mask = i2v_prepare_video_and_mask(img, 64, 64, 5)
        for f in frames[1:]:
            assert f.getpixel((0, 0)) == (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR)

    def test_mask_pattern(self):
        img = _make_img()
        _, mask = i2v_prepare_video_and_mask(img, 64, 64, 5)
        assert mask[0].getpixel((0, 0)) == BLACK_COLOR
        for m in mask[1:]:
            assert m.getpixel((0, 0)) == WHITE_COLOR


class TestV2lf:
    def test_last_frame_is_image(self):
        img = _make_img()
        frames, mask = v2lf_prepare_video_and_mask(img, 64, 64, 5)
        assert frames[-1] is img

    def test_mask_pattern(self):
        img = _make_img()
        _, mask = v2lf_prepare_video_and_mask(img, 64, 64, 5)
        assert mask[-1].getpixel((0, 0)) == BLACK_COLOR
        for m in mask[:-1]:
            assert m.getpixel((0, 0)) == WHITE_COLOR


class TestFlf2v:
    def test_first_and_last_are_images(self):
        first = _make_img(color=(255, 0, 0))
        last = _make_img(color=(0, 255, 0))
        frames, mask = flf2v_prepare_video_and_mask(first, last, 64, 64, 5)
        assert len(frames) == 5
        assert frames[0] is first
        assert frames[-1] is last

    def test_mask_pattern(self):
        first = _make_img()
        last = _make_img()
        _, mask = flf2v_prepare_video_and_mask(first, last, 64, 64, 5)
        assert mask[0].getpixel((0, 0)) == BLACK_COLOR
        assert mask[-1].getpixel((0, 0)) == BLACK_COLOR
        for m in mask[1:-1]:
            assert m.getpixel((0, 0)) == WHITE_COLOR


class TestRandom2v:
    def test_images_at_specified_indices(self):
        imgs = [_make_img(color=(i * 50, 0, 0)) for i in range(3)]
        indices = [0, 3, 7]
        frames, mask = random2v_prepare_video_and_mask(imgs, indices, 64, 64, 10)
        assert len(frames) == 10
        assert len(mask) == 10
        for img, idx in zip(imgs, indices):
            assert frames[idx] is img
            assert mask[idx].getpixel((0, 0)) == BLACK_COLOR

    def test_non_image_frames_are_gray(self):
        imgs = [_make_img()]
        frames, mask = random2v_prepare_video_and_mask(imgs, [0], 64, 64, 5)
        for i in range(1, 5):
            assert mask[i].getpixel((0, 0)) == WHITE_COLOR


class TestInpaint:
    def test_output_length(self):
        video = [_make_img() for _ in range(5)]
        frames, mask = inpaint_prepare_video_and_mask(video, 64, 64, 5)
        assert len(frames) == 5
        assert len(mask) == 5

    def test_mask_has_white_strip(self):
        video = [_make_img() for _ in range(3)]
        _, mask = inpaint_prepare_video_and_mask(video, 64, 64, 3)
        center_pixel = mask[0].getpixel((32, 32))
        assert center_pixel == WHITE_COLOR

    def test_frames_resized(self):
        video = [_make_img(w=128, h=128)]
        frames, _ = inpaint_prepare_video_and_mask(video, 64, 64, 1)
        assert frames[0].size == (64, 64)


class TestOutpaint:
    def test_output_length(self):
        img = _make_img(w=100, h=100)
        frames, mask = outpaint_prepare_video_and_mask(img, ["right"], 0.2, 64, 64, 5)
        assert len(frames) == 5
        assert len(mask) == 5

    def test_first_frame_is_expanded(self):
        img = _make_img(w=100, h=100)
        frames, _ = outpaint_prepare_video_and_mask(img, ["right"], 0.2, 64, 64, 3)
        assert frames[0].size == (100, 100)

    def test_remaining_frames_are_gray(self):
        img = _make_img(w=100, h=100)
        frames, _ = outpaint_prepare_video_and_mask(img, ["right"], 0.2, 64, 64, 3)
        for f in frames[1:]:
            assert f.size == (100, 100)


class TestIwri:
    def test_output_length(self):
        video = [_make_img() for _ in range(5)]
        frames, mask = iwri_prepare_video_and_mask(video, 64, 64, 5)
        assert len(frames) == 5
        assert len(mask) == 5

    def test_mask_has_white_strip(self):
        video = [_make_img() for _ in range(3)]
        _, mask = iwri_prepare_video_and_mask(video, 64, 64, 3)
        center_pixel = mask[0].getpixel((32, 32))
        assert center_pixel == WHITE_COLOR


class TestPrepareVideoAndMask:
    def test_t2v_returns_none(self):
        args = MagicMock()
        args.vace_task = "t2v"
        video, mask, ref = prepare_video_and_mask(args)
        assert video is None
        assert mask is None
        assert ref is None

    def test_i2v_dispatch(self):
        args = MagicMock()
        args.vace_task = "i2v"
        args.image_path = "dummy.png"
        args.height = 64
        args.width = 64
        args.frames = 5
        with patch("x_base.preprocess.vace_preprocess.load_image", return_value=_make_img()):
            with patch("x_base.preprocess.vace_preprocess.i2v_prepare_video_and_mask", return_value=([1], [2])) as mock:
                video, mask, ref = prepare_video_and_mask(args)
                mock.assert_called_once()
        assert video == [1]
        assert mask == [2]
        assert ref is None

    def test_iwri_returns_reference(self):
        args = MagicMock()
        args.vace_task = "iwri"
        args.image_path = "ref.png"
        args.video_path = "video.mp4"
        args.height = 64
        args.width = 64
        args.frames = 5
        with patch("x_base.preprocess.vace_preprocess.load_image", return_value=_make_img()), \
             patch("x_base.preprocess.vace_preprocess.load_video", return_value=[_make_img()] * 10), \
             patch("x_base.preprocess.vace_preprocess.iwri_prepare_video_and_mask", return_value=([1], [2])):
            video, mask, ref = prepare_video_and_mask(args)
        assert ref is not None
