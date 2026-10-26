from __future__ import annotations

import random

import numpy as np
import pytest

from dl_toolbox.base.geometry.bboxes import (
    compute_stride,
    contours_bbox_xyxy_int,
    generate_tile_boxes_xyxy,
    tile_origins_1d,
)

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float64,
    )


def test_xyxy_int_clips_to_the_image(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        image_w = rng.randint(16, 80)
        image_h = rng.randint(16, 80)
        x0 = rng.uniform(-10, image_w + 5)
        y0 = rng.uniform(-10, image_h + 5)
        x1 = x0 + rng.uniform(1, 20)
        y1 = y0 + rng.uniform(1, 20)
        box = contours_bbox_xyxy_int([_rectangle(x0, y0, x1, y1)], image_w, image_h)
        if box is None:
            continue
        bx0, by0, bx1, by1 = box
        assert 0 <= bx0 < bx1 <= image_w
        assert 0 <= by0 < by1 <= image_h


def test_xyxy_int_rejects_a_box_outside_the_image() -> None:
    contour = _rectangle(100, 100, 110, 110)
    assert contours_bbox_xyxy_int([contour], 8, 8) is None
    assert contours_bbox_xyxy_int([], 64, 64) is None


def test_compute_stride_clamps_overlap_and_stays_positive(
    rng: random.Random,
) -> None:
    assert compute_stride(10, 0.0) == 10
    assert compute_stride(10, 0.5) == 5
    assert compute_stride(10, 2.0) >= 1
    assert compute_stride(10, -1.0) == 10
    for _ in range(N_TRIALS):
        crop = rng.randint(1, 64)
        overlap = rng.uniform(-2.0, 2.0)
        stride = compute_stride(crop, overlap)
        assert stride >= 1


def test_tile_origins_cover_the_axis_and_end_flush(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        length = rng.randint(8, 80)
        crop = rng.randint(1, length)
        stride = rng.randint(1, max(1, crop))
        origins = tile_origins_1d(length, crop, stride)
        assert origins.dtype == np.int32
        assert int(origins[0]) == 0
        assert int(origins[-1]) == length - crop
        assert np.all(origins >= 0)
        assert np.all(origins + crop <= length)
        if length == crop:
            assert origins.tolist() == [0]


def test_tile_origins_reject_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="crop_size"):
        tile_origins_1d(10, 0, 1)
    with pytest.raises(ValueError, match="stride"):
        tile_origins_1d(10, 4, 0)
    with pytest.raises(ValueError, match="length_px"):
        tile_origins_1d(0, 4, 1)
    with pytest.raises(ValueError, match="larger than the frame axis"):
        tile_origins_1d(8, 9, 1)


def test_generate_tile_boxes_are_squares_inside_the_frame(
    rng: random.Random,
) -> None:
    empty = generate_tile_boxes_xyxy(0, 10, 4, 0.5)
    assert empty.shape == (0, 4)
    for _ in range(N_TRIALS):
        width = rng.randint(8, 48)
        height = rng.randint(8, 48)
        crop = rng.randint(1, min(width, height))
        overlap = rng.choice((0.0, 0.25, 0.5, 0.75))
        boxes = generate_tile_boxes_xyxy(width, height, crop, overlap)
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert boxes.dtype == np.int32
        assert boxes.shape[0] >= 1
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        assert np.all(widths == crop)
        assert np.all(heights == crop)
        assert np.all(boxes[:, 0] >= 0)
        assert np.all(boxes[:, 1] >= 0)
        assert np.all(boxes[:, 2] <= width)
        assert np.all(boxes[:, 3] <= height)
        assert int(np.max(boxes[:, 2])) == width
        assert int(np.max(boxes[:, 3])) == height


def test_crop_larger_than_frame_raises() -> None:
    with pytest.raises(ValueError, match="larger than the frame"):
        generate_tile_boxes_xyxy(8, 8, 9, 0.5)
