from __future__ import annotations

import random

import numpy as np
import pytest

from dl_toolbox.base.geometry.contours import contour_aabb_wh, contours_bbox

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float64,
    )


def test_empty_contours_have_no_bbox() -> None:
    assert contours_bbox([]) is None
    assert contours_bbox([np.zeros((0, 2))]) is None
    assert contour_aabb_wh(np.zeros((0, 2))) is None


def test_bbox_is_axis_aligned_extrema(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        x0 = rng.uniform(0, 40)
        y0 = rng.uniform(0, 40)
        width = rng.uniform(1, 30)
        height = rng.uniform(1, 30)
        x1 = x0 + width
        y1 = y0 + height
        contour = _rectangle(x0, y0, x1, y1)
        bbox = contours_bbox([contour])
        assert bbox is not None
        assert bbox == (x0, x1, y0, y1)
        aabb = contour_aabb_wh(contour)
        assert aabb is not None
        assert aabb == (bbox[1] - bbox[0], bbox[3] - bbox[2])
        assert aabb == pytest.approx((width, height))


def test_union_bbox_covers_every_contour() -> None:
    left = _rectangle(2, 4, 8, 10)
    right = _rectangle(20, 1, 25, 30)
    bbox = contours_bbox([left, right])
    assert bbox == (2.0, 25.0, 1.0, 30.0)
