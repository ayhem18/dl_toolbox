from __future__ import annotations

import numpy as np

from dl_toolbox.geometry.masks import rasterize_contour, rasterize_contours


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float64,
    )


def test_rasterize_filled_square_covers_the_interior() -> None:
    contour = _rectangle(10, 10, 20, 20)
    mask = rasterize_contour(contour, height=32, width=32)
    assert mask.dtype == bool
    assert mask.shape == (32, 32)
    assert bool(mask[15, 15])
    assert not bool(mask[0, 0])
    assert not bool(mask[31, 31])


def test_short_contour_is_empty_mask() -> None:
    mask = rasterize_contour([[0, 0], [3, 3]], height=8, width=8)
    assert mask.shape == (8, 8)
    assert not bool(mask.any())


def test_rasterize_contours_ors_members() -> None:
    a = _rectangle(2, 2, 6, 6)
    b = _rectangle(10, 10, 14, 14)
    combined = rasterize_contours([a, b], height=16, width=16)
    only_a = rasterize_contour(a, 16, 16)
    only_b = rasterize_contour(b, 16, 16)
    assert np.array_equal(combined, only_a | only_b)
    assert bool(combined.any())
