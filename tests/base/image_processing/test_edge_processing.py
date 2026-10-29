from __future__ import annotations

import numpy as np
import pytest

from dl_toolbox.base.image_processing.edge_processing import (
    SCHARR_STEP_SCALE,
    absolute_normalized_scharr_magnitude,
    non_maximum_suppression,
    robust_percentile_normalized_scharr_magnitude,
    scharr_derivatives,
    scharr_edge_strength,
    to_gray_level_derivatives,
)


def _vertical_step(*, height: int, width: int, delta: int, dtype) -> np.ndarray:
    image = np.zeros((height, width), dtype=dtype)
    image[:, width // 2 :] = delta
    return image


def test_scharr_gray_level_matches_axis_aligned_step() -> None:
    delta = 10
    image = _vertical_step(height=21, width=21, delta=delta, dtype=np.uint8)
    gx_kernel, gy_kernel = scharr_derivatives(image)
    gx_gray, gy_gray = to_gray_level_derivatives(gx_kernel, gy_kernel)
    row = 10
    col = 10
    assert SCHARR_STEP_SCALE == 16.0
    assert float(gx_kernel[row, col]) == pytest.approx(16.0 * delta, abs=1e-3)
    assert float(gx_gray[row, col]) == pytest.approx(float(delta), abs=1e-3)
    assert float(gy_gray[row, col]) == pytest.approx(0.0, abs=1e-3)
    magnitude, _theta_deg = scharr_edge_strength(image)
    assert float(magnitude[row, col]) == pytest.approx(float(delta), abs=1e-3)


def test_scharr_rejects_3d_stack() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        scharr_edge_strength(np.zeros((8, 8, 3), dtype=np.uint8))


def test_absolute_normalized_magnitude_is_dim_on_uint16_ceiling() -> None:
    image = _vertical_step(height=21, width=21, delta=10, dtype=np.uint16)
    unit = absolute_normalized_scharr_magnitude(image, source_dtype=np.uint16)
    assert float(np.max(unit)) < 0.01


def test_robust_percentile_normalized_magnitude_requires_interval() -> None:
    image = _vertical_step(height=32, width=32, delta=40, dtype=np.uint8)
    with pytest.raises(TypeError):
        robust_percentile_normalized_scharr_magnitude(image)
    stretched = robust_percentile_normalized_scharr_magnitude(
        image,
        robust_low_percentile=0.1,
        robust_high_percentile=99.9,
    )
    assert stretched.shape == image.shape
    assert float(np.max(stretched)) == pytest.approx(1.0, abs=1e-5)


def test_nms_keeps_the_vertical_ridge() -> None:
    image = _vertical_step(height=21, width=21, delta=10, dtype=np.uint8)
    magnitude, theta_deg = scharr_edge_strength(image)
    nms = non_maximum_suppression(magnitude, theta_deg)
    assert nms.shape == magnitude.shape
    assert float(nms[10, 10]) > 0.0
    assert float(nms[10, 0]) == 0.0
