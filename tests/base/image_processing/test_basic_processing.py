from __future__ import annotations

import numpy as np
import pytest

from dl_toolbox.base.image_processing.basic_processing import (
    absolute_euclidean_magnitude_ceiling,
    intensity_max_for_dtype,
    normalize_absolute,
    normalize_by_max,
    normalize_by_robust_percentiles,
    require_2d_array,
    unit_interval_to_uint8,
)


def test_require_2d_array_rejects_3d() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        require_2d_array(np.zeros((4, 5, 3), dtype=np.uint8), label="image")


def test_absolute_ceiling_is_dtype_max_times_sqrt2() -> None:
    uint8_ceiling = absolute_euclidean_magnitude_ceiling(source_dtype=np.uint8)
    uint16_ceiling = absolute_euclidean_magnitude_ceiling(source_dtype=np.uint16)
    assert uint8_ceiling == pytest.approx(255.0 * (2.0 ** 0.5))
    assert uint16_ceiling == pytest.approx(65535.0 * (2.0 ** 0.5))
    assert uint16_ceiling > uint8_ceiling
    assert intensity_max_for_dtype(np.uint8) == 255.0


def test_normalize_absolute_uses_source_dtype() -> None:
    plane = np.full((3, 3), 255.0 * (2.0 ** 0.5), dtype=np.float32)
    uint8_unit = normalize_absolute(plane, source_dtype=np.uint8)
    uint16_unit = normalize_absolute(plane, source_dtype=np.uint16)
    assert uint8_unit[1, 1] == pytest.approx(1.0)
    assert uint16_unit[1, 1] < 0.01


def test_normalize_by_robust_percentiles_ignores_outliers() -> None:
    plane = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    plane[0, 0] = 50.0
    stretched = normalize_by_robust_percentiles(
        plane,
        robust_low_percentile=0.1,
        robust_high_percentile=99.9,
    )
    assert stretched[0, 0] == pytest.approx(1.0)
    assert 0.0 <= float(stretched[50, 50]) <= 1.0
    assert float(stretched[50, 50]) < 1.0


def test_normalize_by_robust_percentiles_requires_both_bounds() -> None:
    plane = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(TypeError):
        normalize_by_robust_percentiles(plane, robust_low_percentile=0.1)
    with pytest.raises(TypeError):
        normalize_by_robust_percentiles(plane)


def test_normalize_by_max_and_uint8_roundtrip_peak() -> None:
    plane = np.zeros((4, 4), dtype=np.float32)
    plane[1, 2] = 8.0
    unit = normalize_by_max(plane)
    assert unit[1, 2] == pytest.approx(1.0)
    preview = unit_interval_to_uint8(unit)
    assert preview.dtype == np.uint8
    assert int(preview[1, 2]) == 255
