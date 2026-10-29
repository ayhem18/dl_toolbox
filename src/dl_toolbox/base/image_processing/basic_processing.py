"""Array-level scaling. No operator, no hidden interval, no default stretch.

The robust-percentile stretch is an explicit call:
``normalize_by_robust_percentiles(..., robust_low_percentile=...,
robust_high_percentile=...)``. Those two percentiles are required. They are
not taken from a module constant inside this function.
"""

from __future__ import annotations

import numpy as np

from dl_toolbox.base.parsing.parse_numerical import parse_percentile


def require_2d_array(array: np.ndarray, *, label: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{label} must be a numpy array, got {type(array)!r}")
    if array.ndim != 2:
        raise ValueError(f"{label} must be 2-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    return array


def to_float32_2d(array: np.ndarray, *, label: str) -> np.ndarray:
    plane = require_2d_array(array, label=label)
    return np.ascontiguousarray(plane, dtype=np.float32)


def intensity_max_for_dtype(source_dtype: object) -> float:
    dtype = np.dtype(source_dtype)
    if dtype.kind not in ("u", "i"):
        raise TypeError(
            "source_dtype must be an integer dtype so the absolute intensity "
            f"ceiling is defined, got {dtype}"
        )
    return float(np.iinfo(dtype).max)


def absolute_euclidean_magnitude_ceiling(*, source_dtype: object) -> float:
    """Largest Euclidean gradient magnitude in gray-level units.

    After the Scharr kernel gain is divided out, an axis-aligned intensity
    step of ``intensity_max_for_dtype(source_dtype)`` has magnitude equal to
    that max. Both axes at that max give ``max * sqrt(2)``.
    """
    return intensity_max_for_dtype(source_dtype) * (2.0 ** 0.5)


def normalize_absolute(array: np.ndarray, *, source_dtype: object) -> np.ndarray:
    """Divide by the absolute Euclidean ceiling of ``source_dtype``, clip to [0, 1]."""
    plane = to_float32_2d(array, label="array")
    ceiling = absolute_euclidean_magnitude_ceiling(source_dtype=source_dtype)
    if ceiling <= 0.0:
        raise ValueError(f"absolute Euclidean ceiling must be > 0, got {ceiling}")
    return np.clip(plane / np.float32(ceiling), 0.0, 1.0)


def normalize_by_max(array: np.ndarray) -> np.ndarray:
    """Divide by this array's own maximum. A single hot pixel sets the scale."""
    plane = to_float32_2d(array, label="array")
    peak = float(np.max(plane))
    if peak <= 0.0:
        raise ValueError("normalize_by_max requires a positive maximum")
    return np.clip(plane / np.float32(peak), 0.0, 1.0)


def normalize_by_robust_percentiles(
    array: np.ndarray,
    *,
    robust_low_percentile: float,
    robust_high_percentile: float,
) -> np.ndarray:
    """Robust percentile stretch onto [0, 1].

    ``robust_low_percentile`` and ``robust_high_percentile`` are the inclusive
    interval of the *distribution of this array* that is mapped to [0, 1].
    Values below the low robust percentile clip to 0. Values above the high
    robust percentile clip to 1. Outliers outside that interval do not set
    the scale. This is not min-max and not an absolute dtype ceiling.

    Both percentiles are required arguments. There is no default interval.
    """
    parsed_robust_low_percentile = parse_percentile(
        robust_low_percentile, label="robust_low_percentile"
    )
    parsed_robust_high_percentile = parse_percentile(
        robust_high_percentile, label="robust_high_percentile"
    )
    if not float(parsed_robust_low_percentile) < float(parsed_robust_high_percentile):
        raise ValueError(
            "robust_low_percentile must be < robust_high_percentile, got "
            f"{parsed_robust_low_percentile} and {parsed_robust_high_percentile}"
        )
    plane = to_float32_2d(array, label="array")
    robust_low_value, robust_high_value = np.percentile(
        plane,
        (
            float(parsed_robust_low_percentile),
            float(parsed_robust_high_percentile),
        ),
    )
    robust_span = float(robust_high_value) - float(robust_low_value)
    if robust_span <= 0.0:
        raise ValueError(
            "robust percentile interval has zero span: "
            f"robust_low_percentile={parsed_robust_low_percentile} "
            f"robust_high_percentile={parsed_robust_high_percentile} "
            f"robust_low_value={robust_low_value} "
            f"robust_high_value={robust_high_value}"
        )
    stretched = (plane - np.float32(robust_low_value)) / np.float32(robust_span)
    return np.clip(stretched, 0.0, 1.0)


def unit_interval_to_uint8(array: np.ndarray) -> np.ndarray:
    plane = to_float32_2d(array, label="array")
    uint8_max = float(np.iinfo(np.uint8).max)
    return np.clip(np.rint(plane * uint8_max), 0.0, uint8_max).astype(np.uint8)
