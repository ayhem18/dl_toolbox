"""Scharr edge strength on a 2-D grayscale plane.

OpenCV's 3x3 Scharr maps an axis-aligned intensity step of height Δ to a
kernel response of 16·Δ. ``SCHARR_STEP_SCALE`` is that kernel gain. Magnitude
returned by ``scharr_edge_strength`` is already divided by it, so values are
gray-level units, not kernel units.

This module does not convert color, does not bilateral-filter, and does not
treat a 3-D stack as OpenCV channels. Display scaling lives in
``basic_processing`` and is called with required arguments from here.
"""

from __future__ import annotations

import cv2
import numpy as np

from dl_toolbox.base.image_processing.basic_processing import (
    normalize_absolute,
    normalize_by_robust_percentiles,
    require_2d_array,
    to_float32_2d,
)

SCHARR_STEP_SCALE = 16.0


def scharr_derivatives(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-unit Scharr ``gx``, ``gy``. 2-D only."""
    plane = to_float32_2d(image, label="image")
    gx = cv2.Scharr(plane, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(plane, cv2.CV_32F, 0, 1)
    return gx, gy


def to_gray_level_derivatives(
    gx: np.ndarray, gy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Undo ``SCHARR_STEP_SCALE`` so derivatives are in intensity units."""
    gx_kernel = require_2d_array(gx, label="gx")
    gy_kernel = require_2d_array(gy, label="gy")
    if gx_kernel.shape != gy_kernel.shape:
        raise ValueError(
            f"gx and gy must have the same shape, got {gx_kernel.shape} and "
            f"{gy_kernel.shape}"
        )
    scale = np.float32(SCHARR_STEP_SCALE)
    return gx_kernel / scale, gy_kernel / scale


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    gx_plane = require_2d_array(gx, label="gx")
    gy_plane = require_2d_array(gy, label="gy")
    if gx_plane.shape != gy_plane.shape:
        raise ValueError(
            f"gx and gy must have the same shape, got {gx_plane.shape} and "
            f"{gy_plane.shape}"
        )
    return cv2.magnitude(
        np.ascontiguousarray(gx_plane, dtype=np.float32),
        np.ascontiguousarray(gy_plane, dtype=np.float32),
    )


def gradient_orientation_deg(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    gx_plane = require_2d_array(gx, label="gx")
    gy_plane = require_2d_array(gy, label="gy")
    if gx_plane.shape != gy_plane.shape:
        raise ValueError(
            f"gx and gy must have the same shape, got {gx_plane.shape} and "
            f"{gy_plane.shape}"
        )
    return cv2.phase(
        np.ascontiguousarray(gx_plane, dtype=np.float32),
        np.ascontiguousarray(gy_plane, dtype=np.float32),
        angleInDegrees=True,
    )


def scharr_edge_strength(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gray-level Scharr magnitude and orientation in degrees. 2-D only."""
    gx_kernel, gy_kernel = scharr_derivatives(image)
    gx_gray, gy_gray = to_gray_level_derivatives(gx_kernel, gy_kernel)
    magnitude = gradient_magnitude(gx_gray, gy_gray)
    theta_deg = gradient_orientation_deg(gx_kernel, gy_kernel)
    return magnitude, theta_deg


def absolute_normalized_scharr_magnitude(
    image: np.ndarray, *, source_dtype: object
) -> np.ndarray:
    """Gray-level Scharr magnitude divided by the dtype Euclidean ceiling."""
    magnitude, _theta_deg = scharr_edge_strength(image)
    return normalize_absolute(magnitude, source_dtype=source_dtype)


def robust_percentile_normalized_scharr_magnitude(
    image: np.ndarray,
    *,
    robust_low_percentile: float,
    robust_high_percentile: float,
) -> np.ndarray:
    """Gray-level Scharr magnitude, then a robust-percentile stretch.

    ``robust_low_percentile`` and ``robust_high_percentile`` are required.
    They are applied to this frame's magnitude; they are not a hidden default.
    """
    magnitude, _theta_deg = scharr_edge_strength(image)
    return normalize_by_robust_percentiles(
        magnitude,
        robust_low_percentile=robust_low_percentile,
        robust_high_percentile=robust_high_percentile,
    )


def non_maximum_suppression(
    magnitude: np.ndarray, theta_deg: np.ndarray
) -> np.ndarray:
    """Four-sector NMS on a 2-D magnitude / orientation pair."""
    strength = require_2d_array(magnitude, label="magnitude")
    orientation = require_2d_array(theta_deg, label="theta_deg")
    if strength.shape != orientation.shape:
        raise ValueError(
            "magnitude and theta_deg must have the same shape, got "
            f"{strength.shape} and {orientation.shape}"
        )
    angle = np.mod(orientation, 180.0)
    padded = np.pad(strength, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]

    sector_0 = (angle < 22.5) | (angle >= 157.5)
    sector_45 = (angle >= 22.5) & (angle < 67.5)
    sector_90 = (angle >= 67.5) & (angle < 112.5)
    sector_135 = (angle >= 112.5) & (angle < 157.5)

    keep = np.zeros(strength.shape, dtype=bool)
    keep |= sector_0 & (center >= left) & (center >= right)
    keep |= sector_45 & (center >= up_right) & (center >= down_left)
    keep |= sector_90 & (center >= up) & (center >= down)
    keep |= sector_135 & (center >= up_left) & (center >= down_right)

    out = np.zeros(strength.shape, dtype=np.float32)
    out[keep] = np.asarray(strength, dtype=np.float32)[keep]
    return out
