"""Image-space boxes: clipped contour AABB and overlapping tiles."""

from __future__ import annotations

import numpy as np

from dl_toolbox.base.geometry.contours import contours_bbox


def contours_bbox_xyxy_int(
    contours: list,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    bbox = contours_bbox(contours)
    if bbox is None:
        return None
    x_min, x_max, y_min, y_max = bbox
    x0 = max(0, int(np.floor(x_min)))
    y0 = max(0, int(np.floor(y_min)))
    x1 = min(int(image_width), int(np.ceil(x_max)))
    y1 = min(int(image_height), int(np.ceil(y_max)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def compute_stride(crop_size: int, overlap_fraction: float) -> int:
    overlap_fraction = min(max(float(overlap_fraction), 0.0), 0.95)
    stride = int(round(crop_size * (1.0 - overlap_fraction)))
    return max(1, stride)


def tile_origins_1d(length_px: int, crop_size: int, stride: int) -> np.ndarray:
    length = int(length_px)
    crop = int(crop_size)
    step = int(stride)
    if crop < 1:
        raise ValueError(f"crop_size must be >= 1, got {crop}")
    if step < 1:
        raise ValueError(f"stride must be >= 1, got {step}")
    if length < 1:
        raise ValueError(f"length_px must be >= 1, got {length}")
    if crop > length:
        raise ValueError(f"crop_size {crop} is larger than the frame axis {length}")
    if length == crop:
        return np.array([0], dtype=np.int32)
    origins = np.arange(0, length - crop + 1, step, dtype=np.int32)
    last_origin = length - crop
    if origins[-1] != last_origin:
        origins = np.concatenate([origins, np.array([last_origin], dtype=np.int32)])
    return origins


def generate_tile_boxes_xyxy(
    width: int,
    height: int,
    crop_size: int,
    overlap_fraction: float,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        return np.empty((0, 4), dtype=np.int32)
    side = int(crop_size)
    if side > int(width) or side > int(height):
        raise ValueError(
            f"crop {(side, side)} is larger than the frame {int(width)}x{int(height)}"
        )
    stride = compute_stride(side, overlap_fraction)
    x0 = tile_origins_1d(width, side, stride)
    y0 = tile_origins_1d(height, side, stride)
    yy, xx = np.meshgrid(y0, x0, indexing="ij")
    return np.stack([xx, yy, xx + side, yy + side], axis=-1).reshape(-1, 4).astype(
        np.int32
    )
