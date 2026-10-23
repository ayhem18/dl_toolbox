"""Axis-aligned extrema of contour vertices."""

from __future__ import annotations

import numpy as np


def contours_bbox(
    contours: list,
) -> tuple[float, float, float, float] | None:
    if not contours:
        return None
    xs: list[float] = []
    ys: list[float] = []
    n_contours = len(contours)
    for i in range(n_contours):
        points = np.asarray(contours[i], dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            continue
        xs.extend(points[:, 0].tolist())
        ys.extend(points[:, 1].tolist())
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def contour_aabb_wh(contour: object) -> tuple[float, float] | None:
    bbox = contours_bbox([contour])
    if bbox is None:
        return None
    x_min, x_max, y_min, y_max = bbox
    return float(x_max - x_min), float(y_max - y_min)
