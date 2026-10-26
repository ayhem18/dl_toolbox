"""Rasterize contours into boolean masks."""

from __future__ import annotations

import cv2
import numpy as np


def rasterize_contour(contour: object, height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        return mask.astype(bool)
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def rasterize_contours(contours: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    n_contours = len(contours)
    for i in range(n_contours):
        mask |= rasterize_contour(contours[i], height, width)
    return mask
