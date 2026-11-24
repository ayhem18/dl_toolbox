"""Convert tensors / arrays to a 2D grayscale plane."""

from __future__ import annotations

import numpy as np


def as_gray2d(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0]
        if array.shape[-1] == 1:
            return array[..., 0]
        raise ValueError(
            f"grayscale image must be HW, 1HW, or HW1, got {tuple(array.shape)}"
        )
    raise ValueError(
        f"grayscale image must be 2D or 3D, got ndim={array.ndim} shape={tuple(array.shape)}"
    )
