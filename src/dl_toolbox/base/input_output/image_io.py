"""Untyped OpenCV leftovers. Typed photometric IO is ``images.Image`` and ``images.OpenCVImageIO``."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dl_toolbox.base.file_management.paths import abs_path


def read_bgr(path: Path | str) -> np.ndarray:
    source = abs_path(path)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read BGR image: {source}")
    return image


def read_gray(path: Path | str) -> np.ndarray:
    source = abs_path(path)
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"failed to read grayscale image: {source}")
    return image


def read_mask(path: Path | str) -> np.ndarray:
    source = abs_path(path)
    mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {source}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D class-index, got {mask.shape}")
    return np.asarray(mask, dtype=np.uint8)


def write_image(path: Path | str, image: np.ndarray) -> Path:
    destination = abs_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"failed to write image: {destination}")
    return destination


def write_mask(path: Path | str, mask: np.ndarray) -> Path:
    return write_image(path, np.asarray(mask, dtype=np.uint8))
