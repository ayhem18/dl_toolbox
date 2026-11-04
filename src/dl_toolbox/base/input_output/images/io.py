"""File read/write for ``Image``. The backend owns native spec and codec rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from dl_toolbox.base.file_management.paths import abs_path
from dl_toolbox.base.input_output.images.image import Image
from dl_toolbox.base.input_output.images.image_spec import (
    ColorMode,
    ImageSpec,
    Layout,
    SampleType,
    validate_array,
)

_LOSSY_SUFFIXES = {".jpg", ".jpeg"}
_WRITABLE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


class ImageIO(ABC):
    """Path <-> ``Image``. Spec assignment and encode constraints belong here."""

    @abstractmethod
    def read(self, *, path: Path | str) -> Image:
        """File source. The backend assigns the spec; the caller does not pass one."""

    @abstractmethod
    def write(self, *, image: Image, path: Path | str) -> Path:
        """Write ``image`` at ``path``. The backend converts to its native spec."""


class OpenCVImageIO(ImageIO):
    """OpenCV file backend.

    Native spec is gray ``uint8`` ``(Y_dim, X_dim)`` or BGR ``uint8`` YXC.
    ``.png`` / ``.bmp`` / ``.tif`` / ``.tiff`` are allowed. ``.jpg`` is
    lossy (``jpg -> array -> jpg`` compresses again).
    """

    GRAY_NATIVE = ImageSpec(
        color_mode=ColorMode.GRAY,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    BGR_NATIVE = ImageSpec(
        color_mode=ColorMode.BGR,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )

    def read(self, *, path: Path | str) -> Image:
        source = abs_path(path)
        if not source.is_file():
            raise FileNotFoundError(f"image file does not exist: {source}")
        decoded = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"failed to decode image: {source}")
        array = np.ascontiguousarray(decoded)
        spec = self._spec_from_decoded_array(array=array)
        validate_array(array, spec=spec)
        return Image(spec=spec, array=array)

    def write(self, *, image: Image, path: Path | str) -> Path:
        destination = abs_path(path)
        suffix = destination.suffix.lower()
        if suffix in _LOSSY_SUFFIXES:
            raise ValueError(
                f"lossy codec {suffix!r} is not a value-preserving file write, "
                f"got {destination}"
            )
        if suffix not in _WRITABLE_SUFFIXES:
            raise ValueError(
                f"unsupported image suffix {suffix!r} for file write, "
                f"got {destination}"
            )
        native_spec = self._native_spec(color_mode=image.spec.color_mode)
        native = image.convert(spec=native_spec, strict=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), native.array):
            raise OSError(f"failed to write image: {destination}")
        return destination

    def _native_spec(self, *, color_mode: ColorMode) -> ImageSpec:
        if color_mode == ColorMode.GRAY:
            return self.GRAY_NATIVE
        return self.BGR_NATIVE

    def _spec_from_decoded_array(self, *, array: np.ndarray) -> ImageSpec:
        if array.dtype != np.dtype(np.uint8):
            raise ValueError(
                "OpenCV file source v1 is uint8 "
                f"[0, 255], got dtype {array.dtype}"
            )
        if array.ndim == 2:
            return self.GRAY_NATIVE
        if array.ndim == 3 and array.shape[2] == 3:
            return self.BGR_NATIVE
        raise ValueError(
            "OpenCV file source v1 is gray (Y_dim, X_dim) or BGR YXC "
            f"(Y_dim, X_dim, 3), got shape {array.shape}"
        )
