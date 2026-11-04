"""``Image`` is ``ImageSpec`` plus a numpy array.

Ndarray source: caller must pass ``spec``. File source is ``ImageIO.read``.
"""

from __future__ import annotations

import numpy as np

from dl_toolbox.base.input_output.images.image_spec import (
    ColorMode,
    ImageSpec,
    Layout,
    SampleType,
    validate_array,
)


class Image:
    """Photometric raster: ``spec`` + ``array``. Construction validates."""

    def __init__(self, *, spec: ImageSpec, array: np.ndarray) -> None:
        validate_array(array, spec=spec)
        self._spec = spec
        self._array = array

    @property
    def spec(self) -> ImageSpec:
        return self._spec

    @property
    def array(self) -> np.ndarray:
        return self._array

    @classmethod
    def from_array(cls, *, array: np.ndarray, spec: ImageSpec) -> Image:
        """Ndarray source. ``spec`` is required. Makes the buffer C-contiguous."""
        return cls(spec=spec, array=np.ascontiguousarray(array))

    def convert(self, *, spec: ImageSpec, strict: bool) -> Image:
        """Map this image onto ``spec``.

        ``strict`` is required. ``bgr`` <-> ``rgb`` is always defined.
        ``gray`` -> color stacks the plane when ``strict`` is False and
        raises when ``strict`` is True. Color -> gray raises until a
        reduction is named.
        """
        if spec == self._spec:
            return Image(spec=spec, array=self._array)
        working = self._to_yxc()
        working = self._convert_color_yxc(
            array_yxc=working,
            target=spec.color_mode,
            strict=strict,
        )
        working = self._convert_sample(
            array=working,
            target=spec.sample_type,
        )
        working = self._from_yxc(array_yxc=working, spec=spec)
        return Image(spec=spec, array=np.ascontiguousarray(working))

    def _to_yxc(self) -> np.ndarray:
        if self._spec.color_mode == ColorMode.GRAY:
            return self._array
        if self._spec.layout == Layout.YXC:
            return self._array
        return np.transpose(self._array, (1, 2, 0))

    def _from_yxc(self, *, array_yxc: np.ndarray, spec: ImageSpec) -> np.ndarray:
        if spec.color_mode == ColorMode.GRAY:
            return array_yxc
        if spec.layout == Layout.YXC:
            return array_yxc
        return np.transpose(array_yxc, (2, 0, 1))

    def _convert_color_yxc(
        self,
        *,
        array_yxc: np.ndarray,
        target: ColorMode,
        strict: bool,
    ) -> np.ndarray:
        source = self._spec.color_mode
        if source == target:
            return array_yxc
        pair = {source, target}
        if pair == {ColorMode.BGR, ColorMode.RGB}:
            return np.ascontiguousarray(array_yxc[..., ::-1])
        if source == ColorMode.GRAY and target in (ColorMode.BGR, ColorMode.RGB):
            if strict:
                raise ValueError(
                    "gray to color has no color map; pass strict=False to stack "
                    "the plane onto 3 channels"
                )
            return np.stack((array_yxc, array_yxc, array_yxc), axis=-1)
        if target == ColorMode.GRAY:
            raise ValueError(
                "color to gray is not a unique map (luma vs mean vs one channel)"
            )
        raise ValueError(f"no color conversion from {source!r} to {target!r}")

    def _convert_sample(
        self, *, array: np.ndarray, target: SampleType
    ) -> np.ndarray:
        source = self._spec.sample_type
        if source == target:
            return array
        if source == SampleType.UINT8 and target == SampleType.FLOAT32_UNIT:
            return np.ascontiguousarray(array.astype(np.float32) / np.float32(255.0))
        if source == SampleType.FLOAT32_UNIT and target == SampleType.UINT8:
            scaled = np.rint(array * np.float32(255.0))
            return np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        raise ValueError(f"no sample conversion from {source!r} to {target!r}")
