"""Photometric spec: color mode, sample type, layout.

``Image`` is this spec plus a numpy array. Spatial size (``Y_dim``, ``X_dim``)
lives on the array.
"""

from __future__ import annotations

import numpy as np


class ColorMode:
    """Semantic meaning of the channel axis. Channel count is derived."""

    def __init__(self, *, name: str, n_channels: int) -> None:
        if n_channels not in (0, 3):
            raise ValueError(f"n_channels must be 0 (gray) or 3 (color), got {n_channels}")
        self._name = name
        self._n_channels = n_channels

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def rank(self) -> int:
        if self._n_channels == 0:
            return 2
        return 3

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColorMode) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"ColorMode.{self._name.upper()}"


ColorMode.GRAY = ColorMode(name="gray", n_channels=0)
ColorMode.BGR = ColorMode(name="bgr", n_channels=3)
ColorMode.RGB = ColorMode(name="rgb", n_channels=3)


class SampleType:
    """Dtype and closed value range as one part."""

    def __init__(
        self,
        *,
        name: str,
        dtype: object,
        value_min: float,
        value_max: float,
    ) -> None:
        parsed_dtype = np.dtype(dtype)
        if parsed_dtype == np.dtype(np.uint8):
            if value_min != 0.0 or value_max != 255.0:
                raise ValueError("uint8 sample type must use range [0, 255]")
        elif parsed_dtype == np.dtype(np.float32):
            if value_min != 0.0 or value_max != 1.0:
                raise ValueError("float32 sample type must use range [0, 1]")
        else:
            raise ValueError(f"unsupported sample dtype {parsed_dtype}")
        if not value_min < value_max:
            raise ValueError(
                f"value_min must be < value_max, got {value_min} and {value_max}"
            )
        self._name = name
        self._dtype = parsed_dtype
        self._value_min = float(value_min)
        self._value_max = float(value_max)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def value_min(self) -> float:
        return self._value_min

    @property
    def value_max(self) -> float:
        return self._value_max

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SampleType) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"SampleType.{self._name.upper()}"


SampleType.UINT8 = SampleType(
    name="uint8", dtype=np.uint8, value_min=0.0, value_max=255.0
)
SampleType.FLOAT32_UNIT = SampleType(
    name="float32_unit", dtype=np.float32, value_min=0.0, value_max=1.0
)


class Layout:
    """Channel placement for rank-3 arrays: YXC ``(Y_dim, X_dim, C)`` or CYX ``(C, Y_dim, X_dim)``.

    Gray is ``(Y_dim, X_dim)`` under both layouts.
    """

    def __init__(self, *, name: str) -> None:
        if name not in ("yxc", "cyx"):
            raise ValueError(f"layout name must be 'yxc' or 'cyx', got {name!r}")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Layout) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Layout.{self._name.upper()}"


Layout.YXC = Layout(name="yxc")
Layout.CYX = Layout(name="cyx")


class ImageSpec:
    """Declared photometric state. Orthogonal parts: color mode, sample type, layout."""

    def __init__(
        self,
        *,
        color_mode: ColorMode,
        sample_type: SampleType,
        layout: Layout,
    ) -> None:
        if not isinstance(color_mode, ColorMode):
            raise TypeError(f"color_mode must be ColorMode, got {type(color_mode)!r}")
        if not isinstance(sample_type, SampleType):
            raise TypeError(f"sample_type must be SampleType, got {type(sample_type)!r}")
        if not isinstance(layout, Layout):
            raise TypeError(f"layout must be Layout, got {type(layout)!r}")
        self._color_mode = color_mode
        self._sample_type = sample_type
        self._layout = layout

    @property
    def color_mode(self) -> ColorMode:
        return self._color_mode

    @property
    def sample_type(self) -> SampleType:
        return self._sample_type

    @property
    def layout(self) -> Layout:
        return self._layout

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ImageSpec)
            and self._color_mode == other._color_mode
            and self._sample_type == other._sample_type
            and self._layout == other._layout
        )

    def __hash__(self) -> int:
        return hash((self._color_mode, self._sample_type, self._layout))

    def __repr__(self) -> str:
        return (
            "ImageSpec("
            f"color_mode={self._color_mode!r}, "
            f"sample_type={self._sample_type!r}, "
            f"layout={self._layout!r})"
        )


def expected_shape_suffix(*, spec: ImageSpec) -> str:
    if spec.color_mode.n_channels == 0:
        return "(Y_dim, X_dim)"
    n_channels = spec.color_mode.n_channels
    if spec.layout == Layout.YXC:
        return f"(Y_dim, X_dim, {n_channels})"
    return f"({n_channels}, Y_dim, X_dim)"


def validate_array(array: object, *, spec: ImageSpec) -> np.ndarray:
    """Raise unless ``array`` is already ``spec``. Color mode is a claim, not detected."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a numpy.ndarray, got {type(array)!r}")
    if array.size == 0:
        raise ValueError("array is empty")
    if not array.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")
    if spec.color_mode.n_channels == 0:
        if array.ndim != 2:
            raise ValueError(
                f"gray spec requires shape (Y_dim, X_dim), got {array.shape}"
            )
    else:
        n_channels = spec.color_mode.n_channels
        if spec.layout == Layout.YXC:
            legal = array.ndim == 3 and array.shape[2] == n_channels
        else:
            legal = array.ndim == 3 and array.shape[0] == n_channels
        if not legal:
            raise ValueError(
                f"{spec.color_mode!r} {spec.layout!r} requires shape "
                f"{expected_shape_suffix(spec=spec)}, got {array.shape}"
            )
    if array.dtype != spec.sample_type.dtype:
        raise ValueError(
            f"array dtype {array.dtype} does not match {spec.sample_type!r} "
            f"({spec.sample_type.dtype})"
        )
    if not np.isfinite(array).all():
        raise ValueError("array contains NaN or inf")
    value_min = float(np.min(array))
    value_max = float(np.max(array))
    low = spec.sample_type.value_min
    high = spec.sample_type.value_max
    if value_min < low or value_max > high:
        raise ValueError(
            f"array values [{value_min}, {value_max}] outside "
            f"{spec.sample_type!r} [{low}, {high}]"
        )
    return array
