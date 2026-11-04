from __future__ import annotations

import numpy as np
import pytest

from dl_toolbox.base.input_output.images.image_spec import (
    ColorMode,
    ImageSpec,
    Layout,
    SampleType,
    validate_array,
)


def _spec(*, color_mode: ColorMode, sample_type: SampleType, layout: Layout) -> ImageSpec:
    return ImageSpec(
        color_mode=color_mode, sample_type=sample_type, layout=layout
    )


def test_uint8_sample_type_rejects_unit_range() -> None:
    with pytest.raises(ValueError, match="uint8 sample type must use range"):
        SampleType(name="bad", dtype=np.uint8, value_min=0.0, value_max=1.0)


def test_float32_sample_type_rejects_byte_range() -> None:
    with pytest.raises(ValueError, match="float32 sample type must use range"):
        SampleType(
            name="bad", dtype=np.float32, value_min=0.0, value_max=255.0
        )


def test_validate_array_accepts_matching_gray_uint8() -> None:
    spec = _spec(
        color_mode=ColorMode.GRAY,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert validate_array(array, spec=spec) is array


def test_validate_array_rejects_gray_with_channel_axis() -> None:
    spec = _spec(
        color_mode=ColorMode.GRAY,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    array = np.zeros((3, 4, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(Y_dim, X_dim\)"):
        validate_array(array, spec=spec)


def test_validate_array_rejects_yxc_claim_on_cyx_shape() -> None:
    spec = _spec(
        color_mode=ColorMode.RGB,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    array = np.zeros((3, 8, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(Y_dim, X_dim, 3\)"):
        validate_array(array, spec=spec)


def test_validate_array_rejects_float_outside_unit_interval() -> None:
    spec = _spec(
        color_mode=ColorMode.GRAY,
        sample_type=SampleType.FLOAT32_UNIT,
        layout=Layout.YXC,
    )
    array = np.array([[0.0, 1.5], [0.2, 0.3]], dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        validate_array(array, spec=spec)


def test_validate_array_rejects_uint8_claimed_as_float32() -> None:
    spec = _spec(
        color_mode=ColorMode.GRAY,
        sample_type=SampleType.FLOAT32_UNIT,
        layout=Layout.YXC,
    )
    array = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="dtype"):
        validate_array(array, spec=spec)


def test_specs_with_same_parts_are_equal() -> None:
    left = _spec(
        color_mode=ColorMode.BGR,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    right = _spec(
        color_mode=ColorMode.BGR,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    assert left == right
    assert hash(left) == hash(right)
    rgb = _spec(
        color_mode=ColorMode.RGB,
        sample_type=SampleType.UINT8,
        layout=Layout.YXC,
    )
    assert left != rgb
