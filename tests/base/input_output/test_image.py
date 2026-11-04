from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dl_toolbox.base.input_output.images.image import Image
from dl_toolbox.base.input_output.images.image_spec import (
    ColorMode,
    ImageSpec,
    Layout,
    SampleType,
)
from dl_toolbox.base.input_output.images.io import OpenCVImageIO
from tests.base.base_support import N_TRIALS


def _uint8_spec(*, color_mode: ColorMode, layout: Layout) -> ImageSpec:
    return ImageSpec(
        color_mode=color_mode, sample_type=SampleType.UINT8, layout=layout
    )


def _bgr_yxc(y_dim: int, x_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(y_dim, x_dim, 3), dtype=np.uint8)


def test_from_array_requires_spec() -> None:
    array = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(TypeError):
        Image.from_array(array=array)


def test_from_array_wraps_claimed_spec() -> None:
    spec = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    array = np.zeros((6, 7, 3), dtype=np.uint8)
    array[1, 2] = (10, 20, 30)
    image = Image.from_array(array=array, spec=spec)
    assert image.spec == spec
    assert image.array.shape == (6, 7, 3)
    assert tuple(image.array[1, 2]) == (10, 20, 30)


def test_opencv_read_assigns_bgr_uint8_yxc(tmp_path: Path) -> None:
    bgr = np.zeros((5, 6, 3), dtype=np.uint8)
    bgr[2, 3] = (255, 128, 0)
    path = tmp_path / "native.png"
    cv2.imwrite(str(path), bgr)
    image = OpenCVImageIO().read(path=path)
    assert image.spec.color_mode == ColorMode.BGR
    assert image.spec.sample_type == SampleType.UINT8
    assert image.spec.layout == Layout.YXC
    assert tuple(image.array[2, 3]) == (255, 128, 0)


def test_opencv_read_assigns_gray_yx(tmp_path: Path) -> None:
    gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), gray)
    image = OpenCVImageIO().read(path=path)
    assert image.spec.color_mode == ColorMode.GRAY
    assert image.array.ndim == 2
    assert image.array.shape == (4, 5)


def test_opencv_read_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        OpenCVImageIO().read(path="relative.png")


def test_bgr_rgb_roundtrip_is_channel_swap() -> None:
    bgr_spec = _uint8_spec(color_mode=ColorMode.BGR, layout=Layout.YXC)
    rgb_spec = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    array[0, 0] = (255, 0, 0)
    image = Image.from_array(array=array, spec=bgr_spec)
    rgb = image.convert(spec=rgb_spec, strict=True)
    assert tuple(rgb.array[0, 0]) == (0, 0, 255)
    back = rgb.convert(spec=bgr_spec, strict=True)
    assert np.array_equal(back.array, array)


def test_yxc_cyx_roundtrip() -> None:
    yxc_spec = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    cyx_spec = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.CYX)
    array = np.zeros((4, 5, 3), dtype=np.uint8)
    array[1, 2] = (1, 2, 3)
    image = Image.from_array(array=array, spec=yxc_spec)
    cyx = image.convert(spec=cyx_spec, strict=True)
    assert cyx.array.shape == (3, 4, 5)
    assert tuple(cyx.array[:, 1, 2]) == (1, 2, 3)
    back = cyx.convert(spec=yxc_spec, strict=True)
    assert np.array_equal(back.array, array)


def test_gray_to_rgb_strict_raises() -> None:
    spec = _uint8_spec(color_mode=ColorMode.GRAY, layout=Layout.YXC)
    target = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    image = Image.from_array(array=np.ones((3, 3), dtype=np.uint8), spec=spec)
    with pytest.raises(ValueError, match="gray to color"):
        image.convert(spec=target, strict=True)


def test_gray_to_rgb_nonstrict_stacks_channels() -> None:
    spec = _uint8_spec(color_mode=ColorMode.GRAY, layout=Layout.YXC)
    target = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    gray = np.array([[7, 8], [9, 10]], dtype=np.uint8)
    image = Image.from_array(array=gray, spec=spec)
    rgb = image.convert(spec=target, strict=False)
    assert rgb.array.shape == (2, 2, 3)
    assert tuple(rgb.array[0, 1]) == (8, 8, 8)


def test_color_to_gray_raises() -> None:
    spec = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    target = _uint8_spec(color_mode=ColorMode.GRAY, layout=Layout.YXC)
    image = Image.from_array(
        array=np.zeros((2, 2, 3), dtype=np.uint8), spec=spec
    )
    with pytest.raises(ValueError, match="color to gray"):
        image.convert(spec=target, strict=False)


def test_uint8_float32_unit_roundtrip() -> None:
    uint8_spec = _uint8_spec(color_mode=ColorMode.GRAY, layout=Layout.YXC)
    float_spec = ImageSpec(
        color_mode=ColorMode.GRAY,
        sample_type=SampleType.FLOAT32_UNIT,
        layout=Layout.YXC,
    )
    array = np.array([[0, 127, 255], [1, 64, 192]], dtype=np.uint8)
    image = Image.from_array(array=array, spec=uint8_spec)
    unit = image.convert(spec=float_spec, strict=True)
    assert unit.array.dtype == np.float32
    assert float(unit.array[0, 2]) == pytest.approx(1.0)
    back = unit.convert(spec=uint8_spec, strict=True)
    assert np.array_equal(back.array, array)


def test_opencv_write_png_roundtrips_bgr(tmp_path: Path) -> None:
    spec = _uint8_spec(color_mode=ColorMode.BGR, layout=Layout.YXC)
    array = np.zeros((8, 9, 3), dtype=np.uint8)
    array[4, 5] = (9, 10, 11)
    image = Image.from_array(array=array, spec=spec)
    path = tmp_path / "out" / "round.png"
    io = OpenCVImageIO()
    written = io.write(image=image, path=path)
    loaded = io.read(path=written)
    assert loaded.spec.color_mode == ColorMode.BGR
    assert np.array_equal(loaded.array, array)


def test_opencv_write_converts_rgb_float_to_native(tmp_path: Path) -> None:
    spec = ImageSpec(
        color_mode=ColorMode.RGB,
        sample_type=SampleType.FLOAT32_UNIT,
        layout=Layout.YXC,
    )
    array = np.zeros((3, 3, 3), dtype=np.float32)
    array[1, 1] = (1.0, 0.0, 0.0)
    image = Image.from_array(array=array, spec=spec)
    io = OpenCVImageIO()
    path = tmp_path / "rgb.png"
    io.write(image=image, path=path)
    loaded = io.read(path=path)
    assert loaded.spec.color_mode == ColorMode.BGR
    assert loaded.spec.sample_type == SampleType.UINT8
    assert tuple(loaded.array[1, 1]) == (0, 0, 255)


def test_opencv_write_rejects_jpeg(tmp_path: Path) -> None:
    spec = _uint8_spec(color_mode=ColorMode.BGR, layout=Layout.YXC)
    image = Image.from_array(
        array=np.zeros((2, 2, 3), dtype=np.uint8), spec=spec
    )
    with pytest.raises(ValueError, match="lossy codec"):
        OpenCVImageIO().write(image=image, path=tmp_path / "x.jpg")


def test_randomized_bgr_rgb_and_layout_roundtrips() -> None:
    rng = np.random.default_rng(42)
    bgr_yxc = _uint8_spec(color_mode=ColorMode.BGR, layout=Layout.YXC)
    rgb_yxc = _uint8_spec(color_mode=ColorMode.RGB, layout=Layout.YXC)
    bgr_cyx = _uint8_spec(color_mode=ColorMode.BGR, layout=Layout.CYX)
    for _ in range(N_TRIALS):
        y_dim = int(rng.integers(2, 8))
        x_dim = int(rng.integers(2, 8))
        array = _bgr_yxc(y_dim, x_dim, rng)
        image = Image.from_array(array=array, spec=bgr_yxc)
        rgb = image.convert(spec=rgb_yxc, strict=True)
        back_color = rgb.convert(spec=bgr_yxc, strict=True)
        assert np.array_equal(back_color.array, array)
        cyx = image.convert(spec=bgr_cyx, strict=True)
        assert cyx.array.shape == (3, y_dim, x_dim)
        back_layout = cyx.convert(spec=bgr_yxc, strict=True)
        assert np.array_equal(back_layout.array, array)
