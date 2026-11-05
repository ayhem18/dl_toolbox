from __future__ import annotations

import random

import pytest
import torch

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.patchified_tensor import (
    PatchifiedTensor,
)
from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.spec import (
    PatchLayout,
    PatchifiedTensorSpec,
)
from tests.base.base_support import N_TRIALS


def _tensor_spec(*, layout: PatchLayout, original_channels: int, patch_size: int) -> PatchifiedTensorSpec:
    return PatchifiedTensorSpec(
        layout=layout,
        original_channels=original_channels,
        patch_size=patch_size,
    )


def test_from_tensor_requires_spec() -> None:
    tensor = torch.zeros(1, 3, 4, 4)
    with pytest.raises(TypeError):
        PatchifiedTensor.from_tensor(tensor=tensor)


def test_from_tensor_wraps_claimed_spec() -> None:
    spec = _tensor_spec(
        layout=PatchLayout.SPATIAL, original_channels=3, patch_size=2
    )
    tensor = torch.zeros(2, 3, 6, 4)
    tensor[1, 2, 5, 3] = 7.0
    image = PatchifiedTensor.from_tensor(tensor=tensor, spec=spec)
    assert image.spec == spec
    assert tuple(image.tensor.shape) == (2, 3, 6, 4)
    assert float(image.tensor[1, 2, 5, 3]) == 7.0


def test_spatial_patch_grid_roundtrip(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        batch = rng.randint(1, 3)
        channels = rng.randint(1, 4)
        patch_size = rng.choice((2, 4))
        y_patches = rng.randint(1, 3)
        x_patches = rng.randint(1, 4)
        spatial_spec = _tensor_spec(
            layout=PatchLayout.SPATIAL,
            original_channels=channels,
            patch_size=patch_size,
        )
        grid_spec = _tensor_spec(
            layout=PatchLayout.PATCH_GRID,
            original_channels=channels,
            patch_size=patch_size,
        )
        tensor = torch.randn(
            batch, channels, y_patches * patch_size, x_patches * patch_size
        )
        image = PatchifiedTensor.from_tensor(tensor=tensor, spec=spatial_spec)
        grid = image.convert(spec=grid_spec)
        assert tuple(grid.tensor.shape) == (
            batch,
            channels * patch_size * patch_size,
            y_patches,
            x_patches,
        )
        back = grid.convert(spec=spatial_spec)
        assert torch.equal(back.tensor, tensor.contiguous())


def test_spatial_sequence_unflatten_roundtrip(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        batch = rng.randint(1, 3)
        channels = rng.randint(1, 3)
        patch_size = rng.choice((2, 4))
        y_patches = rng.randint(1, 4)
        x_patches = rng.randint(1, 3)
        spatial_spec = _tensor_spec(
            layout=PatchLayout.SPATIAL,
            original_channels=channels,
            patch_size=patch_size,
        )
        sequence_spec = _tensor_spec(
            layout=PatchLayout.SEQUENCE,
            original_channels=channels,
            patch_size=patch_size,
        )
        grid_spec = _tensor_spec(
            layout=PatchLayout.PATCH_GRID,
            original_channels=channels,
            patch_size=patch_size,
        )
        tensor = torch.randn(
            batch, channels, y_patches * patch_size, x_patches * patch_size
        )
        image = PatchifiedTensor.from_tensor(tensor=tensor, spec=spatial_spec)
        sequence = image.convert(spec=sequence_spec)
        assert tuple(sequence.tensor.shape) == (
            batch,
            y_patches * x_patches,
            channels * patch_size * patch_size,
        )
        with pytest.raises(ValueError, match="not unique"):
            sequence.convert(spec=spatial_spec)
        with pytest.raises(ValueError, match="not unique"):
            sequence.convert(spec=grid_spec)
        grid = sequence.unflatten(Y_patches=y_patches, X_patches=x_patches)
        assert grid.spec == grid_spec
        back = grid.convert(spec=spatial_spec)
        assert torch.equal(back.tensor, tensor.contiguous())


def test_unflatten_rejects_factor_that_does_not_match_length() -> None:
    spec = _tensor_spec(
        layout=PatchLayout.SEQUENCE, original_channels=1, patch_size=2
    )
    tensor = torch.zeros(1, 12, 4)
    sequence = PatchifiedTensor.from_tensor(tensor=tensor, spec=spec)
    with pytest.raises(ValueError, match="does not match SEQUENCE length"):
        sequence.unflatten(Y_patches=2, X_patches=2)


def test_pixel_lands_in_expected_patch_vector_slot(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        channels = rng.randint(1, 3)
        patch_size = rng.choice((2, 4))
        y_patches = rng.randint(1, 3)
        x_patches = rng.randint(1, 3)
        y_dim = y_patches * patch_size
        x_dim = x_patches * patch_size
        channel = rng.randint(0, channels - 1)
        y_index = rng.randint(0, y_dim - 1)
        x_index = rng.randint(0, x_dim - 1)
        tensor = torch.zeros(1, channels, y_dim, x_dim)
        tensor[0, channel, y_index, x_index] = 1.0
        image = PatchifiedTensor.from_tensor(
            tensor=tensor,
            spec=_tensor_spec(
                layout=PatchLayout.SPATIAL,
                original_channels=channels,
                patch_size=patch_size,
            ),
        )
        grid = image.convert(
            spec=_tensor_spec(
                layout=PatchLayout.PATCH_GRID,
                original_channels=channels,
                patch_size=patch_size,
            )
        )
        y_patch = y_index // patch_size
        x_patch = x_index // patch_size
        y_off = y_index % patch_size
        x_off = x_index % patch_size
        slot = (channel * patch_size + y_off) * patch_size + x_off
        assert float(grid.tensor[0, slot, y_patch, x_patch]) == 1.0
        assert float(grid.tensor.sum()) == 1.0


def test_convert_rejects_channel_or_patch_size_change() -> None:
    spec = _tensor_spec(
        layout=PatchLayout.SPATIAL, original_channels=3, patch_size=2
    )
    image = PatchifiedTensor.from_tensor(
        tensor=torch.zeros(1, 3, 4, 4), spec=spec
    )
    with pytest.raises(ValueError, match="original_channels"):
        image.convert(
            spec=_tensor_spec(
                layout=PatchLayout.SPATIAL, original_channels=1, patch_size=2
            )
        )
    with pytest.raises(ValueError, match="patch_size"):
        image.convert(
            spec=_tensor_spec(
                layout=PatchLayout.SPATIAL, original_channels=3, patch_size=4
            )
        )
