from __future__ import annotations

import random

import pytest
import torch

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.patchified_mask import (
    PatchifiedMask,
)
from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.spec import (
    MaskPolarity,
    PatchLayout,
    PatchifiedMaskSpec,
)
from tests.base.base_support import N_TRIALS


def _mask_spec(
    *, layout: PatchLayout, patch_size: int, polarity: MaskPolarity
) -> PatchifiedMaskSpec:
    return PatchifiedMaskSpec(
        layout=layout, patch_size=patch_size, polarity=polarity
    )


def _spatial_from_grid(*, grid: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, y_patches, x_patches = grid.shape
    spatial = torch.empty(
        batch, y_patches * patch_size, x_patches * patch_size, dtype=grid.dtype
    )
    y_index = 0
    while y_index < y_patches:
        x_index = 0
        y_start = y_index * patch_size
        y_end = y_start + patch_size
        while x_index < x_patches:
            x_start = x_index * patch_size
            x_end = x_start + patch_size
            spatial[:, y_start:y_end, x_start:x_end] = grid[
                :, y_index : y_index + 1, x_index : x_index + 1
            ]
            x_index += 1
        y_index += 1
    return spatial


def test_from_tensor_requires_spec() -> None:
    tensor = torch.zeros(1, 4, 4)
    with pytest.raises(TypeError):
        PatchifiedMask.from_tensor(tensor=tensor)


def test_spatial_parse_squeezes_singleton_channel() -> None:
    spec = _mask_spec(
        layout=PatchLayout.SPATIAL,
        patch_size=2,
        polarity=MaskPolarity.REMOVED_IS_ONE,
    )
    grid = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    spatial = _spatial_from_grid(grid=grid, patch_size=2)
    nchw = spatial.unsqueeze(1)
    mask = PatchifiedMask.from_tensor(tensor=nchw, spec=spec)
    assert mask.tensor.ndim == 3
    assert torch.equal(mask.tensor, spatial.contiguous())


def test_spatial_patch_grid_sequence_roundtrip(rng: random.Random) -> None:
    polarities = (
        MaskPolarity.REMOVED_IS_ONE,
        MaskPolarity.VISIBLE_IS_ONE,
    )
    for _ in range(N_TRIALS):
        batch = rng.randint(1, 3)
        patch_size = rng.choice((2, 4))
        y_patches = rng.randint(1, 3)
        x_patches = rng.randint(1, 4)
        polarity = polarities[rng.randint(0, 1)]
        grid = torch.randint(0, 2, (batch, y_patches, x_patches)).float()
        spatial = _spatial_from_grid(grid=grid, patch_size=patch_size)
        spatial_spec = _mask_spec(
            layout=PatchLayout.SPATIAL, patch_size=patch_size, polarity=polarity
        )
        grid_spec = _mask_spec(
            layout=PatchLayout.PATCH_GRID, patch_size=patch_size, polarity=polarity
        )
        sequence_spec = _mask_spec(
            layout=PatchLayout.SEQUENCE, patch_size=patch_size, polarity=polarity
        )
        mask = PatchifiedMask.from_tensor(tensor=spatial, spec=spatial_spec)
        as_grid = mask.convert(spec=grid_spec)
        assert torch.equal(as_grid.tensor, grid.contiguous())
        back = as_grid.convert(spec=spatial_spec)
        assert torch.equal(back.tensor, spatial.contiguous())
        sequence = mask.convert(spec=sequence_spec)
        assert tuple(sequence.tensor.shape) == (batch, y_patches * x_patches)
        with pytest.raises(ValueError, match="not unique"):
            sequence.convert(spec=grid_spec)
        restored = sequence.unflatten(
            Y_patches=y_patches, X_patches=x_patches
        )
        assert torch.equal(restored.tensor, grid.contiguous())


def test_polarity_flip_inverts_values() -> None:
    spec = _mask_spec(
        layout=PatchLayout.PATCH_GRID,
        patch_size=2,
        polarity=MaskPolarity.REMOVED_IS_ONE,
    )
    flipped_spec = _mask_spec(
        layout=PatchLayout.PATCH_GRID,
        patch_size=2,
        polarity=MaskPolarity.VISIBLE_IS_ONE,
    )
    grid = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = PatchifiedMask.from_tensor(tensor=grid, spec=spec)
    flipped = mask.convert(spec=flipped_spec)
    assert torch.equal(flipped.tensor, 1.0 - grid)
    back = flipped.convert(spec=spec)
    assert torch.equal(back.tensor, grid.contiguous())


def test_not_patch_constant_spatial_is_rejected() -> None:
    spec = _mask_spec(
        layout=PatchLayout.SPATIAL,
        patch_size=2,
        polarity=MaskPolarity.REMOVED_IS_ONE,
    )
    tensor = torch.zeros(1, 4, 4)
    tensor[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="not constant"):
        PatchifiedMask.from_tensor(tensor=tensor, spec=spec)


def test_unflatten_rejects_non_sequence() -> None:
    spec = _mask_spec(
        layout=PatchLayout.PATCH_GRID,
        patch_size=2,
        polarity=MaskPolarity.REMOVED_IS_ONE,
    )
    mask = PatchifiedMask.from_tensor(
        tensor=torch.zeros(1, 2, 2), spec=spec
    )
    with pytest.raises(ValueError, match="unflatten requires SEQUENCE"):
        mask.unflatten(Y_patches=2, X_patches=2)
