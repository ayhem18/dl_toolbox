from __future__ import annotations

import pytest
import torch

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.spec import (
    MaskPolarity,
    PatchLayout,
    PatchifiedMaskSpec,
    PatchifiedTensorSpec,
    validate_patchified_mask,
    validate_patchified_tensor,
)


def test_patch_size_must_be_greater_than_one() -> None:
    with pytest.raises(ValueError, match="patch_size must be > 1"):
        PatchifiedTensorSpec(
            layout=PatchLayout.SPATIAL, original_channels=3, patch_size=1
        )
    with pytest.raises(ValueError, match="patch_size must be > 1"):
        PatchifiedMaskSpec(
            layout=PatchLayout.SPATIAL,
            patch_size=1,
            polarity=MaskPolarity.REMOVED_IS_ONE,
        )


def test_original_channels_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="original_channels must be >= 1"):
        PatchifiedTensorSpec(
            layout=PatchLayout.SPATIAL, original_channels=0, patch_size=2
        )


def test_bool_is_not_an_int_channel_count() -> None:
    with pytest.raises(TypeError, match="original_channels must be an int"):
        PatchifiedTensorSpec(
            layout=PatchLayout.SPATIAL, original_channels=True, patch_size=2
        )


def test_spatial_and_patch_grid_are_not_inferred_from_shape() -> None:
    array = torch.zeros(2, 12, 4, 4)
    spatial = PatchifiedTensorSpec(
        layout=PatchLayout.SPATIAL, original_channels=12, patch_size=2
    )
    grid = PatchifiedTensorSpec(
        layout=PatchLayout.PATCH_GRID, original_channels=3, patch_size=2
    )
    validate_patchified_tensor(array.contiguous(), spec=spatial)
    validate_patchified_tensor(array.contiguous(), spec=grid)
    with pytest.raises(ValueError, match="requires shape"):
        validate_patchified_tensor(
            array.contiguous(),
            spec=PatchifiedTensorSpec(
                layout=PatchLayout.SPATIAL, original_channels=3, patch_size=2
            ),
        )


def test_validate_mask_rejects_non_constant_spatial() -> None:
    spec = PatchifiedMaskSpec(
        layout=PatchLayout.SPATIAL,
        patch_size=2,
        polarity=MaskPolarity.REMOVED_IS_ONE,
    )
    mask = torch.zeros(1, 4, 4)
    mask[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="not constant"):
        validate_patchified_mask(mask.contiguous(), spec=spec)


def test_validate_mask_rejects_integer_dtype() -> None:
    spec = PatchifiedMaskSpec(
        layout=PatchLayout.PATCH_GRID,
        patch_size=2,
        polarity=MaskPolarity.VISIBLE_IS_ONE,
    )
    with pytest.raises(ValueError, match="floating point"):
        validate_patchified_mask(torch.zeros(1, 2, 2, dtype=torch.int64), spec=spec)


def test_specs_with_same_parts_are_equal() -> None:
    left = PatchifiedTensorSpec(
        layout=PatchLayout.SEQUENCE, original_channels=3, patch_size=4
    )
    right = PatchifiedTensorSpec(
        layout=PatchLayout.SEQUENCE, original_channels=3, patch_size=4
    )
    assert left == right
    assert hash(left) == hash(right)
    other = PatchifiedTensorSpec(
        layout=PatchLayout.PATCH_GRID, original_channels=3, patch_size=4
    )
    assert left != other
