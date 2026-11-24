from __future__ import annotations

import pytest
import torch
from dl_toolbox.dl_engine.dl_engine_utils.image_patchify import (
    PatchLayout,
    PatchifiedTensor,
    PatchifiedTensorSpec,
)

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_loss import (
    MaskedMseLoss,
    NormPixelLoss,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_zscore import (
    VISIBLE_ZSCORE_ATOL,
    spatial_per_patch_zscore,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.mask.fcmae_mask_sampling import (
    random_patch_mask,
)

IMAGE_H = 64
IMAGE_W = 96
PATCH = 32
CHANNELS = 1


def _spatial_to_pred(images: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    spatial = PatchifiedTensor.from_tensor(
        tensor=images,
        spec=PatchifiedTensorSpec(
            layout=PatchLayout.SPATIAL,
            original_channels=int(images.shape[1]),
            patch_size=patch_size,
        ),
    )
    return spatial.convert(
        spec=PatchifiedTensorSpec(
            layout=PatchLayout.PATCH_GRID,
            original_channels=int(images.shape[1]),
            patch_size=patch_size,
        )
    ).tensor


def _full_content_bbox(batch: int) -> torch.Tensor:
    return torch.tensor(
        [[0, 0, IMAGE_H, IMAGE_W]] * batch, dtype=torch.long
    )


def test_masked_mse_identity_pred_is_zero() -> None:
    images = torch.randn(2, CHANNELS, IMAGE_H, IMAGE_W)
    mask = torch.ones(2, 1, IMAGE_H, IMAGE_W)
    pred = _spatial_to_pred(images, patch_size=PATCH)
    loss = MaskedMseLoss(patch_size=PATCH, original_channels=CHANNELS)
    value = loss(pred=pred, image=images, mask=mask)
    assert float(value) == 0.0


def test_norm_pixel_accepts_unnormalized_visible_patches() -> None:
    images = torch.rand(1, CHANNELS, IMAGE_H, IMAGE_W)
    torch.manual_seed(0)
    mask = random_patch_mask(
        images,
        patch_size=PATCH,
        mask_ratio=0.6,
        content_bbox=_full_content_bbox(1),
    )
    pred = _spatial_to_pred(images, patch_size=PATCH)
    loss = NormPixelLoss(patch_size=PATCH, original_channels=CHANNELS)
    value = loss(pred=pred, image=images, mask=mask)
    assert torch.isfinite(value)


def test_spatial_zscore_centers_near_zero_variance_without_scaling() -> None:
    torch.manual_seed(0)
    images = torch.full((1, CHANNELS, IMAGE_H, IMAGE_W), 0.4)
    noise = 1.0e-8 * torch.randn(1, CHANNELS, PATCH, PATCH)
    images[:, :, :PATCH, :PATCH] = 0.4 + noise
    z_images = spatial_per_patch_zscore(images, patch_size=PATCH)
    patch = z_images[:, :, :PATCH, :PATCH]
    assert float(patch.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(patch.var()) <= VISIBLE_ZSCORE_ATOL
    assert float(patch.abs().max()) < 1.0e-6


def test_norm_pixel_accepts_constant_visible_patches() -> None:
    images = torch.zeros(1, CHANNELS, IMAGE_H, IMAGE_W)
    mask = torch.zeros(1, 1, IMAGE_H, IMAGE_W)
    mask[:, :, :PATCH, :PATCH] = 1.0
    pred = _spatial_to_pred(images, patch_size=PATCH)
    loss = NormPixelLoss(patch_size=PATCH, original_channels=CHANNELS)
    value = loss(pred=pred, image=images, mask=mask)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


def test_norm_pixel_identity_after_zscore_is_zero() -> None:
    images = torch.randn(2, CHANNELS, IMAGE_H, IMAGE_W)
    z_images = spatial_per_patch_zscore(images, patch_size=PATCH)
    torch.manual_seed(1)
    mask = random_patch_mask(
        z_images,
        patch_size=PATCH,
        mask_ratio=0.6,
        content_bbox=_full_content_bbox(2),
    )
    pred = _spatial_to_pred(z_images, patch_size=PATCH)
    loss = NormPixelLoss(patch_size=PATCH, original_channels=CHANNELS)
    value = loss(pred=pred, image=z_images, mask=mask)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


def test_masked_mse_ignores_visible_patches() -> None:
    images = torch.randn(1, CHANNELS, IMAGE_H, IMAGE_W)
    mask = torch.zeros(1, 1, IMAGE_H, IMAGE_W)
    mask[:, :, :PATCH, :PATCH] = 1.0
    pred_images = images.clone()
    pred_images[:, :, PATCH:, :] = 7.0
    pred = _spatial_to_pred(pred_images, patch_size=PATCH)
    loss = MaskedMseLoss(patch_size=PATCH, original_channels=CHANNELS)
    value = loss.per_item(pred=pred, image=images, mask=mask)
    assert float(value) == pytest.approx(0.0, abs=1e-6)
