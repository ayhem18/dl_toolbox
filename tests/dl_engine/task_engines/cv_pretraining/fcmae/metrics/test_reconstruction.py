from __future__ import annotations

import pytest
import torch
from dl_toolbox.dl_engine.dl_engine_utils.image_patchify import (
    PatchLayout,
    PatchifiedTensor,
    PatchifiedTensorSpec,
)

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_zscore import (
    spatial_per_patch_zscore,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.metrics.fcmae_reconstruction import (
    EPOCH_REDUCERS,
    MSE_PIXEL,
    MSE_Z,
    mse_pixel_metric,
    mse_z_metric,
)

IMAGE_H = 64
IMAGE_W = 96
PATCH = 32
CHANNELS = 1


def _spatial_to_pred(images: torch.Tensor) -> torch.Tensor:
    return PatchifiedTensor.from_tensor(
        tensor=images,
        spec=PatchifiedTensorSpec(
            layout=PatchLayout.SPATIAL,
            original_channels=CHANNELS,
            patch_size=PATCH,
        ),
    ).convert(
        spec=PatchifiedTensorSpec(
            layout=PatchLayout.PATCH_GRID,
            original_channels=CHANNELS,
            patch_size=PATCH,
        )
    ).tensor


def test_metric_names_and_reducers() -> None:
    pixel = mse_pixel_metric(patch_size=PATCH, original_channels=CHANNELS)
    z_metric = mse_z_metric(patch_size=PATCH, original_channels=CHANNELS)
    assert pixel.name == MSE_PIXEL
    assert z_metric.name == MSE_Z
    assert tuple(pixel.reducers) == EPOCH_REDUCERS
    assert tuple(z_metric.reducers) == EPOCH_REDUCERS


def test_identity_pred_is_zero_pixel_mse() -> None:
    images = torch.randn(2, CHANNELS, IMAGE_H, IMAGE_W)
    mask = torch.ones(2, 1, IMAGE_H, IMAGE_W)
    pred = _spatial_to_pred(images)
    tensors = {
        "output": pred,
        "image_pixel": images,
        "mask": mask,
    }
    pixel = mse_pixel_metric(
        patch_size=PATCH, original_channels=CHANNELS
    ).metric_function(tensors)
    z_values = mse_z_metric(
        patch_size=PATCH, original_channels=CHANNELS
    ).metric_function(tensors)
    assert pixel.shape == (2,)
    assert z_values.shape == (2,)
    assert float(pixel.max()) == 0.0
    assert torch.isfinite(z_values).all()


def test_identity_z_pred_is_zero_z_mse() -> None:
    images = torch.randn(2, CHANNELS, IMAGE_H, IMAGE_W)
    z_images = spatial_per_patch_zscore(images, patch_size=PATCH)
    mask = torch.ones(2, 1, IMAGE_H, IMAGE_W)
    pred = _spatial_to_pred(z_images)
    z_values = mse_z_metric(
        patch_size=PATCH, original_channels=CHANNELS
    ).metric_function(
        {
            "output": pred,
            "image_pixel": images,
            "mask": mask,
        }
    )
    assert float(z_values.max()) == pytest.approx(0.0, abs=1e-5)
