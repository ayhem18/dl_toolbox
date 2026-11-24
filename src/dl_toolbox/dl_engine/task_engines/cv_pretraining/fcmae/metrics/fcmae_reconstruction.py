"""Pixel MSE and per-patch z-MSE. Per-item vectors, then named reducers."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject, concat_dim0
from dl_toolbox.dl_engine.metrics.reducers import max_items, mean_items, min_items, std_items

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_loss import (
    MaskedMseLoss,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_zscore import (
    spatial_per_patch_zscore,
)

KEY_IMAGE = "image"
KEY_IMAGE_PIXEL = "image_pixel"
KEY_MASK = "mask"
RECONSTRUCTION_KEYS = frozenset({KEY_OUTPUT, KEY_IMAGE_PIXEL, KEY_MASK})
MSE_PIXEL = "mse_pixel"
MSE_Z = "mse_z"
EPOCH_REDUCERS = ("mean", "std", "min", "max")
_REDUCERS = {
    "mean": mean_items,
    "std": std_items,
    "min": min_items,
    "max": max_items,
}


def mse_pixel_metric(*, patch_size: int, original_channels: int) -> MetricObject:
    loss = MaskedMseLoss(
        patch_size=patch_size, original_channels=original_channels
    )

    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        return loss.per_item(
            pred=tensors[KEY_OUTPUT],
            image=tensors[KEY_IMAGE_PIXEL],
            mask=tensors[KEY_MASK],
        )

    return MetricObject(
        name=MSE_PIXEL,
        required_keys=RECONSTRUCTION_KEYS,
        metric_function=measure,
        combiner=concat_dim0,
        reducers=_REDUCERS,
    )


def mse_z_metric(*, patch_size: int, original_channels: int) -> MetricObject:
    loss = MaskedMseLoss(
        patch_size=patch_size, original_channels=original_channels
    )

    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        z_image = spatial_per_patch_zscore(
            tensors[KEY_IMAGE_PIXEL], patch_size=patch_size
        )
        return loss.per_item(
            pred=tensors[KEY_OUTPUT],
            image=z_image,
            mask=tensors[KEY_MASK],
        )

    return MetricObject(
        name=MSE_Z,
        required_keys=RECONSTRUCTION_KEYS,
        metric_function=measure,
        combiner=concat_dim0,
        reducers=_REDUCERS,
    )
