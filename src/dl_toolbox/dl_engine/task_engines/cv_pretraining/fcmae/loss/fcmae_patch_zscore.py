"""Per-patch z-score for FCMAE norm-pixel targets."""

from __future__ import annotations

import torch
from torch import Tensor

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify import (
    PatchLayout,
    PatchifiedTensor,
    PatchifiedTensorSpec,
)

ZSCORE_EPS = 1.0e-6
VISIBLE_ZSCORE_ATOL = 1.0e-4
VISIBLE_ZSCORE_RTOL = 1.0e-4


def per_patch_zscore(tensor: PatchifiedTensor) -> PatchifiedTensor:
    """Z-score each patch independently. Result is SEQUENCE layout.

    Patches with variance at or below ``VISIBLE_ZSCORE_ATOL`` are centered
    only (mean 0). Dividing those by ``sqrt(var + eps)`` would amplify noise.
    """
    seq_spec = PatchifiedTensorSpec(
        layout=PatchLayout.SEQUENCE,
        original_channels=tensor.spec.original_channels,
        patch_size=tensor.spec.patch_size,
    )
    if tensor.spec.layout == PatchLayout.SEQUENCE:
        seq = tensor
    else:
        seq = tensor.convert(spec=seq_spec)
    values = seq.tensor
    mean = values.mean(dim=-1, keepdim=True)
    var = values.var(dim=-1, keepdim=True)
    centered = values - mean
    scaled = centered / (var + ZSCORE_EPS).sqrt()
    # return the centered values for patches with variance near 0
    z_values = torch.where(var <= VISIBLE_ZSCORE_ATOL, centered, scaled)
    return PatchifiedTensor(spec=seq_spec, tensor=z_values.contiguous())


def spatial_per_patch_zscore(image: Tensor, *, patch_size: int) -> Tensor:
    """SPATIAL image -> per-patch z-score -> SPATIAL image."""
    if image.ndim != 4:
        raise ValueError(
            f"image must have shape (B, C, Y_dim, X_dim), got {tuple(image.shape)}"
        )
    spatial_spec = PatchifiedTensorSpec(
        layout=PatchLayout.SPATIAL,
        original_channels=int(image.shape[1]),
        patch_size=patch_size,
    )
    spatial = PatchifiedTensor.from_tensor(tensor=image, spec=spatial_spec)
    z_seq = per_patch_zscore(spatial)
    y_patches = int(image.shape[2]) // patch_size
    x_patches = int(image.shape[3]) // patch_size
    return (
        z_seq.unflatten(Y_patches=y_patches, X_patches=x_patches)
        .convert(spec=spatial_spec)
        .tensor
    )
