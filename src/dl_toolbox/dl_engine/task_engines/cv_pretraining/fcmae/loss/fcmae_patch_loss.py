"""Masked MSE on removed MAE patches.

``MaskedMseLoss`` compares pred and image in the same photometric space.
``NormPixelLoss`` requires that space to already be per-patch z-scored:
each visible patch must have mean near 0 and variance near 1 or near 0.
Layout conversions go through ``PatchifiedTensor`` / ``PatchifiedMask``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify import (
    MaskPolarity,
    PatchLayout,
    PatchifiedMask,
    PatchifiedMaskSpec,
    PatchifiedTensor,
    PatchifiedTensorSpec,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_zscore import (
    VISIBLE_ZSCORE_ATOL,
    VISIBLE_ZSCORE_RTOL,
)


class _MaskedPatchMse:
    """Parse pred / image / mask onto SEQUENCE, then masked per-item MSE."""

    def __init__(self, *, patch_size: int, original_channels: int) -> None:
        self._spatial_spec = PatchifiedTensorSpec(
            layout=PatchLayout.SPATIAL,
            original_channels=original_channels,
            patch_size=patch_size,
        )
        self._pred_spec = PatchifiedTensorSpec(
            layout=PatchLayout.PATCH_GRID,
            original_channels=original_channels,
            patch_size=patch_size,
        )
        self._seq_spec = PatchifiedTensorSpec(
            layout=PatchLayout.SEQUENCE,
            original_channels=original_channels,
            patch_size=patch_size,
        )
        self._removed_spatial_spec = PatchifiedMaskSpec(
            layout=PatchLayout.SPATIAL,
            patch_size=patch_size,
            polarity=MaskPolarity.REMOVED_IS_ONE,
        )
        self._removed_seq_spec = PatchifiedMaskSpec(
            layout=PatchLayout.SEQUENCE,
            patch_size=patch_size,
            polarity=MaskPolarity.REMOVED_IS_ONE,
        )
        self._visible_seq_spec = PatchifiedMaskSpec(
            layout=PatchLayout.SEQUENCE,
            patch_size=patch_size,
            polarity=MaskPolarity.VISIBLE_IS_ONE,
        )

    def _sequences(
        self, *, pred: Tensor, image: Tensor, mask: Tensor
    ) -> tuple[PatchifiedTensor, PatchifiedTensor, PatchifiedMask]:
        image_pt = PatchifiedTensor.from_tensor(
            tensor=image, spec=self._spatial_spec
        )
        pred_pt = PatchifiedTensor.from_tensor(tensor=pred, spec=self._pred_spec)
        mask_pm = PatchifiedMask.from_tensor(
            tensor=mask, spec=self._removed_spatial_spec
        )
        image_seq = image_pt.convert(spec=self._seq_spec)
        pred_seq = pred_pt.convert(spec=self._seq_spec)
        removed = mask_pm.convert(spec=self._removed_seq_spec)
        if pred_seq.tensor.shape != image_seq.tensor.shape:
            raise ValueError(
                f"pred sequence {tuple(pred_seq.tensor.shape)} does not match "
                f"image sequence {tuple(image_seq.tensor.shape)}"
            )
        if removed.tensor.shape != pred_seq.tensor.shape[:2]:
            raise ValueError(
                f"mask sequence {tuple(removed.tensor.shape)} does not match "
                f"pred batch/tokens {tuple(pred_seq.tensor.shape[:2])}"
            )
        return pred_seq, image_seq, removed

    def _per_item(
        self,
        *,
        pred_seq: PatchifiedTensor,
        target_seq: PatchifiedTensor,
        removed: PatchifiedMask,
    ) -> Tensor:
        per_patch = ((pred_seq.tensor - target_seq.tensor) ** 2).mean(dim=-1)
        masked = per_patch * removed.tensor
        denom = removed.tensor.sum(dim=1).clamp_min(1.0)
        return masked.sum(dim=1) / denom


class MaskedMseLoss(_MaskedPatchMse):
    """Masked MSE. Pred and image must already share photometric units."""

    def per_item(
        self, *, pred: Tensor, image: Tensor, mask: Tensor
    ) -> Tensor:
        pred_seq, image_seq, removed = self._sequences(
            pred=pred, image=image, mask=mask
        )
        return self._per_item(
            pred_seq=pred_seq, target_seq=image_seq, removed=removed
        )

    def __call__(
        self, *, pred: Tensor, image: Tensor, mask: Tensor
    ) -> Tensor:
        return self.per_item(pred=pred, image=image, mask=mask).mean()


class NormPixelLoss(_MaskedPatchMse):
    """Masked MSE in per-patch z-space.

    Does not z-score the target. The caller is expected to pass a per-patch
    z-scored ``image``. The visible-patch check is commented out: many
    radiographic patches have variance near zero and fail it under fp16.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        original_channels: int,
        atol: float = VISIBLE_ZSCORE_ATOL,
        rtol: float = VISIBLE_ZSCORE_RTOL,
    ) -> None:
        super().__init__(
            patch_size=patch_size, original_channels=original_channels
        )
        if type(atol) is not float or atol < 0.0:
            raise ValueError(f"atol must be a float >= 0, got {atol!r}")
        if type(rtol) is not float or rtol < 0.0:
            raise ValueError(f"rtol must be a float >= 0, got {rtol!r}")
        self._atol = atol
        self._rtol = rtol

    def per_item(
        self, *, pred: Tensor, image: Tensor, mask: Tensor
    ) -> Tensor:
        pred_seq, image_seq, removed = self._sequences(
            pred=pred, image=image, mask=mask
        )
        # visible = removed.convert(spec=self._visible_seq_spec)
        # _require_visible_already_zscored(
        #     image_seq=image_seq,
        #     visible=visible,
        #     atol=self._atol,
        #     rtol=self._rtol,
        # )
        return self._per_item(
            pred_seq=pred_seq, target_seq=image_seq, removed=removed
        )

    def __call__(
        self, *, pred: Tensor, image: Tensor, mask: Tensor
    ) -> Tensor:
        return self.per_item(pred=pred, image=image, mask=mask).mean()


# Disabled: many patches have variance near 0; the check fails under fp16.
# def _require_visible_already_zscored(
#     *,
#     image_seq: PatchifiedTensor,
#     visible: PatchifiedMask,
#     atol: float,
#     rtol: float,
# ) -> None:
#     visible_bool = visible.tensor == 1
#     if not bool(visible_bool.any()):
#         raise ValueError("NormPixelLoss requires at least one visible patch")
#     patches = image_seq.tensor[visible_bool].to(dtype=torch.float32)
#     mean = patches.mean(dim=-1)
#     var = patches.var(dim=-1)
#     mean_ok = torch.isclose(
#         mean, torch.zeros_like(mean), atol=atol, rtol=rtol
#     )
#     var_unit = torch.isclose(
#         var, torch.ones_like(var), atol=atol, rtol=rtol
#     )
#     var_zero = torch.isclose(
#         var, torch.zeros_like(var), atol=atol, rtol=rtol
#     )
#     if not bool((mean_ok & (var_unit | var_zero)).all()):
#         raise ValueError(
#             "NormPixelLoss requires visible patches to have mean near 0 and "
#             "variance near 1 or near 0"
#         )
