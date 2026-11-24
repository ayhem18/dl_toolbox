from __future__ import annotations

import torch
from torch import nn
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_loss import (
    NormPixelLoss,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.loss.fcmae_patch_zscore import (
    spatial_per_patch_zscore,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.mask.fcmae_mask_sampling import (
    random_patch_mask,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_decoder import (
    FcmaeConvNextV2,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_masked_encoder import (
    MaskedConvNextV2,
)

IMAGE_H = 64
IMAGE_W = 96
MAE_PATCH = 32


def _tiny_encoder() -> MaskedConvNextV2:
    config = ConvNextV2Config(
        num_channels=1,
        patch_size=4,
        num_stages=4,
        hidden_sizes=(8, 16, 32, 64),
        depths=(1, 1, 1, 1),
        drop_path_rate=0.0,
        image_size=IMAGE_H,
    )
    return MaskedConvNextV2(config)


def test_decoder_pred_matches_mae_grid() -> None:
    model = FcmaeConvNextV2(_tiny_encoder(), decoder_depth=1, decoder_embed_dim=32)
    assert isinstance(model.proj, nn.Linear)
    assert isinstance(model.pred, nn.Linear)
    images = torch.randn(2, 1, IMAGE_H, IMAGE_W)
    z_images = spatial_per_patch_zscore(images, patch_size=MAE_PATCH)
    torch.manual_seed(0)
    mask = random_patch_mask(
        z_images,
        patch_size=MAE_PATCH,
        mask_ratio=0.6,
        content_bbox=torch.tensor(
            [[0, 0, IMAGE_H, IMAGE_W], [0, 0, IMAGE_H, IMAGE_W]],
            dtype=torch.long,
        ),
    )
    pred = model(z_images, mask)
    assert pred.shape == (
        2,
        MAE_PATCH * MAE_PATCH * 1,
        IMAGE_H // MAE_PATCH,
        IMAGE_W // MAE_PATCH,
    )
    loss = NormPixelLoss(patch_size=MAE_PATCH, original_channels=1)(
        pred=pred, image=z_images, mask=mask
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.proj.weight.grad is not None
    assert model.pred.weight.grad is not None
    assert model.proj.weight.grad.is_contiguous()
    assert model.pred.weight.grad.is_contiguous()
