from __future__ import annotations

import pytest
import torch
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Model

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_masked_encoder import (
    MaskedConvNextV2,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_weights import (
    inject_unmasked_weights,
)

IMAGE_H = 64
IMAGE_W = 96
MAE_PATCH = 32
TOKEN_H = IMAGE_H // MAE_PATCH
TOKEN_W = IMAGE_W // MAE_PATCH


def _tiny_config() -> ConvNextV2Config:
    return ConvNextV2Config(
        num_channels=1,
        patch_size=4,
        num_stages=4,
        hidden_sizes=(8, 16, 32, 64),
        depths=(1, 1, 1, 1),
        drop_path_rate=0.0,
        image_size=IMAGE_H,
    )


def _image_mask(batch: int, masked_tokens: tuple[tuple[int, int], ...] = ()) -> torch.Tensor:
    mask = torch.zeros(batch, 1, IMAGE_H, IMAGE_W)
    for token_y, token_x in masked_tokens:
        y0 = token_y * MAE_PATCH
        x0 = token_x * MAE_PATCH
        mask[:, :, y0 : y0 + MAE_PATCH, x0 : x0 + MAE_PATCH] = 1.0
    return mask


def test_constructor_matches_hf_config_only() -> None:
    config = _tiny_config()
    model = MaskedConvNextV2(config)
    assert model.config is config
    x = torch.randn(2, 1, IMAGE_H, IMAGE_W)
    out = model(x, _image_mask(2))
    assert out.shape == (2, 64, TOKEN_H, TOKEN_W)


def test_injected_weights_match_unmasked_when_nothing_is_masked() -> None:
    config = _tiny_config()
    unmasked = ConvNextV2Model(config)
    unmasked.eval()
    masked = MaskedConvNextV2(config)
    inject_unmasked_weights(masked, unmasked)
    masked.eval()

    x = torch.randn(2, 1, IMAGE_H, IMAGE_W)
    with torch.no_grad():
        expected = unmasked(x).last_hidden_state
        got = masked(x, _image_mask(2))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_from_unmasked_copies_backbone() -> None:
    config = _tiny_config()
    unmasked = ConvNextV2Model(config)
    unmasked.eval()
    masked = MaskedConvNextV2.from_unmasked(unmasked)
    masked.eval()
    x = torch.randn(1, 1, IMAGE_H, IMAGE_W)
    with torch.no_grad():
        expected = unmasked(x).last_hidden_state
        got = masked(x, _image_mask(1))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_visible_positions_ignore_masked_patch_content() -> None:
    config = _tiny_config()
    torch.manual_seed(0)
    masked = MaskedConvNextV2(config)
    masked.eval()

    token_mask = _image_mask(1, masked_tokens=((0, 0),))
    visible = torch.randn(1, 1, IMAGE_H, IMAGE_W)
    zeros_fill = visible.clone()
    rand_fill = visible.clone()
    zeros_fill[:, :, :MAE_PATCH, :MAE_PATCH] = 0.0
    rand_fill[:, :, :MAE_PATCH, :MAE_PATCH] = torch.randn(1, 1, MAE_PATCH, MAE_PATCH)

    with torch.no_grad():
        out_zeros = masked(zeros_fill, token_mask)
        out_rand = masked(rand_fill, token_mask)

    keep = torch.ones(TOKEN_H, TOKEN_W, dtype=torch.bool)
    keep[0, 0] = False
    torch.testing.assert_close(
        out_zeros[:, :, keep],
        out_rand[:, :, keep],
        atol=1e-5,
        rtol=1e-5,
    )


def test_rejects_image_not_divisible_by_mae_patch() -> None:
    model = MaskedConvNextV2(_tiny_config())
    x = torch.randn(1, 1, 48, 64)
    mask = torch.zeros(1, 1, 48, 64)
    with pytest.raises(ValueError, match="divisible"):
        model(x, mask)


def test_rejects_mask_that_does_not_match_image() -> None:
    model = MaskedConvNextV2(_tiny_config())
    x = torch.randn(1, 1, IMAGE_H, IMAGE_W)
    mask = torch.zeros(1, 1, TOKEN_H, TOKEN_W)
    with pytest.raises(ValueError, match="must match image spatial"):
        model(x, mask)


def test_rejects_mask_that_cuts_a_patch() -> None:
    model = MaskedConvNextV2(_tiny_config())
    x = torch.randn(1, 1, IMAGE_H, IMAGE_W)
    mask = torch.zeros(1, 1, IMAGE_H, IMAGE_W)
    mask[:, :, :16, :16] = 1.0
    with pytest.raises(ValueError, match="non-overlapping"):
        model(x, mask)
