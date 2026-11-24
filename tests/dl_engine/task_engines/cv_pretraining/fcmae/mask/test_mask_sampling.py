from __future__ import annotations

import torch

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.mask.fcmae_mask_sampling import (
    random_patch_mask,
)

IMAGE_H = 64
IMAGE_W = 96
PATCH = 32


def _full_bbox(batch: int) -> torch.Tensor:
    rows = [[0, 0, IMAGE_H, IMAGE_W]] * batch
    return torch.tensor(rows, dtype=torch.long)


def test_mask_is_image_sized_and_32_aligned() -> None:
    images = torch.zeros(2, 1, IMAGE_H, IMAGE_W)
    torch.manual_seed(0)
    mask = random_patch_mask(
        images, patch_size=PATCH, mask_ratio=0.6, content_bbox=_full_bbox(2)
    )
    assert mask.shape == (2, 1, IMAGE_H, IMAGE_W)
    token_h = IMAGE_H // PATCH
    token_w = IMAGE_W // PATCH
    for batch_i in range(2):
        for token_y in range(token_h):
            for token_x in range(token_w):
                y0 = token_y * PATCH
                x0 = token_x * PATCH
                block = mask[batch_i, 0, y0 : y0 + PATCH, x0 : x0 + PATCH]
                unique = torch.unique(block)
                assert unique.numel() == 1
                assert float(unique[0]) in (0.0, 1.0)


def test_mask_ratio_keeps_int_unmasked_tokens() -> None:
    images = torch.zeros(1, 1, IMAGE_H, IMAGE_W)
    torch.manual_seed(1)
    mask = random_patch_mask(
        images, patch_size=PATCH, mask_ratio=0.6, content_bbox=_full_bbox(1)
    )
    tokens = mask[0, 0, ::PATCH, ::PATCH]
    n_tokens = tokens.numel()
    n_keep = int(n_tokens * (1.0 - 0.6))
    assert int((tokens == 0).sum()) == n_keep
    assert int((tokens == 1).sum()) == n_tokens - n_keep


def test_removed_patches_stay_inside_content_bbox() -> None:
    images = torch.zeros(1, 1, IMAGE_H, IMAGE_W)
    content_x1 = 64
    torch.manual_seed(0)
    mask = random_patch_mask(
        images,
        patch_size=PATCH,
        mask_ratio=0.6,
        content_bbox=torch.tensor([[0, 0, IMAGE_H, content_x1]], dtype=torch.long),
    )
    tokens = mask[0, 0, ::PATCH, ::PATCH]
    token_h = IMAGE_H // PATCH
    token_w = IMAGE_W // PATCH
    n_eligible = 0
    n_removed = 0
    for token_y in range(token_h):
        for token_x in range(token_w):
            x0 = token_x * PATCH
            x1 = x0 + PATCH
            fully_inside = x1 <= content_x1
            if fully_inside:
                n_eligible += 1
                if float(tokens[token_y, token_x]) == 1.0:
                    n_removed += 1
            else:
                assert float(tokens[token_y, token_x]) == 0.0
    n_keep = int(n_eligible * (1.0 - 0.6))
    assert n_removed == n_eligible - n_keep
    assert float(mask[0, 0, :, content_x1:].max()) == 0.0
