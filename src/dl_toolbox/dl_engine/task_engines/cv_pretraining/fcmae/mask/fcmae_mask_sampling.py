"""Random non-overlapping MAE patch masks at image resolution."""

from __future__ import annotations

import torch
from torch import Tensor


def random_patch_mask(
    images: Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    content_bbox: Tensor,
) -> Tensor:
    """Return (N, 1, H, W) with 1 on removed patches and 0 on visible patches.

    ``content_bbox`` is ``(N, 4)`` ``[y0, x0, y1, x1]`` in the current image.
    Only patches fully inside that box can be removed. Pad and boundary
    patches stay visible, so masked MSE never sees padded pixels.
    ``mask_ratio`` is the removed fraction of those fully-inside patches.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape (N, C, H, W), got {tuple(images.shape)}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    batch, _, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"image spatial {(height, width)} must be divisible by "
            f"{patch_size}x{patch_size}"
        )
    bbox = _content_bbox(
        content_bbox, batch=batch, height=height, width=width
    ).to(device=images.device)
    token_h = height // patch_size
    token_w = width // patch_size
    n_tokens = token_h * token_w
    eligible = _patches_inside_bbox(
        bbox, patch_size=patch_size, token_h=token_h, token_w=token_w
    ).reshape(batch, n_tokens)
    n_eligible = eligible.sum(dim=1)
    if bool((n_eligible < 1).any()):
        raise ValueError(
            "content_bbox must contain at least one full MAE patch per image"
        )
    n_keep = (n_eligible.to(dtype=torch.float32) * (1.0 - mask_ratio)).to(
        dtype=torch.long
    )
    noise = torch.rand(batch, n_tokens, device=images.device, dtype=torch.float32)
    noise = noise.masked_fill(~eligible, 2.0)
    rank = torch.argsort(torch.argsort(noise, dim=1), dim=1)
    removed = eligible & (rank >= n_keep.unsqueeze(1))
    token_map = removed.to(dtype=images.dtype).reshape(batch, 1, token_h, token_w)
    return token_map.repeat_interleave(patch_size, dim=2).repeat_interleave(
        patch_size, dim=3
    )


def _content_bbox(
    raw: Tensor, *, batch: int, height: int, width: int
) -> Tensor:
    if raw.ndim != 2 or raw.shape[0] != batch or raw.shape[1] != 4:
        raise ValueError(
            f"content_bbox must have shape ({batch}, 4), got {tuple(raw.shape)}"
        )
    bbox = raw.to(dtype=torch.long)
    y0 = bbox[:, 0]
    x0 = bbox[:, 1]
    y1 = bbox[:, 2]
    x1 = bbox[:, 3]
    if bool((y0 < 0).any()) or bool((x0 < 0).any()):
        raise ValueError("content_bbox origin must be >= 0")
    if bool((y1 > height).any()) or bool((x1 > width).any()):
        raise ValueError(
            f"content_bbox must fit in image {(height, width)}, got {bbox.tolist()}"
        )
    if bool((y1 <= y0).any()) or bool((x1 <= x0).any()):
        raise ValueError("content_bbox must have y1 > y0 and x1 > x0")
    return bbox


def _patches_inside_bbox(
    bbox: Tensor,
    *,
    patch_size: int,
    token_h: int,
    token_w: int,
) -> Tensor:
    y0 = bbox[:, 0].view(-1, 1, 1)
    x0 = bbox[:, 1].view(-1, 1, 1)
    y1 = bbox[:, 2].view(-1, 1, 1)
    x1 = bbox[:, 3].view(-1, 1, 1)
    token_y = torch.arange(token_h, device=bbox.device, dtype=bbox.dtype)
    token_x = torch.arange(token_w, device=bbox.device, dtype=bbox.dtype)
    top = token_y.view(1, token_h, 1) * patch_size
    left = token_x.view(1, 1, token_w) * patch_size
    bottom = top + patch_size
    right = left + patch_size
    return (top >= y0) & (left >= x0) & (bottom <= y1) & (right <= x1)
