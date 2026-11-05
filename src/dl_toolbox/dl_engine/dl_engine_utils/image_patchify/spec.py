"""Patch layout, polarity, and specs.

``PatchifiedTensor`` / ``PatchifiedMask`` are a spec plus a batched tensor.
Grid size lives on the payload except on ``SEQUENCE``, where it is an
``unflatten`` argument, not a spec field.
"""

from __future__ import annotations

import torch
from torch import Tensor


class PatchLayout:
    """How a patch-partitioned payload is arranged."""

    def __init__(self, *, name: str) -> None:
        if name not in ("spatial", "patch_grid", "sequence"):
            raise ValueError(
                "layout name must be 'spatial', 'patch_grid', or 'sequence', "
                f"got {name!r}"
            )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PatchLayout) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"PatchLayout.{self._name.upper()}"


PatchLayout.SPATIAL = PatchLayout(name="spatial")
PatchLayout.PATCH_GRID = PatchLayout(name="patch_grid")
PatchLayout.SEQUENCE = PatchLayout(name="sequence")


class MaskPolarity:
    """Meaning of value 1. A claim; the tensor cannot prove it."""

    def __init__(self, *, name: str) -> None:
        if name not in ("removed_is_one", "visible_is_one"):
            raise ValueError(
                "polarity name must be 'removed_is_one' or 'visible_is_one', "
                f"got {name!r}"
            )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MaskPolarity) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"MaskPolarity.{self._name.upper()}"


MaskPolarity.REMOVED_IS_ONE = MaskPolarity(name="removed_is_one")
MaskPolarity.VISIBLE_IS_ONE = MaskPolarity(name="visible_is_one")


def require_patch_size(*, patch_size: object) -> int:
    if not isinstance(patch_size, int) or isinstance(patch_size, bool):
        raise TypeError(f"patch_size must be an int, got {type(patch_size)!r}")
    if patch_size <= 1:
        raise ValueError(f"patch_size must be > 1, got {patch_size}")
    return patch_size


def require_original_channels(*, original_channels: object) -> int:
    if not isinstance(original_channels, int) or isinstance(original_channels, bool):
        raise TypeError(
            f"original_channels must be an int, got {type(original_channels)!r}"
        )
    if original_channels < 1:
        raise ValueError(
            f"original_channels must be >= 1, got {original_channels}"
        )
    return original_channels


def require_patch_count(*, name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value)!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


class PatchifiedTensorSpec:
    """``PatchLayout`` × ``original_channels`` × ``patch_size``."""

    def __init__(
        self,
        *,
        layout: PatchLayout,
        original_channels: int,
        patch_size: int,
    ) -> None:
        if not isinstance(layout, PatchLayout):
            raise TypeError(f"layout must be PatchLayout, got {type(layout)!r}")
        self._layout = layout
        self._original_channels = require_original_channels(
            original_channels=original_channels
        )
        self._patch_size = require_patch_size(patch_size=patch_size)

    @property
    def layout(self) -> PatchLayout:
        return self._layout

    @property
    def original_channels(self) -> int:
        return self._original_channels

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def patch_vector_size(self) -> int:
        patch_size = self._patch_size
        return self._original_channels * patch_size * patch_size

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PatchifiedTensorSpec)
            and self._layout == other._layout
            and self._original_channels == other._original_channels
            and self._patch_size == other._patch_size
        )

    def __hash__(self) -> int:
        return hash((self._layout, self._original_channels, self._patch_size))

    def __repr__(self) -> str:
        return (
            "PatchifiedTensorSpec("
            f"layout={self._layout!r}, "
            f"original_channels={self._original_channels}, "
            f"patch_size={self._patch_size})"
        )


class PatchifiedMaskSpec:
    """``PatchLayout`` × ``patch_size`` × ``MaskPolarity``. No ``original_channels``."""

    def __init__(
        self,
        *,
        layout: PatchLayout,
        patch_size: int,
        polarity: MaskPolarity,
    ) -> None:
        if not isinstance(layout, PatchLayout):
            raise TypeError(f"layout must be PatchLayout, got {type(layout)!r}")
        if not isinstance(polarity, MaskPolarity):
            raise TypeError(f"polarity must be MaskPolarity, got {type(polarity)!r}")
        self._layout = layout
        self._patch_size = require_patch_size(patch_size=patch_size)
        self._polarity = polarity

    @property
    def layout(self) -> PatchLayout:
        return self._layout

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def polarity(self) -> MaskPolarity:
        return self._polarity

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PatchifiedMaskSpec)
            and self._layout == other._layout
            and self._patch_size == other._patch_size
            and self._polarity == other._polarity
        )

    def __hash__(self) -> int:
        return hash((self._layout, self._patch_size, self._polarity))

    def __repr__(self) -> str:
        return (
            "PatchifiedMaskSpec("
            f"layout={self._layout!r}, "
            f"patch_size={self._patch_size}, "
            f"polarity={self._polarity!r})"
        )


def _require_batched_tensor(tensor: object) -> Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.numel() == 0:
        raise ValueError("tensor is empty")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.ndim < 1 or tensor.shape[0] < 1:
        raise ValueError(
            f"tensor must have a batch dimension, got shape {tuple(tensor.shape)}"
        )
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("tensor contains NaN or inf")
    return tensor


def expected_tensor_shape_suffix(*, spec: PatchifiedTensorSpec) -> str:
    channels = spec.original_channels
    patch_size = spec.patch_size
    patch_vector = spec.patch_vector_size
    if spec.layout == PatchLayout.SPATIAL:
        return f"(B, {channels}, Y_dim, X_dim) with Y_dim, X_dim divisible by {patch_size}"
    if spec.layout == PatchLayout.PATCH_GRID:
        return f"(B, {patch_vector}, Y_patches, X_patches)"
    return f"(B, Y_patches * X_patches, {patch_vector})"


def expected_mask_shape_suffix(*, spec: PatchifiedMaskSpec) -> str:
    patch_size = spec.patch_size
    if spec.layout == PatchLayout.SPATIAL:
        return f"(B, Y_dim, X_dim) with Y_dim, X_dim divisible by {patch_size}"
    if spec.layout == PatchLayout.PATCH_GRID:
        return "(B, Y_patches, X_patches)"
    return "(B, Y_patches * X_patches)"


def validate_patchified_tensor(
    tensor: object, *, spec: PatchifiedTensorSpec
) -> Tensor:
    """Raise unless ``tensor`` already matches ``spec``. Layout is a claim."""
    payload = _require_batched_tensor(tensor)
    patch_size = spec.patch_size
    channels = spec.original_channels
    patch_vector = spec.patch_vector_size
    if spec.layout == PatchLayout.SPATIAL:
        legal = (
            payload.ndim == 4
            and payload.shape[1] == channels
            and payload.shape[2] >= patch_size
            and payload.shape[3] >= patch_size
            and payload.shape[2] % patch_size == 0
            and payload.shape[3] % patch_size == 0
        )
    elif spec.layout == PatchLayout.PATCH_GRID:
        legal = (
            payload.ndim == 4
            and payload.shape[1] == patch_vector
            and payload.shape[2] >= 1
            and payload.shape[3] >= 1
        )
    else:
        legal = payload.ndim == 3 and payload.shape[1] >= 1 and payload.shape[2] == patch_vector
    if not legal:
        raise ValueError(
            f"{spec!r} requires shape {expected_tensor_shape_suffix(spec=spec)}, "
            f"got {tuple(payload.shape)}"
        )
    return payload


def _require_binary_float(payload: Tensor) -> None:
    if not payload.is_floating_point():
        raise ValueError(
            f"mask tensor must be floating point with values in {{0, 1}}, "
            f"got dtype {payload.dtype}"
        )
    binary = torch.logical_or(payload == 0, payload == 1)
    if not bool(binary.all()):
        raise ValueError("mask tensor values must be in {0, 1}")


def _require_spatial_mask_patch_constant(*, payload: Tensor, patch_size: int) -> None:
    batch, y_dim, x_dim = payload.shape
    y_patches = y_dim // patch_size
    x_patches = x_dim // patch_size
    blocks = payload.reshape(batch, y_patches, patch_size, x_patches, patch_size)
    anchor = blocks[:, :, 0:1, :, 0:1]
    if not torch.equal(blocks, anchor.expand_as(blocks)):
        raise ValueError(
            f"spatial mask is not constant on non-overlapping "
            f"{patch_size}x{patch_size} patches"
        )


def validate_patchified_mask(tensor: object, *, spec: PatchifiedMaskSpec) -> Tensor:
    """Raise unless ``tensor`` already matches ``spec``. Polarity is a claim."""
    payload = _require_batched_tensor(tensor)
    _require_binary_float(payload)
    patch_size = spec.patch_size
    if spec.layout == PatchLayout.SPATIAL:
        legal = (
            payload.ndim == 3
            and payload.shape[1] >= patch_size
            and payload.shape[2] >= patch_size
            and payload.shape[1] % patch_size == 0
            and payload.shape[2] % patch_size == 0
        )
        if legal:
            _require_spatial_mask_patch_constant(
                payload=payload, patch_size=patch_size
            )
    elif spec.layout == PatchLayout.PATCH_GRID:
        legal = payload.ndim == 3 and payload.shape[1] >= 1 and payload.shape[2] >= 1
    else:
        legal = payload.ndim == 2 and payload.shape[1] >= 1
    if not legal:
        raise ValueError(
            f"{spec!r} requires shape {expected_mask_shape_suffix(spec=spec)}, "
            f"got {tuple(payload.shape)}"
        )
    return payload
