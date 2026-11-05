"""``PatchifiedTensor`` is ``PatchifiedTensorSpec`` plus a batched tensor.

Tensor source: caller must pass ``spec``. ``SEQUENCE`` → grid is ``unflatten``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.spec import (
    PatchLayout,
    PatchifiedTensorSpec,
    require_patch_count,
    validate_patchified_tensor,
)


class PatchifiedTensor:
    """Patch-partitioned batched tensor: ``spec`` + ``tensor``. Construction validates."""

    def __init__(self, *, spec: PatchifiedTensorSpec, tensor: Tensor) -> None:
        validate_patchified_tensor(tensor, spec=spec)
        self._spec = spec
        self._tensor = tensor

    @property
    def spec(self) -> PatchifiedTensorSpec:
        return self._spec

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @classmethod
    def from_tensor(cls, *, tensor: Tensor, spec: PatchifiedTensorSpec) -> PatchifiedTensor:
        """Tensor source. ``spec`` is required. Makes the buffer contiguous."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
        return cls(spec=spec, tensor=tensor.contiguous())

    def convert(self, *, spec: PatchifiedTensorSpec) -> PatchifiedTensor:
        """Map this tensor onto ``spec``.

        ``original_channels`` and ``patch_size`` must match. ``SPATIAL`` ↔
        ``PATCH_GRID`` and both → ``SEQUENCE`` are unique. ``SEQUENCE`` →
        ``PATCH_GRID`` / ``SPATIAL`` is ``unflatten``, not ``convert``.
        """
        if spec == self._spec:
            return PatchifiedTensor(spec=spec, tensor=self._tensor)
        if spec.original_channels != self._spec.original_channels:
            raise ValueError(
                "convert cannot change original_channels "
                f"({self._spec.original_channels} -> {spec.original_channels})"
            )
        if spec.patch_size != self._spec.patch_size:
            raise ValueError(
                "convert cannot change patch_size "
                f"({self._spec.patch_size} -> {spec.patch_size})"
            )
        if self._spec.layout == PatchLayout.SEQUENCE:
            raise ValueError(
                "SEQUENCE to PATCH_GRID or SPATIAL is not unique; "
                "call unflatten(Y_patches=..., X_patches=...)"
            )
        grid = self._to_patch_grid()
        if spec.layout == PatchLayout.PATCH_GRID:
            return PatchifiedTensor(spec=spec, tensor=grid)
        if spec.layout == PatchLayout.SPATIAL:
            return PatchifiedTensor(
                spec=spec, tensor=self._spatial_from_patch_grid(grid=grid)
            )
        if spec.layout == PatchLayout.SEQUENCE:
            return PatchifiedTensor(
                spec=spec, tensor=self._sequence_from_patch_grid(grid=grid)
            )
        raise ValueError(f"no layout conversion to {spec.layout!r}")

    def unflatten(self, *, Y_patches: int, X_patches: int) -> PatchifiedTensor:
        """Named inverse of flatten. ``SEQUENCE`` → ``PATCH_GRID``. Grid is not spec."""
        if self._spec.layout != PatchLayout.SEQUENCE:
            raise ValueError(
                f"unflatten requires SEQUENCE layout, got {self._spec.layout!r}"
            )
        y_patches = require_patch_count(name="Y_patches", value=Y_patches)
        x_patches = require_patch_count(name="X_patches", value=X_patches)
        batch, n_tokens, patch_vector = self._tensor.shape
        expected = y_patches * x_patches
        if n_tokens != expected:
            raise ValueError(
                f"Y_patches * X_patches = {y_patches} * {x_patches} = {expected} "
                f"does not match SEQUENCE length {n_tokens}"
            )
        grid = (
            self._tensor.transpose(1, 2)
            .contiguous()
            .reshape(batch, patch_vector, y_patches, x_patches)
        )
        spec = PatchifiedTensorSpec(
            layout=PatchLayout.PATCH_GRID,
            original_channels=self._spec.original_channels,
            patch_size=self._spec.patch_size,
        )
        return PatchifiedTensor(spec=spec, tensor=grid.contiguous())

    def _to_patch_grid(self) -> Tensor:
        if self._spec.layout == PatchLayout.PATCH_GRID:
            return self._tensor
        batch, channels, y_dim, x_dim = self._tensor.shape
        patch_size = self._spec.patch_size
        y_patches = y_dim // patch_size
        x_patches = x_dim // patch_size
        patches = self._tensor.reshape(
            batch, channels, y_patches, patch_size, x_patches, patch_size
        )
        patches = patches.permute(0, 1, 3, 5, 2, 4).contiguous()
        return patches.reshape(
            batch, channels * patch_size * patch_size, y_patches, x_patches
        )

    def _spatial_from_patch_grid(self, *, grid: Tensor) -> Tensor:
        batch, _, y_patches, x_patches = grid.shape
        patch_size = self._spec.patch_size
        channels = self._spec.original_channels
        patches = grid.reshape(
            batch, channels, patch_size, patch_size, y_patches, x_patches
        )
        patches = patches.permute(0, 1, 4, 2, 5, 3).contiguous()
        return patches.reshape(
            batch, channels, y_patches * patch_size, x_patches * patch_size
        )

    def _sequence_from_patch_grid(self, *, grid: Tensor) -> Tensor:
        batch, patch_vector, y_patches, x_patches = grid.shape
        return (
            grid.reshape(batch, patch_vector, y_patches * x_patches)
            .transpose(1, 2)
            .contiguous()
        )
