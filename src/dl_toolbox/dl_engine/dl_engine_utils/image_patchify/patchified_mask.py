"""``PatchifiedMask`` is ``PatchifiedMaskSpec`` plus a batched binary tensor.

Tensor source: caller must pass ``spec``. ``(B, 1, Y_dim, X_dim)`` squeezes
on spatial parse. ``SEQUENCE`` → grid is ``unflatten``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dl_toolbox.dl_engine.dl_engine_utils.image_patchify.spec import (
    PatchLayout,
    PatchifiedMaskSpec,
    require_patch_count,
    validate_patchified_mask,
)


class PatchifiedMask:
    """Patch-constant binary mask: ``spec`` + ``tensor``. Construction validates."""

    def __init__(self, *, spec: PatchifiedMaskSpec, tensor: Tensor) -> None:
        validate_patchified_mask(tensor, spec=spec)
        self._spec = spec
        self._tensor = tensor

    @property
    def spec(self) -> PatchifiedMaskSpec:
        return self._spec

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @classmethod
    def from_tensor(cls, *, tensor: Tensor, spec: PatchifiedMaskSpec) -> PatchifiedMask:
        """Tensor source. ``spec`` is required. Spatial ``(B, 1, Y_dim, X_dim)`` squeezes."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
        working = tensor
        if (
            spec.layout == PatchLayout.SPATIAL
            and working.ndim == 4
            and working.shape[1] == 1
        ):
            working = working[:, 0]
        return cls(spec=spec, tensor=working.contiguous())

    def convert(self, *, spec: PatchifiedMaskSpec) -> PatchifiedMask:
        """Map this mask onto ``spec``.

        ``patch_size`` must match. Layout maps match the tensor entity.
        Polarity flip is ``1 - mask``. ``SEQUENCE`` → grid is ``unflatten``.
        """
        if spec == self._spec:
            return PatchifiedMask(spec=spec, tensor=self._tensor)
        if spec.patch_size != self._spec.patch_size:
            raise ValueError(
                "convert cannot change patch_size "
                f"({self._spec.patch_size} -> {spec.patch_size})"
            )
        working_spec = PatchifiedMaskSpec(
            layout=spec.layout,
            patch_size=self._spec.patch_size,
            polarity=self._spec.polarity,
        )
        if working_spec.layout != self._spec.layout:
            if self._spec.layout == PatchLayout.SEQUENCE:
                raise ValueError(
                    "SEQUENCE to PATCH_GRID or SPATIAL is not unique; "
                    "call unflatten(Y_patches=..., X_patches=...)"
                )
            grid = self._to_patch_grid()
            if working_spec.layout == PatchLayout.PATCH_GRID:
                working = grid
            elif working_spec.layout == PatchLayout.SPATIAL:
                working = self._spatial_from_patch_grid(grid=grid)
            elif working_spec.layout == PatchLayout.SEQUENCE:
                working = self._sequence_from_patch_grid(grid=grid)
            else:
                raise ValueError(f"no layout conversion to {spec.layout!r}")
        else:
            working = self._tensor
        if spec.polarity != self._spec.polarity:
            working = (1.0 - working).contiguous()
        return PatchifiedMask(spec=spec, tensor=working)

    def unflatten(self, *, Y_patches: int, X_patches: int) -> PatchifiedMask:
        """Named inverse of flatten. ``SEQUENCE`` → ``PATCH_GRID``. Grid is not spec."""
        if self._spec.layout != PatchLayout.SEQUENCE:
            raise ValueError(
                f"unflatten requires SEQUENCE layout, got {self._spec.layout!r}"
            )
        y_patches = require_patch_count(name="Y_patches", value=Y_patches)
        x_patches = require_patch_count(name="X_patches", value=X_patches)
        batch, n_tokens = self._tensor.shape
        expected = y_patches * x_patches
        if n_tokens != expected:
            raise ValueError(
                f"Y_patches * X_patches = {y_patches} * {x_patches} = {expected} "
                f"does not match SEQUENCE length {n_tokens}"
            )
        grid = self._tensor.reshape(batch, y_patches, x_patches).contiguous()
        spec = PatchifiedMaskSpec(
            layout=PatchLayout.PATCH_GRID,
            patch_size=self._spec.patch_size,
            polarity=self._spec.polarity,
        )
        return PatchifiedMask(spec=spec, tensor=grid)

    def _to_patch_grid(self) -> Tensor:
        if self._spec.layout == PatchLayout.PATCH_GRID:
            return self._tensor
        batch, y_dim, x_dim = self._tensor.shape
        patch_size = self._spec.patch_size
        y_patches = y_dim // patch_size
        x_patches = x_dim // patch_size
        blocks = self._tensor.reshape(
            batch, y_patches, patch_size, x_patches, patch_size
        )
        return blocks[:, :, 0, :, 0].contiguous()

    def _spatial_from_patch_grid(self, *, grid: Tensor) -> Tensor:
        batch, y_patches, x_patches = grid.shape
        patch_size = self._spec.patch_size
        expanded = grid.reshape(batch, y_patches, 1, x_patches, 1).expand(
            batch, y_patches, patch_size, x_patches, patch_size
        )
        return expanded.contiguous().reshape(
            batch, y_patches * patch_size, x_patches * patch_size
        )

    def _sequence_from_patch_grid(self, *, grid: Tensor) -> Tensor:
        batch, y_patches, x_patches = grid.shape
        return grid.reshape(batch, y_patches * x_patches).contiguous()
