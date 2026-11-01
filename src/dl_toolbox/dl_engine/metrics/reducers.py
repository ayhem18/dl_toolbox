"""Named reductions over a concatenated per-item metric vector."""

from __future__ import annotations

from torch import Tensor


def mean_items(values: Tensor) -> Tensor:
    return values.mean()


def std_items(values: Tensor) -> Tensor:
    if values.numel() <= 1:
        return values.new_zeros(())
    return values.std(unbiased=False)


def min_items(values: Tensor) -> Tensor:
    return values.min()


def max_items(values: Tensor) -> Tensor:
    return values.max()
