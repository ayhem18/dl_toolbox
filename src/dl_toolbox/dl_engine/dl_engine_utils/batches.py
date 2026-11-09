"""Cycle a DataLoader and move prepared tensors onto a device."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader


def cycling_batches(*, loader: DataLoader, num_iter: int) -> Iterator[Any]:
    if not isinstance(loader, DataLoader):
        raise TypeError(f"loader must be DataLoader, got {type(loader)!r}")
    if isinstance(num_iter, bool) or not isinstance(num_iter, int) or num_iter < 1:
        raise ValueError(f"num_iter must be an int >= 1, got {num_iter!r}")
    iterator = iter(loader)
    for _ in range(num_iter):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                yield next(iterator)
            except StopIteration as exc:
                raise ValueError("DataLoader is empty") from exc


def mapping_to_device(
    *, mapping: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"mapping must be a mapping, got {type(mapping)!r}")
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be torch.device, got {type(device)!r}")
    moved: dict[str, Any] = {}
    keys = list(mapping)
    n_keys = len(keys)
    for i in range(n_keys):
        key = keys[i]
        value = mapping[key]
        if isinstance(value, Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
