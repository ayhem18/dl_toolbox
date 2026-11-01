"""Cycle the train loader and move a prepared batch onto the finder device."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError
from dl_toolbox.dl_engine.lr_finder.step import LrFinderStep


def cycling_batches(loader: DataLoader, num_iter: int) -> Iterator[Any]:
    iterator = iter(loader)
    for _ in range(num_iter):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                yield next(iterator)
            except StopIteration as exc:
                raise LrFinderError("train loader is empty") from exc


def prepare_on_device(
    step: LrFinderStep,
    raw_batch: Any,
    device: torch.device,
) -> dict[str, Any]:
    prepared = step.prepare_tensors_fn(raw_batch)
    if not isinstance(prepared, dict):
        raise TypeError(
            f"prepare_tensors_fn must return a dictionary, got {type(prepared)}"
        )
    moved: dict[str, Any] = {}
    keys = list(prepared)
    n_keys = len(keys)
    for i in range(n_keys):
        key = keys[i]
        value = prepared[key]
        if isinstance(value, Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    n = len(optimizer.param_groups)
    for i in range(n):
        optimizer.param_groups[i]["lr"] = float(lr)
