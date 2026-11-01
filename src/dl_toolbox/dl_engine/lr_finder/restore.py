"""Snapshot and restore model plus optimizer so the finder is not a fit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class TrainableSnapshot:
    model: dict[str, Tensor]
    optimizer: dict


def snapshot_trainable(model: nn.Module, optimizer: Optimizer) -> TrainableSnapshot:
    return TrainableSnapshot(
        model=_cpu_state_dict(model),
        optimizer=deepcopy(optimizer.state_dict()),
    )


def restore_trainable(
    model: nn.Module,
    optimizer: Optimizer,
    snapshot: TrainableSnapshot,
    device: torch.device,
) -> None:
    model.load_state_dict(snapshot.model)
    optimizer.load_state_dict(snapshot.optimizer)
    model.to(device)


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state = model.state_dict()
    keys = list(state)
    n = len(keys)
    copied: dict[str, Tensor] = {}
    for i in range(n):
        key = keys[i]
        copied[key] = state[key].detach().cpu().clone()
    return copied
