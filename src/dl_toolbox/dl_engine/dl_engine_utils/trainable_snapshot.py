"""CPU clone of an nn.Module plus optimizer, and the restore of that clone."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class TrainableSnapshot:
    def __init__(self, *, model: dict[str, Tensor], optimizer: dict) -> None:
        if not isinstance(model, dict):
            raise TypeError(f"model must be a dict, got {type(model)!r}")
        if not isinstance(optimizer, dict):
            raise TypeError(f"optimizer must be a dict, got {type(optimizer)!r}")
        self.model = model
        self.optimizer = optimizer


def snapshot_trainable(
    *, model: nn.Module, optimizer: Optimizer
) -> TrainableSnapshot:
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)!r}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
    return TrainableSnapshot(
        model=_cpu_state_dict(model),
        optimizer=deepcopy(optimizer.state_dict()),
    )


def restore_trainable(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    snapshot: TrainableSnapshot,
    device: torch.device,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)!r}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
    if not isinstance(snapshot, TrainableSnapshot):
        raise TypeError(
            f"snapshot must be TrainableSnapshot, got {type(snapshot)!r}"
        )
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be torch.device, got {type(device)!r}")
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
