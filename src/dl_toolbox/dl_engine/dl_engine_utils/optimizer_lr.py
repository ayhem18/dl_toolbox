"""Write one learning rate onto every optimizer param group."""

from __future__ import annotations

from torch.optim import Optimizer


def set_optimizer_lr(*, optimizer: Optimizer, lr: float) -> None:
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
    if type(lr) is not float:
        raise TypeError(f"lr must be a float, got {type(lr)!r}")
    n = len(optimizer.param_groups)
    for i in range(n):
        optimizer.param_groups[i]["lr"] = lr
