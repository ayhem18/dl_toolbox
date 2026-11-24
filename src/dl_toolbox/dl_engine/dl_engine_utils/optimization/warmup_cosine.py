"""Linear warmup then half-cycle cosine, as in official FCMAE ``adjust_learning_rate``."""

from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def warmup_cosine_lambda(
    epoch: int,
    *,
    warmup_epochs: int,
    max_epochs: int,
    min_lr: float,
    base_lr: float,
) -> float:
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
    if min_lr < 0.0:
        raise ValueError(f"min_lr must be >= 0, got {min_lr}")
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if epoch < warmup_epochs:
        if warmup_epochs == 0:
            return 1.0
        return float(epoch) / float(warmup_epochs)
    cosine_epochs = max_epochs - warmup_epochs
    if cosine_epochs <= 0:
        return min_lr / base_lr
    progress = float(epoch - warmup_epochs) / float(cosine_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = min_lr / base_lr
    return floor + (1.0 - floor) * cosine


def make_warmup_cosine_scheduler(
    *,
    warmup_epochs: int,
    max_epochs: int,
    min_lr: float,
    base_lr: float,
) -> Callable[[Optimizer], LRScheduler]:
    def factory(optimizer: Optimizer) -> LRScheduler:
        def lr_lambda(epoch: int) -> float:
            return warmup_cosine_lambda(
                epoch,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
                min_lr=min_lr,
                base_lr=base_lr,
            )

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    return factory
