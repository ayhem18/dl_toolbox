"""Suggested LRs from a recorded range-test curve."""

from __future__ import annotations

import math
from collections.abc import Sequence

from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError


def suggest_lrs(
    lrs: Sequence[float],
    losses: Sequence[float],
) -> dict[str, float]:
    n = len(lrs)
    if n != len(losses):
        raise LrFinderError("lrs and losses must be the same length")
    if n < 2:
        raise LrFinderError("need at least 2 curve points to suggest LRs")
    min_i = _argmin(losses)
    if min_i < 1:
        raise LrFinderError(
            "loss minimum is the first recorded point; widen the LR range"
        )
    steepest_i = _steepest_index(lrs, losses, stop=min_i)
    min_lr = float(lrs[min_i])
    return {
        "steepest": float(lrs[steepest_i]),
        "before_min": min_lr / 10.0,
        "min_loss_lr": min_lr,
    }


def _argmin(losses: Sequence[float]) -> int:
    n = len(losses)
    best_i = 0
    best = float(losses[0])
    for i in range(1, n):
        value = float(losses[i])
        if value < best:
            best = value
            best_i = i
    return best_i


def _steepest_index(
    lrs: Sequence[float],
    losses: Sequence[float],
    *,
    stop: int,
) -> int:
    best_i = 0
    best_grad = float("inf")
    found = False
    for i in range(stop):
        left = math.log10(float(lrs[i]))
        right = math.log10(float(lrs[i + 1]))
        width = right - left
        if width <= 0.0:
            continue
        grad = (float(losses[i + 1]) - float(losses[i])) / width
        if not found or grad < best_grad:
            best_grad = grad
            best_i = i
            found = True
    if not found:
        raise LrFinderError("cannot compute d(loss)/d(log10 lr) before the minimum")
    return best_i
