from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.runs.lr_finder_run.range_test import (
    suggest_lrs,
    suggest_min_loss_lr,
)
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import FindLrError


def test_bowl_curve() -> None:
    n = 11
    lrs: list[float] = [0.0] * n
    losses: list[float] = [0.0] * n
    for i in range(n):
        lrs[i] = 10 ** (-6 + i * 0.5)
        losses[i] = float((i - 6) * (i - 6))
    suggested = suggest_lrs(lrs, losses)
    assert suggested["min_loss_lr"] == lrs[6]
    assert suggested["before_min"] == lrs[6] / 10.0
    assert suggested["steepest"] < suggested["min_loss_lr"]


def test_minimum_at_the_first_point_fails() -> None:
    with pytest.raises(FindLrError, match="first recorded point"):
        suggest_lrs((1e-4, 1e-3), (0.1, 0.2))


def test_length_mismatch_fails() -> None:
    with pytest.raises(FindLrError, match="same length"):
        suggest_lrs((1e-4, 1e-3, 1e-2), (0.2, 0.1))


def test_min_loss_lr_allows_first_point() -> None:
    suggested = suggest_min_loss_lr((1e-4, 1e-3, 1e-2), (0.1, 0.4, 0.3))
    assert suggested == {"min_loss_lr": 1e-4}


def test_suggest_rejects_non_finite() -> None:
    with pytest.raises(FindLrError, match="finite"):
        suggest_min_loss_lr((1e-4, 1e-3), (0.1, float("inf")))
    with pytest.raises(FindLrError, match="finite"):
        suggest_lrs((1e-4, 1e-3), (float("nan"), 0.2))
