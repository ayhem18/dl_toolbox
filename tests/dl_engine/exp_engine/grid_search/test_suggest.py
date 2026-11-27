from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.exp_engine.grid_search import (
    suggest_lin_values,
    suggest_log_learning_rates,
)


def test_log_grid_includes_both_ends() -> None:
    rates = suggest_log_learning_rates(start_lr=1e-4, end_lr=1e-2, num=3)
    assert rates[0] == pytest.approx(1e-4)
    assert rates[-1] == pytest.approx(1e-2)
    assert rates[1] == pytest.approx(1e-3)


def test_lin_grid_includes_both_ends() -> None:
    values = suggest_lin_values(start=0.1, end=0.3, num=3)
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(0.2)
    assert values[2] == pytest.approx(0.3)


def test_rejects_equal_log_ends() -> None:
    with pytest.raises(ValueError, match="differ"):
        suggest_log_learning_rates(start_lr=1e-3, end_lr=1e-3, num=3)
