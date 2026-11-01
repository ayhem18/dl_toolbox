from __future__ import annotations

import torch

from dl_toolbox.dl_engine.metrics.reducers import (
    max_items,
    mean_items,
    min_items,
    std_items,
)


def test_mean_min_max_on_vector() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(mean_items(values)) == 2.5
    assert float(min_items(values)) == 1.0
    assert float(max_items(values)) == 4.0


def test_std_is_population_and_zero_for_one_item() -> None:
    values = torch.tensor([1.0, 3.0])
    assert float(std_items(values)) == 1.0
    single = torch.tensor([4.0])
    assert float(std_items(single)) == 0.0
