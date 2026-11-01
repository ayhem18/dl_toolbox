from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.lr_finder import LrFinderError, LrFinderSpec, run_range_test
from tests.dl_engine.lr_finder.support import LinearBinary, make_inputs, make_loader


def test_restores_weights_and_records_increasing_lrs() -> None:
    model = LinearBinary()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    lrs, losses = run_range_test(make_inputs(model=model))
    assert len(lrs) >= 2
    assert len(lrs) == len(losses)
    assert lrs[0] < lrs[-1]
    after = model.state_dict()
    keys = list(before)
    n = len(keys)
    for i in range(n):
        key = keys[i]
        assert torch.equal(before[key], after[key])


def test_empty_loader_errors() -> None:
    x = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    with pytest.raises(LrFinderError, match="train loader is empty"):
        run_range_test(make_inputs(loader=loader))


def test_diverge_stops_before_keeping_the_spike() -> None:
    step = {"i": 0}

    def exploding_loss(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
        values = (1.0, 1.05, 8.0, 8.0)
        i = step["i"]
        step["i"] += 1
        return output.sum() * 0.0 + values[min(i, 3)]

    spec = LrFinderSpec(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=4,
        smooth_f=0.0,
        diverge_th=4.0,
    )
    lrs, losses = run_range_test(
        make_inputs(spec=spec, loss=exploding_loss, loader=make_loader())
    )
    assert len(lrs) == 2
    assert losses[0] == pytest.approx(1.0)
    assert losses[1] == pytest.approx(1.05)
