# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import math

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.lr_finder import (
    LrFinderError,
    LrFinderSpec,
    LrFinderStep,
    run_full_ds_range_test,
)
from dl_toolbox.dl_engine.lr_finder.spec import log_spaced_lrs
from tests.dl_engine.lr_finder.support import (
    LinearBinary,
    loss_fn,
    make_inputs,
    make_loader,
    prepare_tensors,
)


def test_restores_weights_and_records_every_candidate() -> None:
    model = LinearBinary()
    optimizer_lr = 1e-4
    inputs = make_inputs(model=model)
    assert inputs.optimizer.param_groups[0]["lr"] == pytest.approx(optimizer_lr)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    lrs, losses = run_full_ds_range_test(inputs)
    spec = inputs.config.spec
    expected = log_spaced_lrs(spec)
    assert lrs == expected
    assert len(losses) == spec.num_iter
    for i in range(len(losses)):
        assert math.isfinite(losses[i])
    after = model.state_dict()
    keys = list(before)
    n = len(keys)
    for i in range(n):
        key = keys[i]
        assert torch.equal(before[key], after[key])
    assert inputs.optimizer.param_groups[0]["lr"] == pytest.approx(optimizer_lr)


def test_resets_weights_before_each_candidate() -> None:
    starts: list[Tensor] = []

    def forward(net: LinearBinary, batch: dict[str, Tensor]) -> Tensor:
        starts.append(net.fc.weight.detach().cpu().clone())
        return net(batch["input"])

    spec = LrFinderSpec(
        start_lr=1e-2,
        end_lr=1e-1,
        num_iter=2,
        smooth_f=0.05,
        diverge_th=4.0,
    )
    inputs = make_inputs(spec=spec, loader=make_loader(n=8, batch_size=8))
    inputs.step = LrFinderStep(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward,
        loss_fn=loss_fn,
    )
    run_full_ds_range_test(inputs)
    assert len(starts) == 4
    assert torch.equal(starts[0], starts[2])
    assert not torch.equal(starts[0], starts[1])


def test_trains_requested_passes_then_evals_once() -> None:
    n_train = {"n": 0}
    n_eval = {"n": 0}

    def forward(net: LinearBinary, batch: dict[str, Tensor]) -> Tensor:
        if net.training:
            n_train["n"] += 1
        else:
            n_eval["n"] += 1
        return net(batch["input"])

    spec = LrFinderSpec(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=2,
        smooth_f=0.05,
        diverge_th=4.0,
    )
    inputs = make_inputs(
        spec=spec,
        loader=make_loader(n=8, batch_size=4),
        num_iterations_per_lr=3,
    )
    inputs.step = LrFinderStep(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward,
        loss_fn=loss_fn,
    )
    run_full_ds_range_test(inputs)
    assert n_train["n"] == 12
    assert n_eval["n"] == 4


def test_empty_loader_errors() -> None:
    x = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    with pytest.raises(LrFinderError, match="train loader is empty"):
        run_full_ds_range_test(make_inputs(loader=loader))


def test_diverging_lr_does_not_abort_later_candidates() -> None:
    step = {"i": 0}

    def exploding_first_then_ok(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
        i = step["i"]
        step["i"] += 1
        if i == 0:
            return output.sum() * 0.0 + float("inf")
        return output.sum() * 0.0 + 0.5

    spec = LrFinderSpec(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=3,
        smooth_f=0.05,
        diverge_th=4.0,
    )
    lrs, losses = run_full_ds_range_test(
        make_inputs(
            spec=spec,
            loss=exploding_first_then_ok,
            loader=make_loader(n=8, batch_size=4),
        )
    )
    assert lrs == log_spaced_lrs(spec)
    assert math.isinf(losses[0])
    assert losses[1] == pytest.approx(0.5)
    assert losses[2] == pytest.approx(0.5)
