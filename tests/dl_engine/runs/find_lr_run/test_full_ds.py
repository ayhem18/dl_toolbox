from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.dl_engine_utils.trainable_snapshot import snapshot_trainable
from dl_toolbox.dl_engine.runs.lr_finder_run.range_test import run_full_ds_range_test
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrError,
    FindLrSpec,
    FindLrSteps,
    log_spaced_lrs,
)
from tests.dl_engine.runs.find_lr_run.support import (
    cpu_runtime,
    finder_steps,
    full_ds_spec,
    net_optimizer_loader,
)
from tests.dl_engine.runs.support import LinearBinary, forward_fn, loss_fn, prepare_tensors


def test_records_finite_candidates() -> None:
    net, optimizer, loader = net_optimizer_loader()
    spec = full_ds_spec(num_iter=4, num_iterations_per_lr=1)
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    lrs, losses = run_full_ds_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=finder_steps(),
        spec=spec,
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
        snapshot=snapshot,
    )
    expected = log_spaced_lrs(spec)
    assert lrs == expected
    assert len(losses) == spec.num_iter
    for i in range(len(losses)):
        assert math.isfinite(losses[i])


def test_resets_weights_before_each_candidate() -> None:
    starts: list[Tensor] = []

    def forward(net: LinearBinary, batch: dict[str, Tensor]) -> Tensor:
        starts.append(net.fc.weight.detach().cpu().clone())
        return net(batch["input"])

    net, optimizer, loader = net_optimizer_loader(n=8, batch_size=8)
    spec = FindLrSpec.full_ds(
        start_lr=1e-2,
        end_lr=1e-1,
        num_iter=2,
        num_iterations_per_lr=1,
    )
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    run_full_ds_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward,
            loss_fn=loss_fn,
        ),
        spec=spec,
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
        snapshot=snapshot,
    )
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

    net, optimizer, loader = net_optimizer_loader(n=8, batch_size=4)
    spec = FindLrSpec.full_ds(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=2,
        num_iterations_per_lr=3,
    )
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    run_full_ds_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward,
            loss_fn=loss_fn,
        ),
        spec=spec,
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
        snapshot=snapshot,
    )
    assert n_train["n"] == 12
    assert n_eval["n"] == 4


def test_empty_loader_errors() -> None:
    net = LinearBinary()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    with pytest.raises(FindLrError, match="train loader is empty"):
        run_full_ds_range_test(
            net=net,
            optimizer=optimizer,
            train_loader=loader,
            steps=finder_steps(),
            spec=full_ds_spec(num_iter=2, num_iterations_per_lr=1),
            device=torch.device("cpu"),
            runtime=cpu_runtime(),
            snapshot=snapshot,
        )


def test_non_finite_candidate_is_skipped_later_lrs_run() -> None:
    step = {"i": 0}

    def exploding_first_then_ok(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
        i = step["i"]
        step["i"] += 1
        if i == 0:
            return output.sum() * 0.0 + float("inf")
        return output.sum() * 0.0 + 0.5

    net, optimizer, loader = net_optimizer_loader(n=8, batch_size=4)
    spec = FindLrSpec.full_ds(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=3,
        num_iterations_per_lr=1,
    )
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    lrs, losses = run_full_ds_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward_fn,
            loss_fn=exploding_first_then_ok,
        ),
        spec=spec,
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
        snapshot=snapshot,
    )
    expected = log_spaced_lrs(spec)
    assert lrs == expected[1:]
    assert losses[0] == pytest.approx(0.5)
    assert losses[1] == pytest.approx(0.5)
    for i in range(len(losses)):
        assert math.isfinite(losses[i])


def test_value_error_from_loss_skips_candidate_later_lrs_run() -> None:
    step = {"i": 0}

    def raising_first_then_ok(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
        i = step["i"]
        step["i"] += 1
        if i == 0:
            raise ValueError("tensor contains NaN or inf")
        return output.sum() * 0.0 + 0.5

    net, optimizer, loader = net_optimizer_loader(n=8, batch_size=4)
    spec = FindLrSpec.full_ds(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=3,
        num_iterations_per_lr=1,
    )
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    lrs, losses = run_full_ds_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward_fn,
            loss_fn=raising_first_then_ok,
        ),
        spec=spec,
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
        snapshot=snapshot,
    )
    expected = log_spaced_lrs(spec)
    assert lrs == expected[1:]
    assert losses[0] == pytest.approx(0.5)
    assert losses[1] == pytest.approx(0.5)
    for i in range(len(losses)):
        assert math.isfinite(losses[i])
