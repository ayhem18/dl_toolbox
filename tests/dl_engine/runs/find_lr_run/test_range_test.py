from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.runs.lr_finder_run.range_test import run_range_test
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import FindLrError, FindLrSpec, FindLrSteps
from tests.dl_engine.runs.find_lr_run.support import (
    cpu_runtime,
    finder_steps,
    net_optimizer_loader,
)
from tests.dl_engine.runs.support import LinearBinary, forward_fn, prepare_tensors


def test_records_increasing_lrs() -> None:
    net, optimizer, loader = net_optimizer_loader()
    lrs, losses = run_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=finder_steps(),
        spec=FindLrSpec.range(
            start_lr=1e-4,
            end_lr=1e-1,
            num_iter=4,
            smooth_f=0.05,
            diverge_th=4.0,
        ),
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
    )
    assert len(lrs) >= 2
    assert len(lrs) == len(losses)
    assert lrs[0] < lrs[-1]


def test_empty_loader_errors() -> None:
    net = LinearBinary()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    with pytest.raises(FindLrError, match="train loader is empty"):
        run_range_test(
            net=net,
            optimizer=optimizer,
            train_loader=loader,
            steps=finder_steps(),
            spec=FindLrSpec.range(
                start_lr=1e-4,
                end_lr=1e-1,
                num_iter=4,
                smooth_f=0.05,
                diverge_th=4.0,
            ),
            device=torch.device("cpu"),
            runtime=cpu_runtime(),
        )


def test_diverge_stops_before_keeping_the_spike() -> None:
    step = {"i": 0}

    def exploding_loss(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
        values = (1.0, 1.05, 8.0, 8.0)
        i = step["i"]
        step["i"] += 1
        return output.sum() * 0.0 + values[min(i, 3)]

    net, optimizer, loader = net_optimizer_loader()
    lrs, losses = run_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward_fn,
            loss_fn=exploding_loss,
        ),
        spec=FindLrSpec.range(
            start_lr=1e-4,
            end_lr=1e-1,
            num_iter=4,
            smooth_f=0.0,
            diverge_th=4.0,
        ),
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
    )
    assert len(lrs) == 2
    assert losses[0] == pytest.approx(1.0)
    assert losses[1] == pytest.approx(1.05)


def test_value_error_from_loss_stops_the_walk() -> None:
    step = {"i": 0}

    def raising_loss(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
        i = step["i"]
        step["i"] += 1
        if i >= 2:
            raise ValueError("tensor contains NaN or inf")
        return output.sum() * 0.0 + 1.0

    net, optimizer, loader = net_optimizer_loader()
    lrs, losses = run_range_test(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward_fn,
            loss_fn=raising_loss,
        ),
        spec=FindLrSpec.range(
            start_lr=1e-4,
            end_lr=1e-1,
            num_iter=4,
            smooth_f=0.0,
            diverge_th=4.0,
        ),
        device=torch.device("cpu"),
        runtime=cpu_runtime(),
    )
    assert len(lrs) == 2
    assert losses[0] == pytest.approx(1.0)
    assert losses[1] == pytest.approx(1.0)
