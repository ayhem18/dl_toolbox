from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream
from dl_toolbox.dl_engine.runs.common.run_config import Accelerator
from dl_toolbox.dl_engine.runs.lr_finder_run.find_lr_run import FindLrRun, find_lr
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrError,
    FindLrRuntime,
    FindLrSteps,
    resolve_device,
)
from tests.dl_engine.runs.find_lr_run.support import (
    FindLrLogger,
    artifacts_directory,
    cpu_runtime,
    finder_steps,
    full_ds_spec,
    net_optimizer_loader,
    range_spec,
    run_log,
)
from tests.dl_engine.runs.support import LinearBinary, steps as fit_steps


def test_find_lr_has_no_defaults() -> None:
    params = inspect.signature(find_lr).parameters
    for name, param in params.items():
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_find_lr_writes_curve_plot_and_closes(tmp_path: Path) -> None:
    torch.manual_seed(3)
    logger = FindLrLogger()
    net, optimizer, loader = net_optimizer_loader()
    before = {key: value.detach().clone() for key, value in net.state_dict().items()}
    optimizer_lr = optimizer.param_groups[0]["lr"]
    artifacts = artifacts_directory(tmp_path)
    spec = range_spec()
    result = find_lr(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=finder_steps(),
        spec=spec,
        runtime=cpu_runtime(),
        logger=logger,
        artifacts=artifacts,
        run_log=run_log(),
    )
    assert result.kind == spec.kind
    payload = read_object(file=result.curve)
    assert payload["kind"] == spec.kind
    assert len(payload["lrs"]) >= 2
    assert "steepest" in payload["suggested"]
    assert result.plot.path.name == FindLrRun.PLOT_FILENAME
    assert result.curve.path.name == FindLrRun.CURVE_FILENAME
    after = net.state_dict()
    keys = list(before)
    for i in range(len(keys)):
        key = keys[i]
        assert torch.equal(before[key], after[key])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(optimizer_lr)
    assert logger.setup_calls == [(LogStream.FIT, LogExistence.CREATE)]
    assert logger.close_calls == 1
    curve_steps = [
        mapping
        for mapping, _step in logger.scalars
        if FindLrRun.TAG_LR in mapping and FindLrRun.TAG_LOSS in mapping
    ]
    assert len(curve_steps) == len(payload["lrs"])
    suggested_keys = {
        key
        for mapping, _step in logger.scalars
        for key in mapping
        if key.startswith(FindLrRun.SUGGESTED_PREFIX)
    }
    assert f"{FindLrRun.SUGGESTED_PREFIX}steepest" in suggested_keys
    assert f"{FindLrRun.SUGGESTED_PREFIX}before_min" in suggested_keys
    assert f"{FindLrRun.SUGGESTED_PREFIX}min_loss_lr" in suggested_keys
    assert logger.images[0][0] == FindLrRun.TAG_PLOT
    assert logger.tables[0][0] == FindLrRun.TAG_TABLE
    assert len(logger.tables[0][1]) == len(payload["lrs"])


def test_find_lr_full_ds_writes_min_loss_lr(tmp_path: Path) -> None:
    torch.manual_seed(3)
    logger = FindLrLogger()
    net, optimizer, loader = net_optimizer_loader()
    spec = full_ds_spec(num_iter=4, num_iterations_per_lr=1)
    result = find_lr(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=finder_steps(),
        spec=spec,
        runtime=cpu_runtime(),
        logger=logger,
        artifacts=artifacts_directory(tmp_path),
        run_log=run_log(),
    )
    payload = read_object(file=result.curve)
    assert payload["kind"] == spec.kind
    assert payload["lrs"] == list(result.lrs)
    assert payload["suggested"] == {"min_loss_lr": result.suggested["min_loss_lr"]}
    suggested_keys = {
        key
        for mapping, _step in logger.scalars
        for key in mapping
        if key.startswith(FindLrRun.SUGGESTED_PREFIX)
    }
    assert suggested_keys == {f"{FindLrRun.SUGGESTED_PREFIX}min_loss_lr"}
    assert logger.close_calls == 1


def test_find_lr_closes_logger_on_failure(tmp_path: Path) -> None:
    x = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    logger = FindLrLogger()
    net = LinearBinary()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    with pytest.raises(FindLrError, match="empty"):
        find_lr(
            net=net,
            optimizer=optimizer,
            train_loader=loader,
            steps=finder_steps(),
            spec=range_spec(),
            runtime=cpu_runtime(),
            logger=logger,
            artifacts=artifacts_directory(tmp_path),
            run_log=run_log(),
        )
    assert logger.setup_calls == [(LogStream.FIT, LogExistence.CREATE)]
    assert logger.close_calls == 1
    assert not (tmp_path / FindLrRun.CURVE_FILENAME).exists()


def test_from_fit_steps(tmp_path: Path) -> None:
    torch.manual_seed(3)
    net, optimizer, loader = net_optimizer_loader()
    result = find_lr(
        net=net,
        optimizer=optimizer,
        train_loader=loader,
        steps=FindLrSteps.from_fit_steps(fit_steps()),
        spec=range_spec(),
        runtime=cpu_runtime(),
        logger=FindLrLogger(),
        artifacts=artifacts_directory(tmp_path),
        run_log=run_log(),
    )
    assert len(result.lrs) >= 2


def test_resolve_device_gpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FindLrRuntime(
        accelerator=Accelerator.GPU,
        precision=32,
        enable_progress_bar=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(FindLrError, match="CUDA"):
        resolve_device(runtime)
