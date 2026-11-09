from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from dl_toolbox.base.input_output.json_io import dump_object, flatten_object, JsonTextForm
from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    LogStream,
    Logger,
)
from dl_toolbox.dl_engine.runs.common.run_config import (
    FitSteps,
    MonitorConfig,
    OptimizationConfig,
    RunLogConfig,
)
from dl_toolbox.dl_engine.runs.common.run_loop import (
    TRACK_PROG_BAR,
    log_epoch_row,
    prepare_batch,
)
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import TrainOnlyMetrics
from dl_toolbox.dl_engine.runs.train_only_run.train_only_module import TrainOnlyLightningModule
from tests.dl_engine.runs.support import RecordingLogger


class _FakeTrainer:
    def __init__(
        self,
        *,
        is_global_zero: bool,
        epoch: int,
        callback_metrics: dict[str, Tensor] | None = None,
    ) -> None:
        self.is_global_zero = is_global_zero
        self.current_epoch = epoch
        self.callback_metrics = callback_metrics or {}
        self.sanity_checking = False
        self.global_rank = 0 if is_global_zero else 1
        self.world_size = 1 if is_global_zero else 4


class _CpuModule(TrainOnlyLightningModule):
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


def _identity_prepare(raw_batch: Any) -> Mapping[str, Any]:
    return raw_batch


def _forward(net: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    del net, batch
    return torch.zeros(1)


def _loss(output: Tensor, batch: Mapping[str, Any]) -> Tensor:
    del batch
    return output.sum()


def _metric_inputs(output: Tensor, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
    del batch
    return {"output": output}


def _module(
    logger: Logger | None = None,
    *,
    scheduler: object | None = None,
) -> _CpuModule:
    net = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    if scheduler is None:
        scheduler = OptimizationConfig.ABSENT
    return _CpuModule(
        net=net,
        steps=FitSteps(
            prepare_tensors_fn=_identity_prepare,
            forward_fn=_forward,
            loss_fn=_loss,
            build_metric_inputs_fn=_metric_inputs,
        ),
        metrics=TrainOnlyMetrics(
            train_metrics=[],
            monitor=MonitorConfig(key="loss/train/epoch/mean", mode="min"),
        ),
        optimization=OptimizationConfig(optimizer=optimizer, scheduler=scheduler),
        logger=logger or RecordingLogger(),
        run_log=RunLogConfig(payload={"task": "unit"}, filename="run_config"),
    )


def test_no_validation_methods() -> None:
    owned = TrainOnlyLightningModule.__dict__
    assert "validation_step" not in owned
    assert "on_validation_epoch_start" not in owned
    assert "on_validation_epoch_end" not in owned


def test_on_fit_start_calls_setup_fit_create() -> None:
    logger = RecordingLogger()
    module = _module(logger)
    module._trainer = _FakeTrainer(is_global_zero=True, epoch=0)
    module.on_fit_start()
    assert logger.setup_calls == [(LogStream.FIT, LogExistence.CREATE)]
    assert logger.params == [flatten_object({"task": "unit"})]
    assert logger.texts[0][0] == "run_config"
    assert logger.texts[0][1] == dump_object(
        {"task": "unit"}, form=JsonTextForm.ASCII_PRETTY
    )


def test_on_fit_start_and_end_are_rank_zero_only() -> None:
    logger = RecordingLogger()
    module = _module(logger)
    module._trainer = _FakeTrainer(is_global_zero=False, epoch=0)
    module.on_fit_start()
    module.on_fit_end()
    assert logger.setup_calls == []
    assert logger.close_calls == 0


def test_train_epoch_end_logs_with_prog_bar() -> None:
    module = _module()
    module._trainer = _FakeTrainer(is_global_zero=True, epoch=1)
    logged: list[dict[str, Any]] = []

    def _log(name: str, value: float, **kwargs: Any) -> None:
        del name, value
        logged.append(kwargs)

    module.log = _log  # type: ignore[method-assign]
    log_epoch_row(
        module,
        {"loss/train/epoch/mean": 0.5},
        track=TRACK_PROG_BAR,
        expected_keys=["loss/train/epoch/mean"],
        epoch_logger=module._epoch_logger,
    )
    assert logged
    assert logged[0]["prog_bar"] is True
    assert logged[0]["sync_dist"] is True


def test_train_epoch_end_errors_when_monitor_missing() -> None:
    module = _module()
    module._trainer = _FakeTrainer(is_global_zero=True, epoch=0)
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.on_train_epoch_start()
    with pytest.raises(ValueError, match="monitor.key"):
        module.on_train_epoch_end()


def test_configure_optimizers_absent_and_live() -> None:
    module = _module()
    assert module.configure_optimizers() is module._optimization.optimizer
    net = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    with_sched = _module(scheduler=scheduler)
    configured = with_sched.configure_optimizers()
    assert isinstance(configured, dict)
    assert configured["lr_scheduler"]["interval"] == "epoch"
    assert configured["lr_scheduler"]["scheduler"] is scheduler


def test_module_init_has_no_defaults() -> None:
    params = inspect.signature(TrainOnlyLightningModule.__init__).parameters
    for name, param in params.items():
        if name == "self":
            continue
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_prepare_moves_tensors_and_leaves_strings() -> None:
    module = _module()
    image = torch.ones(1, 1, 4, 8)
    batch = {
        "image": image,
        "dcm_file_path": ["scan.dcm"],
    }
    prepared = prepare_batch(
        module, batch, prepare_tensors_fn=module._steps.prepare_tensors_fn
    )
    assert prepared["dcm_file_path"] is batch["dcm_file_path"]
    torch.testing.assert_close(prepared["image"], image)


def test_load_net_has_no_defaults() -> None:
    params = inspect.signature(TrainOnlyLightningModule.load_net).parameters
    for name, param in params.items():
        if name == "cls":
            continue
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_load_net_copies_inner_weights(tmp_path: Path) -> None:
    source = nn.Linear(4, 1)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(-1.5)
    shell = nn.Module()
    shell.add_module("net", source)
    path = tmp_path / "net.ckpt"
    torch.save({"state_dict": shell.state_dict()}, path)
    dest = nn.Linear(4, 1)
    loaded = TrainOnlyLightningModule.load_net(
        net=dest, file=path, map_location=torch.device("cpu")
    )
    assert loaded is dest
    assert torch.equal(dest.weight, source.weight)
    assert torch.equal(dest.bias, source.bias)


def test_load_net_errors_on_unexpected_keys(tmp_path: Path) -> None:
    extra = nn.Sequential(nn.Linear(4, 1), nn.Linear(1, 1))
    shell = nn.Module()
    shell.add_module("net", extra)
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": shell.state_dict()}, path)
    dest = nn.Linear(4, 1)
    try:
        TrainOnlyLightningModule.load_net(
            net=dest, file=path, map_location=torch.device("cpu")
        )
    except ValueError as exc:
        assert "state_dict" in str(exc)
        return
    raise AssertionError("expected ValueError")
