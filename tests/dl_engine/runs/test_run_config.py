from __future__ import annotations

import inspect

import pytest
import torch
from pydantic import ValidationError
from torch.optim.lr_scheduler import LambdaLR

from dl_toolbox.dl_engine.runs.common.run_config import (
    Accelerator,
    EarlyStoppingConfig,
    MonitorConfig,
    OptimizationConfig,
    RunLogConfig,
    Strategy,
    TrainerConfig,
)


def _cpu_ok(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "accelerator": Accelerator.CPU,
        "devices": (),
        "strategy": Strategy.SINGLE_DEVICE,
        "precision": 32,
        "max_epochs": 1,
        "accumulate_grad_batches": 1,
        "enable_progress_bar": False,
        "early_stopping": EarlyStoppingConfig.none(),
    }
    values.update(overrides)
    return values


def test_legal_cpu_row() -> None:
    config = TrainerConfig(**_cpu_ok())
    assert config.accelerator == Accelerator.CPU
    assert config.devices == ()
    assert config.strategy == Strategy.SINGLE_DEVICE


def test_legal_one_gpu_row() -> None:
    config = TrainerConfig(
        accelerator=Accelerator.GPU,
        devices=(3,),
        strategy=Strategy.SINGLE_DEVICE,
        precision=32,
        max_epochs=1,
        accumulate_grad_batches=1,
        enable_progress_bar=False,
        early_stopping=EarlyStoppingConfig.none(),
    )
    assert config.devices == (3,)


def test_legal_ddp_row() -> None:
    config = TrainerConfig(
        accelerator=Accelerator.GPU,
        devices=(0, 3, 5),
        strategy=Strategy.DDP,
        precision="16-mixed",
        max_epochs=2,
        accumulate_grad_batches=4,
        enable_progress_bar=True,
        early_stopping=EarlyStoppingConfig(
            name=EarlyStoppingConfig.PATIENCE, patience=3, min_delta=1e-5
        ),
    )
    assert config.devices == (0, 3, 5)
    assert config.strategy == Strategy.DDP
    assert config.early_stopping.patience == 3
    assert config.early_stopping.min_delta == 1e-5


def test_devices_count_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(**_cpu_ok(accelerator=Accelerator.GPU, devices=4))


def test_devices_list_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(**_cpu_ok(accelerator=Accelerator.GPU, devices=[0]))


def test_gpu_empty_devices_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(
            **_cpu_ok(
                accelerator=Accelerator.GPU,
                devices=(),
                strategy=Strategy.SINGLE_DEVICE,
            )
        )


def test_cpu_with_index_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(**_cpu_ok(devices=(0,)))


def test_ddp_with_one_device_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(
            accelerator=Accelerator.GPU,
            devices=(0,),
            strategy=Strategy.DDP,
            precision=32,
            max_epochs=1,
            accumulate_grad_batches=1,
            enable_progress_bar=False,
            early_stopping=EarlyStoppingConfig.none(),
        )


def test_single_device_with_several_gpus_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(
            accelerator=Accelerator.GPU,
            devices=(0, 1),
            strategy=Strategy.SINGLE_DEVICE,
            precision=32,
            max_epochs=1,
            accumulate_grad_batches=1,
            enable_progress_bar=False,
            early_stopping=EarlyStoppingConfig.none(),
        )


def test_duplicate_gpu_indices_are_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(
            accelerator=Accelerator.GPU,
            devices=(0, 0),
            strategy=Strategy.DDP,
            precision=32,
            max_epochs=1,
            accumulate_grad_batches=1,
            enable_progress_bar=False,
            early_stopping=EarlyStoppingConfig.none(),
        )


def test_cpu_ddp_is_illegal() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(**_cpu_ok(strategy=Strategy.DDP))


def test_max_epochs_and_accumulate_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        TrainerConfig(**_cpu_ok(max_epochs=0))
    with pytest.raises(ValidationError):
        TrainerConfig(**_cpu_ok(accumulate_grad_batches=0))


def test_trainer_config_has_no_defaults() -> None:
    params = inspect.signature(TrainerConfig).parameters
    for name, param in params.items():
        if name == "self":
            continue
        assert param.default is inspect.Parameter.empty, name


def test_scheduler_none_is_illegal() -> None:
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    with pytest.raises((TypeError, ValidationError)):
        OptimizationConfig(optimizer=optimizer, scheduler=None)


def test_scheduler_absent_identity() -> None:
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    config = OptimizationConfig(
        optimizer=optimizer, scheduler=OptimizationConfig.ABSENT
    )
    assert config.scheduler is OptimizationConfig.ABSENT


def test_scheduler_live_instance() -> None:
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    config = OptimizationConfig(optimizer=optimizer, scheduler=scheduler)
    assert config.scheduler is scheduler


def test_early_stopping_none_has_no_patience_slot() -> None:
    config = EarlyStoppingConfig.none()
    assert config.name == EarlyStoppingConfig.NONE
    assert "patience" not in vars(config)
    assert "min_delta" not in vars(config)


def test_early_stopping_none_via_init_is_illegal() -> None:
    with pytest.raises(ValueError):
        EarlyStoppingConfig(
            name=EarlyStoppingConfig.NONE, patience=1, min_delta=0.0
        )


def test_run_log_payload_must_be_json_object() -> None:
    with pytest.raises(ValidationError):
        RunLogConfig(payload=["not", "an", "object"], filename="run")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RunLogConfig(payload={"task": "unit"}, filename="")
    config = RunLogConfig(payload={"task": "unit"}, filename="run_config")
    assert config.filename == "run_config"


def test_monitor_mode_must_be_min_or_max() -> None:
    with pytest.raises(ValidationError):
        MonitorConfig(key="loss/val/epoch/mean", mode="auto")  # type: ignore[arg-type]
