"""Shared run inputs. No defaults. No None sentinels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dl_toolbox.base.input_output.json_io import dump_object, JsonTextForm
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_STRICT_LIVE = ConfigDict(
    extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
)


class Accelerator:
    GPU = "gpu"
    CPU = "cpu"


class Strategy:
    SINGLE_DEVICE = "single_device"
    DDP = "ddp"


class _SchedulerAbsent:
    def __repr__(self) -> str:
        return "OptimizationConfig.ABSENT"


class FitSteps:
    def __init__(
        self,
        *,
        prepare_tensors_fn: Callable[[Any], Mapping[str, Any]],
        forward_fn: Callable[[nn.Module, Mapping[str, Any]], Tensor],
        loss_fn: Callable[[Tensor, Mapping[str, Any]], Tensor],
        build_metric_inputs_fn: Callable[
            [Tensor, Mapping[str, Any]], Mapping[str, Tensor]
        ],
    ) -> None:
        if not callable(prepare_tensors_fn):
            raise TypeError("prepare_tensors_fn must be callable")
        if not callable(forward_fn):
            raise TypeError("forward_fn must be callable")
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        if not callable(build_metric_inputs_fn):
            raise TypeError("build_metric_inputs_fn must be callable")
        self.prepare_tensors_fn = prepare_tensors_fn
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.build_metric_inputs_fn = build_metric_inputs_fn


class MonitorConfig(BaseModel):
    model_config = _STRICT

    key: str
    mode: Literal["min", "max"]

    @field_validator("key")
    @classmethod
    def _key_non_empty(cls, value: str) -> str:
        return parse_non_empty_str(value, label="monitor.key")


class OptimizationConfig:
    ABSENT = _SchedulerAbsent()

    def __init__(self, *, optimizer: Optimizer, scheduler: object) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
        if scheduler is not OptimizationConfig.ABSENT and not isinstance(
            scheduler, LRScheduler
        ):
            raise TypeError(
                "scheduler must be OptimizationConfig.ABSENT or an LRScheduler, "
                f"got {type(scheduler)!r}"
            )
        self.optimizer = optimizer
        self.scheduler = scheduler


class EarlyStoppingConfig:
    NONE = "none"
    PATIENCE = "patience"

    def __init__(self, *, name: str, patience: int, min_delta: float) -> None:
        if name == EarlyStoppingConfig.NONE:
            raise ValueError(
                "EarlyStoppingConfig.NONE has no patience fields; "
                "call EarlyStoppingConfig.none()"
            )
        if name != EarlyStoppingConfig.PATIENCE:
            raise ValueError(
                f"early stopping name must be {EarlyStoppingConfig.PATIENCE!r} "
                f"or use none(), got {name!r}"
            )
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError(f"patience must be an int >= 1, got {patience!r}")
        if type(min_delta) is not float or min_delta < 0.0:
            raise ValueError(f"min_delta must be a float >= 0, got {min_delta!r}")
        self.name = name
        self.patience = patience
        self.min_delta = min_delta

    @classmethod
    def none(cls) -> EarlyStoppingConfig:
        config = cls.__new__(cls)
        config.name = EarlyStoppingConfig.NONE
        return config


class TrainerConfig(BaseModel):
    model_config = _STRICT_LIVE

    accelerator: Literal["gpu", "cpu"]
    devices: tuple[int, ...]
    strategy: Literal["single_device", "ddp"]
    precision: int | str
    max_epochs: int = Field(ge=1)
    accumulate_grad_batches: int = Field(ge=1)
    enable_progress_bar: bool
    early_stopping: EarlyStoppingConfig

    @field_validator("devices")
    @classmethod
    def _gpu_indices(cls, devices: tuple[int, ...]) -> tuple[int, ...]:
        if not isinstance(devices, tuple):
            raise TypeError(
                f"devices must be a tuple of GPU indices, got {type(devices)!r}"
            )
        seen: dict[int, None] = {}
        n_devices = len(devices)
        for i in range(n_devices):
            index = devices[i]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(
                    f"devices[{i}] must be an int GPU index >= 0, got {index!r}"
                )
            if index in seen:
                raise ValueError(f"devices has duplicate index {index}")
            seen[index] = None
        return devices

    @field_validator("max_epochs", "accumulate_grad_batches")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"must be an int >= 1, got {value!r}")
        return value

    @field_validator("precision")
    @classmethod
    def _precision(cls, value: int | str) -> int | str:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise TypeError(f"precision must be int or str, got {type(value)!r}")
        return value

    @model_validator(mode="after")
    def _legal_compute_row(self) -> Self:
        n_devices = len(self.devices)
        if self.accelerator == Accelerator.CPU:
            if n_devices != 0:
                raise ValueError(f"cpu devices must be (), got {self.devices!r}")
            if self.strategy != Strategy.SINGLE_DEVICE:
                raise ValueError(
                    f"cpu strategy must be {Strategy.SINGLE_DEVICE!r}, "
                    f"got {self.strategy!r}"
                )
            return self
        if n_devices == 0:
            raise ValueError("gpu devices must be a non-empty tuple of GPU indices")
        if n_devices == 1:
            if self.strategy != Strategy.SINGLE_DEVICE:
                raise ValueError(
                    f"one GPU requires strategy {Strategy.SINGLE_DEVICE!r}, "
                    f"got {self.strategy!r}"
                )
            return self
        if self.strategy != Strategy.DDP:
            raise ValueError(
                f"several GPUs require strategy {Strategy.DDP!r}, "
                f"got {self.strategy!r}"
            )
        return self


class RunLogConfig(BaseModel):
    model_config = _STRICT

    payload: dict[str, Any]
    filename: str

    @field_validator("filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        return parse_non_empty_str(value, label="run_log.filename")

    @field_validator("payload")
    @classmethod
    def _json_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        dump_object(value, form=JsonTextForm.ASCII_PRETTY)
        return value
