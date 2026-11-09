"""FindLrError, FindLrSpec, FindLrSteps, FindLrRuntime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Self

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from torch import Tensor, nn

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import (
    one_of_message,
    parse_one_of,
    require_key,
)
from dl_toolbox.dl_engine.runs.common.run_config import Accelerator, FitSteps

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class FindLrError(ValueError):
    """Illegal spec, empty loader, too few kept points, suggestion failure, GPU without CUDA."""


class _FindLrGridDocument(BaseModel):
    model_config = _STRICT

    start_lr: float
    end_lr: float
    num_iter: int

    @field_validator("start_lr", "end_lr", mode="before")
    @classmethod
    def _grid_floats(cls, value: object, info: ValidationInfo) -> float:
        if type(value) is not float:
            raise ValueError(f"{info.field_name} must be a float, got {value!r}")
        return value

    @field_validator("num_iter", mode="before")
    @classmethod
    def _num_iter(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"num_iter must be an int >= 2, got {value!r}")
        return value

    @model_validator(mode="after")
    def _ordered_grid(self) -> Self:
        if self.start_lr <= 0.0:
            raise ValueError(f"start_lr must be a float > 0, got {self.start_lr!r}")
        if self.end_lr <= self.start_lr:
            raise ValueError(
                f"need 0 < start_lr < end_lr, got start_lr={self.start_lr!r} "
                f"end_lr={self.end_lr!r}"
            )
        return self


class RangeFindLrSpecDocument(_FindLrGridDocument):
    kind: Literal["range"]
    smooth_f: float
    diverge_th: float

    @field_validator("smooth_f", "diverge_th", mode="before")
    @classmethod
    def _range_floats(cls, value: object, info: ValidationInfo) -> float:
        if type(value) is not float:
            raise ValueError(f"{info.field_name} must be a float, got {value!r}")
        return value

    @model_validator(mode="after")
    def _range_rules(self) -> Self:
        if not 0.0 <= self.smooth_f < 1.0:
            raise ValueError(
                f"smooth_f must be a float in [0, 1), got {self.smooth_f!r}"
            )
        if self.diverge_th <= 1.0:
            raise ValueError(
                f"diverge_th must be a float > 1, got {self.diverge_th!r}"
            )
        return self


class FullDsFindLrSpecDocument(_FindLrGridDocument):
    kind: Literal["full_ds"]
    num_iterations_per_lr: int

    @field_validator("num_iterations_per_lr", mode="before")
    @classmethod
    def _passes(cls, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                "num_iterations_per_lr must be an int >= 1, "
                f"got {value!r}"
            )
        return value


class FindLrSpec:
    RANGE = "range"
    FULL_DS = "full_ds"
    KINDS = (RANGE, FULL_DS)

    def __init__(self) -> None:
        raise FindLrError(
            "FindLrSpec() is not a constructor; call FindLrSpec.range(...) "
            "or FindLrSpec.full_ds(...)"
        )

    @classmethod
    def range(
        cls,
        *,
        start_lr: float,
        end_lr: float,
        num_iter: int,
        smooth_f: float,
        diverge_th: float,
    ) -> FindLrSpec:
        return _from_document(
            _validate_document(
                RangeFindLrSpecDocument,
                {
                    "kind": cls.RANGE,
                    "start_lr": start_lr,
                    "end_lr": end_lr,
                    "num_iter": num_iter,
                    "smooth_f": smooth_f,
                    "diverge_th": diverge_th,
                },
            )
        )

    @classmethod
    def full_ds(
        cls,
        *,
        start_lr: float,
        end_lr: float,
        num_iter: int,
        num_iterations_per_lr: int,
    ) -> FindLrSpec:
        return _from_document(
            _validate_document(
                FullDsFindLrSpecDocument,
                {
                    "kind": cls.FULL_DS,
                    "start_lr": start_lr,
                    "end_lr": end_lr,
                    "num_iter": num_iter,
                    "num_iterations_per_lr": num_iterations_per_lr,
                },
            )
        )

    def to_dict(self) -> dict[str, float | int | str]:
        if self.kind == FindLrSpec.RANGE:
            return {
                "kind": self.kind,
                "start_lr": self.start_lr,
                "end_lr": self.end_lr,
                "num_iter": self.num_iter,
                "smooth_f": self.smooth_f,
                "diverge_th": self.diverge_th,
            }
        return {
            "kind": self.kind,
            "start_lr": self.start_lr,
            "end_lr": self.end_lr,
            "num_iter": self.num_iter,
            "num_iterations_per_lr": self.num_iterations_per_lr,
        }


def log_spaced_lrs(spec: FindLrSpec) -> tuple[float, ...]:
    if not isinstance(spec, FindLrSpec):
        raise TypeError(f"spec must be FindLrSpec, got {type(spec)!r}")
    num_iter = spec.num_iter
    gamma = spec.end_lr / spec.start_lr
    denominator = float(num_iter - 1)
    lrs: list[float] = [0.0] * num_iter
    for i in range(num_iter):
        lrs[i] = spec.start_lr * (gamma ** (float(i) / denominator))
    return tuple(lrs)


def parse_find_lr_spec(payload: object) -> FindLrSpec:
    obj = parse_mapping(payload, label="FindLrSpec")
    kind = parse_one_of(
        require_key(obj, "kind", label="FindLrSpec"),
        allowed=FindLrSpec.KINDS,
        label="kind",
        error_type=FindLrError,
        error_message=one_of_message,
    )
    if kind == FindLrSpec.RANGE:
        return _from_document(_validate_document(RangeFindLrSpecDocument, obj))
    return _from_document(_validate_document(FullDsFindLrSpecDocument, obj))


def _validate_document(
    model: type[RangeFindLrSpecDocument] | type[FullDsFindLrSpecDocument],
    payload: object,
) -> RangeFindLrSpecDocument | FullDsFindLrSpecDocument:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise FindLrError(f"FindLrSpec: {exc}") from exc


def _from_document(
    document: RangeFindLrSpecDocument | FullDsFindLrSpecDocument,
) -> FindLrSpec:
    spec = FindLrSpec.__new__(FindLrSpec)
    spec.kind = document.kind
    spec.start_lr = document.start_lr
    spec.end_lr = document.end_lr
    spec.num_iter = document.num_iter
    if isinstance(document, RangeFindLrSpecDocument):
        spec.smooth_f = document.smooth_f
        spec.diverge_th = document.diverge_th
        return spec
    spec.num_iterations_per_lr = document.num_iterations_per_lr
    return spec


class FindLrSteps:
    def __init__(
        self,
        *,
        prepare_tensors_fn: Callable[[Any], Mapping[str, Any]],
        forward_fn: Callable[[nn.Module, Mapping[str, Any]], Tensor],
        loss_fn: Callable[[Tensor, Mapping[str, Any]], Tensor],
    ) -> None:
        if not callable(prepare_tensors_fn):
            raise TypeError("prepare_tensors_fn must be callable")
        if not callable(forward_fn):
            raise TypeError("forward_fn must be callable")
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        self.prepare_tensors_fn = prepare_tensors_fn
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn

    @classmethod
    def from_fit_steps(cls, fit_steps: FitSteps) -> FindLrSteps:
        if not isinstance(fit_steps, FitSteps):
            raise TypeError(f"fit_steps must be FitSteps, got {type(fit_steps)!r}")
        return cls(
            prepare_tensors_fn=fit_steps.prepare_tensors_fn,
            forward_fn=fit_steps.forward_fn,
            loss_fn=fit_steps.loss_fn,
        )


class FindLrRuntime:
    def __init__(
        self,
        *,
        accelerator: str,
        precision: int | str,
        enable_progress_bar: bool,
    ) -> None:
        self.accelerator = parse_one_of(
            accelerator,
            allowed=(Accelerator.CPU, Accelerator.GPU),
            label="accelerator",
            error_type=FindLrError,
            error_message=one_of_message,
        )
        if isinstance(precision, bool) or not isinstance(precision, (int, str)):
            raise FindLrError(
                f"precision must be int or str, got {type(precision)!r}"
            )
        if not isinstance(enable_progress_bar, bool):
            raise FindLrError(
                f"enable_progress_bar must be bool, got {type(enable_progress_bar)!r}"
            )
        self.precision = precision
        self.enable_progress_bar = enable_progress_bar


def resolve_device(runtime: FindLrRuntime) -> torch.device:
    if not isinstance(runtime, FindLrRuntime):
        raise TypeError(f"runtime must be FindLrRuntime, got {type(runtime)!r}")
    if runtime.accelerator == Accelerator.CPU:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise FindLrError("accelerator is gpu but CUDA is not available")
    return torch.device("cuda")
