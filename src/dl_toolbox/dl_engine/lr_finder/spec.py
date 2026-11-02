"""Sweep hyperparameters. Parse from run_config; validate in one place."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import (
    parse_half_open_unit_interval,
    parse_positive_int,
    parse_positive_number,
)
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError

LR_FINDER_CONFIG_KEY = "lr_finder"
NUM_ITERATIONS_PER_LR_KEY = "num_iterations_per_lr"
LR_FINDER_SPEC_KEYS = (
    "start_lr",
    "end_lr",
    "num_iter",
    "smooth_f",
    "diverge_th",
)


@dataclass(frozen=True)
class LrFinderSpec:
    start_lr: float
    end_lr: float
    num_iter: int
    smooth_f: float
    diverge_th: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "start_lr": self.start_lr,
            "end_lr": self.end_lr,
            "num_iter": self.num_iter,
            "smooth_f": self.smooth_f,
            "diverge_th": self.diverge_th,
        }


def validate_lr_finder_spec(spec: LrFinderSpec) -> None:
    if spec.num_iter < 2:
        raise LrFinderError(f"num_iter must be >= 2, got {spec.num_iter}")
    if spec.start_lr <= 0.0 or spec.end_lr <= spec.start_lr:
        raise LrFinderError(
            f"need 0 < start_lr < end_lr, got start_lr={spec.start_lr} "
            f"end_lr={spec.end_lr}"
        )
    if not 0.0 <= spec.smooth_f < 1.0:
        raise LrFinderError(f"smooth_f must be in [0, 1), got {spec.smooth_f}")
    if spec.diverge_th <= 1.0:
        raise LrFinderError(f"diverge_th must be > 1, got {spec.diverge_th}")


def log_spaced_lrs(spec: LrFinderSpec) -> tuple[float, ...]:
    """Inclusive log grid: ``start * (end/start) ** (i / (n-1))``."""
    validate_lr_finder_spec(spec)
    num_iter = spec.num_iter
    gamma = spec.end_lr / spec.start_lr
    denominator = float(num_iter - 1)
    lrs: list[float] = [0.0] * num_iter
    for i in range(num_iter):
        lrs[i] = spec.start_lr * (gamma ** (float(i) / denominator))
    return tuple(lrs)


def parse_lr_finder_spec(run_config: Mapping[str, Any]) -> LrFinderSpec:
    raw = _lr_finder_raw(run_config)
    spec = LrFinderSpec(
        start_lr=parse_positive_number(raw["start_lr"], label="lr_finder.start_lr"),
        end_lr=parse_positive_number(raw["end_lr"], label="lr_finder.end_lr"),
        num_iter=parse_positive_int(raw["num_iter"], label="lr_finder.num_iter"),
        smooth_f=parse_half_open_unit_interval(
            raw["smooth_f"], label="lr_finder.smooth_f"
        ),
        diverge_th=parse_positive_number(
            raw["diverge_th"], label="lr_finder.diverge_th"
        ),
    )
    validate_lr_finder_spec(spec)
    return spec


def parse_num_iterations_per_lr(run_config: Mapping[str, Any]) -> int:
    raw = _lr_finder_raw(run_config)
    if NUM_ITERATIONS_PER_LR_KEY not in raw:
        return 1
    n = parse_positive_int(
        raw[NUM_ITERATIONS_PER_LR_KEY],
        label="lr_finder.num_iterations_per_lr",
    )
    validate_num_iterations_per_lr(n)
    return n


def validate_num_iterations_per_lr(n: int) -> None:
    if n < 1:
        raise LrFinderError(f"num_iterations_per_lr must be >= 1, got {n}")


def _lr_finder_raw(run_config: Mapping[str, Any]) -> Mapping[str, Any]:
    obj = parse_mapping(run_config, label="run_config")
    if LR_FINDER_CONFIG_KEY not in obj:
        raise LrFinderError("run_config must contain 'lr_finder'")
    raw = parse_mapping(obj[LR_FINDER_CONFIG_KEY], label="lr_finder")
    keys = LR_FINDER_SPEC_KEYS
    if NUM_ITERATIONS_PER_LR_KEY in raw:
        keys = LR_FINDER_SPEC_KEYS + (NUM_ITERATIONS_PER_LR_KEY,)
    require_exact_keys(raw, keys, label="lr_finder")
    return raw
