"""One metric: tensor inputs, a fold, and named reducers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import AbstractSet

from torch import Tensor

LEVEL_BATCH = "batch"
LEVEL_EPOCH = "epoch"

MetricSpec = tuple["MetricObject", Sequence[str], Sequence[str]]


@dataclass(frozen=True)
class MetricObject:
    name: str
    required_keys: AbstractSet[str]
    metric_function: Callable[[Mapping[str, Tensor]], Tensor]
    combiner: Callable[[Tensor, Tensor], Tensor]
    reducers: Mapping[str, Callable[[Tensor], Tensor]]


def metric_log_name(name: str, split: str, level: str, reducer_name: str) -> str:
    return f"{name}/{split}/{level}/{reducer_name}"
