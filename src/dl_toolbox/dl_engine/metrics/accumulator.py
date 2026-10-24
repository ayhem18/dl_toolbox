from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .contracts import slice_metric_inputs
from .metric_object import LEVEL_BATCH, LEVEL_EPOCH, MetricSpec, metric_log_name


class MetricAccumulator:
    def __init__(self, specs: Sequence[MetricSpec], split: str) -> None:
        self.specs = list(specs)
        self.split = split
        self._combined: dict[str, Tensor | None] = {spec[0].name: None for spec in self.specs}
        self.n_batches = 0

    def reset(self) -> None:
        for key in self._combined:
            self._combined[key] = None
        self.n_batches = 0

    def update(self, metric_inputs: Mapping[str, Tensor]) -> dict[str, float]:
        self.n_batches += 1
        batch_logs: dict[str, float] = {}
        for metric, batch_reducers, _epoch_reducers in self.specs:
            inputs = slice_metric_inputs(metric_inputs, metric.required_keys)
            value = metric.metric_function(inputs).detach()
            for reducer_name in batch_reducers:
                key = metric_log_name(metric.name, self.split, LEVEL_BATCH, reducer_name)
                batch_logs[key] = _as_float(metric.reducers[reducer_name](value))
            previous = self._combined[metric.name]
            self._combined[metric.name] = (
                value if previous is None else metric.combiner(previous, value)
            )
        return batch_logs

    def epoch_scalars(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric, _batch_reducers, epoch_reducers in self.specs:
            values = self._combined[metric.name]
            if values is None:
                continue
            for reducer_name in epoch_reducers:
                key = metric_log_name(metric.name, self.split, LEVEL_EPOCH, reducer_name)
                out[key] = _as_float(metric.reducers[reducer_name](values))
        return out


def _as_float(value: Tensor) -> float:
    if value.numel() != 1:
        raise ValueError(f"reducer must return a scalar tensor, got shape {tuple(value.shape)}")
    return float(value.detach().cpu().item())
