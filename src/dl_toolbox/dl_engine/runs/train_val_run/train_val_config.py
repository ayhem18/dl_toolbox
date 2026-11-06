"""Train/val-only grouped inputs."""

from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.dl_engine.runs.common.run_config import MonitorConfig


class TrainValLoaders:
    def __init__(self, *, train: DataLoader, val: DataLoader) -> None:
        if not isinstance(train, DataLoader):
            raise TypeError(f"train must be a DataLoader, got {type(train)!r}")
        if not isinstance(val, DataLoader):
            raise TypeError(f"val must be a DataLoader, got {type(val)!r}")
        self.train = train
        self.val = val


class TrainValMetrics:
    def __init__(
        self,
        *,
        train_metrics: Sequence[MetricSpec],
        val_metrics: Sequence[MetricSpec],
        monitor: MonitorConfig,
    ) -> None:
        if not isinstance(monitor, MonitorConfig):
            raise TypeError(f"monitor must be MonitorConfig, got {type(monitor)!r}")
        self.train_metrics = list(train_metrics)
        self.val_metrics = list(val_metrics)
        self.monitor = monitor
