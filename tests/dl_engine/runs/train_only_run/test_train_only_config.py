from __future__ import annotations

import inspect

import pytest
from torch.utils.data import DataLoader, TensorDataset
import torch

from dl_toolbox.dl_engine.runs.common.run_config import MonitorConfig
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import (
    TrainOnlyLoaders,
    TrainOnlyMetrics,
)


def test_loaders_have_no_val_slot() -> None:
    params = inspect.signature(TrainOnlyLoaders.__init__).parameters
    assert "val" not in params
    loader = DataLoader(TensorDataset(torch.zeros(2, 4), torch.zeros(2)), batch_size=2)
    loaders = TrainOnlyLoaders(train=loader)
    assert loaders.train is loader
    assert not hasattr(loaders, "val")


def test_loaders_require_dataloader() -> None:
    with pytest.raises(TypeError):
        TrainOnlyLoaders(train=[])  # type: ignore[arg-type]


def test_metrics_have_no_val_metrics_slot() -> None:
    params = inspect.signature(TrainOnlyMetrics.__init__).parameters
    assert "val_metrics" not in params
    metrics = TrainOnlyMetrics(
        train_metrics=[],
        monitor=MonitorConfig(key="loss/train/epoch/mean", mode="min"),
    )
    assert not hasattr(metrics, "val_metrics")
    assert metrics.monitor.key == "loss/train/epoch/mean"
