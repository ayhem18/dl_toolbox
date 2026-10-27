"""Shared experiment_management fixtures. Not collected by pytest."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.metrics.classification.classification import (
    binary_bce_loss_metric,
)
from dl_toolbox.dl_engine.metrics.contracts import KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.training.config import MonitorConfig
from dl_toolbox.dl_engine.training.fit import build_metric_inputs
from dl_toolbox.experiment_management.experiment.build import ExperimentBuild
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager


class _LinearBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _prepare_tensors(batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    features, labels = batch
    return {"input": features, KEY_TARGET: labels}


def _forward(net: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    return net(batch["input"])


def _loss(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
    target = batch[KEY_TARGET].float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(output, target)


class TinyExperimentConfigManager(ExperimentConfigManager):
    def build(
        self,
        experiment_config: Mapping[str, Any],
        *,
        project_root: Path,
    ) -> ExperimentBuild:
        del experiment_config, project_root
        x = torch.randn(16, 4)
        y = (x[:, 0] > 0).long()
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        model = _LinearBinary()
        loss_metric = binary_bce_loss_metric()
        specs = [(loss_metric, (), ("mean",))]
        monitor = metric_log_name("loss", "val", LEVEL_EPOCH, "mean")
        return ExperimentBuild(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
            loss_fn=_loss,
            prepare_tensors_fn=_prepare_tensors,
            forward_fn=_forward,
            build_metric_inputs_fn=build_metric_inputs,
            train_metrics=specs,
            val_metrics=specs,
            monitor=MonitorConfig(key=monitor, mode="min"),
        )


def tensorboard_toolbox_config(*, seed: int = 0, run_name: str = "tiny_run") -> dict:
    return {
        "logger_config": {"run_name": run_name},
        "checkpoint_config": {"save_top_k": -1, "save_last": True},
        "trainer_config": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
            "enable_progress_bar": False,
        },
        "seed_config": {"seed": seed},
        "early_stopping_config": {"name": "none", "params": {}},
    }


def placeholder_recipe(*, name: str, seed: int = 0, run_name: str = "tiny_run") -> dict:
    return {
        "human_readable_name": name,
        "dl_toolbox_config": tensorboard_toolbox_config(seed=seed, run_name=run_name),
        "experiment_config": {"task": "tiny"},
    }
