from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CheckpointMonitor,
    CheckpointPolicy,
)
from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.metrics.classification.classification import (
    binary_bce_loss_metric,
    binary_classification_accuracy_metric,
)
from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.runs.common.run_config import (
    Accelerator,
    EarlyStoppingConfig,
    FitSteps,
    MonitorConfig,
    RunLogConfig,
    Strategy,
    TrainerConfig,
)


class RecordingLogger(Logger):
    def __init__(self) -> None:
        self.setup_calls: list[tuple[str, str]] = []
        self.close_calls = 0
        self.params: list[Mapping[str, str]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.scalars: list[tuple[dict[str, float], int]] = []

    def setup(self, *, stream: str, existence: str) -> None:
        self.setup_calls.append((stream, existence))

    def log_params(self, params: Mapping[str, str]) -> None:
        self.params.append(dict(params))

    def log_tags(self, tags: Mapping[str, str]) -> None:
        del tags

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        self.scalars.append((dict(values), int(step)))

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        del tag, image, step

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        del tag, rows, step

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        self.texts.append((tag, text, int(step)))

    def close(self) -> None:
        self.close_calls += 1


class LinearBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def prepare_tensors(batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    features, labels = batch
    return {"input": features, KEY_TARGET: labels}


def forward_fn(net: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    return net(batch["input"])


def loss_fn(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
    target = batch[KEY_TARGET].float().unsqueeze(1)
    return torch.nn.functional.binary_cross_entropy_with_logits(output, target)


def build_metric_inputs(
    output: Tensor, batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    return {**dict(batch), KEY_OUTPUT: output}


def binary_specs() -> list:
    loss_metric = binary_bce_loss_metric()
    acc = binary_classification_accuracy_metric()
    return [(loss_metric, (), ("mean",)), (acc, (), ("acc",))]


def loss_monitor_key(*, split: str) -> str:
    return metric_log_name("loss", split, LEVEL_EPOCH, "mean")


def cpu_trainer(
    *,
    max_epochs: int,
    accumulate_grad_batches: int,
    early_stopping: EarlyStoppingConfig,
) -> TrainerConfig:
    return TrainerConfig(
        accelerator=Accelerator.CPU,
        devices=(),
        strategy=Strategy.SINGLE_DEVICE,
        precision=32,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=False,
        early_stopping=early_stopping,
    )


def steps() -> FitSteps:
    return FitSteps(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs_fn=build_metric_inputs,
    )


def binary_loader(*, n: int, batch_size: int) -> DataLoader:
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def run_log() -> RunLogConfig:
    return RunLogConfig(payload={"task": "unit"}, filename="run_config")


def checkpointer(
    run_artifacts_dir: Path, *, max_epochs: int, monitor_key: str
) -> InspectableCheckpointer:
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return InspectableCheckpointer(
        run_artifacts_dir=run_artifacts_dir,
        policy=CheckpointPolicy(save_top_k=-1, save_last=True),
        monitor=CheckpointMonitor(key=monitor_key, mode="min"),
        max_epochs=max_epochs,
        lock_timeout_seconds=5.0,
    )


def train_val_metrics(specs: list):
    from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import TrainValMetrics

    key = loss_monitor_key(split="val")
    return TrainValMetrics(
        train_metrics=specs,
        val_metrics=specs,
        monitor=MonitorConfig(key=key, mode="min"),
    )


def train_only_metrics(specs: list):
    from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import TrainOnlyMetrics

    key = loss_monitor_key(split="train")
    return TrainOnlyMetrics(
        train_metrics=specs,
        monitor=MonitorConfig(key=key, mode="min"),
    )
