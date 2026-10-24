from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT
from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.dl_engine.training.config import FitConfig
from dl_toolbox.dl_engine.training.module import ClassificationLightningModule


class FitError(ValueError):
    pass


def build_metric_inputs(
    output: Tensor,
    batch: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    return {**batch, KEY_OUTPUT: output}


def fit_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    *,
    prepare_tensors: Callable[[Any], Mapping[str, Tensor]],
    forward_fn: Callable[[nn.Module, Mapping[str, Tensor]], Tensor],
    loss_fn: Callable[[Tensor, Mapping[str, Tensor]], Tensor],
    build_metric_inputs_fn: Callable[[Tensor, Mapping[str, Tensor]], Mapping[str, Tensor]],
    train_metrics: Sequence[MetricSpec],
    val_metrics: Sequence[MetricSpec],
    config: FitConfig,
) -> L.Trainer:
    if config.trainer.max_epochs < 1:
        raise FitError(f"max_epochs must be >= 1, got {config.trainer.max_epochs}")

    checkpoint_dir = Path(config.checkpoint.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lit_module = ClassificationLightningModule(
        net=model,
        optimizer=optimizer,
        epoch_logger=config.logger.logger,
        prepare_tensors=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs=build_metric_inputs_fn,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        monitor_key=config.monitor.key,
        run_config=config.logger.run_config,
        run_config_filename=config.logger.run_config_filename,
    )

    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=config.checkpoint.filename,
            monitor=config.monitor.key,
            mode=config.monitor.mode,
            save_top_k=config.checkpoint.save_top_k,
            save_last=config.checkpoint.save_last,
        )
    ]
    if config.early_stopping is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config.monitor.key,
                patience=config.early_stopping.patience,
                mode=config.monitor.mode,
                min_delta=config.early_stopping.min_delta,
            )
        )

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        callbacks=callbacks,
        enable_progress_bar=config.trainer.enable_progress_bar,
        logger=False,
    )
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer
