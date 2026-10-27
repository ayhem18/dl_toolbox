from __future__ import annotations

from typing import Any
from collections.abc import Callable, Mapping, Sequence

import lightning as L
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping

from dl_toolbox.dl_engine.loggers.base_logger import Logger
from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT
from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.dl_engine.training.checkpointer.checkpointer import Checkpointer
from dl_toolbox.dl_engine.training.config import EarlyStoppingConfig, FitConfig
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
    prepare_tensors_fn: Callable[[Any], Mapping[str, Tensor]],
    forward_fn: Callable[[nn.Module, Mapping[str, Tensor]], Tensor],
    loss_fn: Callable[[Tensor, Mapping[str, Tensor]], Tensor],
    build_metric_inputs_fn: Callable[[Tensor, Mapping[str, Tensor]], Mapping[str, Tensor]],
    train_metrics: Sequence[MetricSpec],
    val_metrics: Sequence[MetricSpec],
    config: FitConfig,
    checkpointer: Checkpointer,
    logger: Logger,
    run_config: Mapping[str, Any] | None = None, # TODO: no defaults
    run_config_filename: str = "run_config", # TODO: no defaults: these 2 parameters may or may not be necessary.
) -> L.Trainer:
    if config.trainer.max_epochs < 1:
        raise FitError(f"max_epochs must be >= 1, got {config.trainer.max_epochs}")

    lit_module = ClassificationLightningModule(
        net=model,
        optimizer=optimizer,
        epoch_logger=logger,
        prepare_tensors_fn=prepare_tensors_fn,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs_fn=build_metric_inputs_fn,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        monitor_key=config.monitor.key,
        run_config=dict(run_config or {}),
        run_config_filename=run_config_filename,
    )

    callbacks: list[L.Callback] = [checkpointer.as_callback()]
    if config.early_stopping.name == EarlyStoppingConfig.PATIENCE:
        if config.early_stopping.patience is None:
            raise FitError("patience early stopping requires patience")
        callbacks.append(
            EarlyStopping(
                monitor=config.monitor.key,
                patience=config.early_stopping.patience,
                mode=config.monitor.mode,
                min_delta=float(config.early_stopping.min_delta or 0.0), # I see an "or" here ... which assumes that the min_delta can be none... 
            )
        )

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        callbacks=callbacks,
        enable_progress_bar=config.trainer.enable_progress_bar,
        logger=False,
        enable_checkpointing=False,
    )
    try:
        trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    finally:
        logger.close()
    return trainer
