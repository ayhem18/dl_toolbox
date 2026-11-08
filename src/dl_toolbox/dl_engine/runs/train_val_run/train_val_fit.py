"""fit() for a train+val Lightning module.

on_fit_start (rank 0) is LogStream.FIT / LogExistence.CREATE, flatten_object
into log_params, dump_object into log_text. Lightning logger=False,
enable_checkpointing=False. Close is this procedure's finally.
"""

from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.callback import (
    InspectableCheckpointCallback,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.runs.common.run_config import (
    EarlyStoppingConfig,
    FitSteps,
    OptimizationConfig,
    RunLogConfig,
    TrainerConfig,
)
from dl_toolbox.dl_engine.runs.common.run_loop import epoch_metric_keys
from dl_toolbox.dl_engine.runs.common.run_trainer import (
    FitError,
    lightning_devices,
    lightning_strategy,
    require_single_machine,
)
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import (
    TrainValLoaders,
    TrainValMetrics,
)
from dl_toolbox.dl_engine.runs.train_val_run.train_val_module import TrainValLightningModule


def fit(
    *,
    net: nn.Module,
    loaders: TrainValLoaders,
    steps: FitSteps,
    metrics: TrainValMetrics,
    optimization: OptimizationConfig,
    trainer: TrainerConfig,
    checkpointer: InspectableCheckpointer,
    logger: Logger,
    run_log: RunLogConfig,
) -> L.Trainer:
    if not isinstance(net, nn.Module):
        raise TypeError(f"net must be nn.Module, got {type(net)!r}")
    if not isinstance(loaders, TrainValLoaders):
        raise TypeError(f"loaders must be TrainValLoaders, got {type(loaders)!r}")
    if not isinstance(steps, FitSteps):
        raise TypeError(f"steps must be FitSteps, got {type(steps)!r}")
    if not isinstance(metrics, TrainValMetrics):
        raise TypeError(f"metrics must be TrainValMetrics, got {type(metrics)!r}")
    if not isinstance(optimization, OptimizationConfig):
        raise TypeError(
            f"optimization must be OptimizationConfig, got {type(optimization)!r}"
        )
    if not isinstance(trainer, TrainerConfig):
        raise TypeError(f"trainer must be TrainerConfig, got {type(trainer)!r}")
    if not isinstance(checkpointer, InspectableCheckpointer):
        raise TypeError(
            f"checkpointer must be InspectableCheckpointer, got {type(checkpointer)!r}"
        )
    if not isinstance(logger, Logger):
        raise TypeError(f"logger must be a dl_toolbox Logger, got {type(logger)!r}")
    if not isinstance(run_log, RunLogConfig):
        raise TypeError(f"run_log must be RunLogConfig, got {type(run_log)!r}")

    val_keys = epoch_metric_keys(metrics.val_metrics, split="val")
    if metrics.monitor.key not in val_keys:
        raise FitError(
            f"monitor.key {metrics.monitor.key!r} missing from val epoch metrics: "
            f"{sorted(val_keys)}"
        )

    lit_module = TrainValLightningModule(
        net=net,
        steps=steps,
        metrics=metrics,
        optimization=optimization,
        logger=logger,
        run_log=run_log,
    )

    callbacks: list[L.Callback] = [
        InspectableCheckpointCallback(checkpointer=checkpointer)
    ]
    if trainer.early_stopping.name == EarlyStoppingConfig.PATIENCE:
        callbacks.append(
            EarlyStopping(
                monitor=metrics.monitor.key,
                patience=trainer.early_stopping.patience,
                mode=metrics.monitor.mode,
                min_delta=trainer.early_stopping.min_delta,
            )
        )

    lightning_trainer = L.Trainer(
        max_epochs=trainer.max_epochs,
        accelerator=trainer.accelerator,
        devices=lightning_devices(
            accelerator=trainer.accelerator, devices=trainer.devices
        ),
        strategy=lightning_strategy(trainer.strategy),
        precision=trainer.precision,
        accumulate_grad_batches=trainer.accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=trainer.enable_progress_bar,
        logger=False,
        enable_checkpointing=False,
        check_val_every_n_epoch=1,
        num_nodes=1,
    )

    
    require_single_machine(trainer=lightning_trainer, config=trainer)

    try:
        lightning_trainer.fit(
            lit_module,
            train_dataloaders=loaders.train,
            val_dataloaders=loaders.val,
        )
    finally:
        logger.close()
    return lightning_trainer
