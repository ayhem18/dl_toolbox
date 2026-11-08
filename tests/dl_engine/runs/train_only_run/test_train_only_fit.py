from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.checkpoint import Checkpoint
from torch.optim.lr_scheduler import LambdaLR

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.callback import (
    InspectableCheckpointCallback,
    TrainEpochCheckpointCallback,
)
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream
from dl_toolbox.dl_engine.metrics.classification.classification import (
    binary_bce_loss_metric,
)
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.runs.common.run_config import (
    EarlyStoppingConfig,
    MonitorConfig,
    OptimizationConfig,
)
from dl_toolbox.dl_engine.runs.common.run_trainer import FitError
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import (
    TrainOnlyLoaders,
    TrainOnlyMetrics,
)
from dl_toolbox.dl_engine.runs.train_only_run.train_only_fit import fit
from tests.dl_engine.runs.support import (
    LinearBinary,
    RecordingLogger,
    binary_loader,
    binary_specs,
    checkpointer,
    cpu_trainer,
    loss_monitor_key,
    run_log,
    steps,
    train_only_metrics,
)


def test_fit_has_no_defaults() -> None:
    params = inspect.signature(fit).parameters
    for name, param in params.items():
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_train_epoch_callback_is_lightning_checkpoint() -> None:
    assert issubclass(TrainEpochCheckpointCallback, Checkpoint)
    assert not issubclass(InspectableCheckpointCallback, Checkpoint)


def test_fit_cpu_end_to_end_writes_catalog_without_val(tmp_path: Path) -> None:
    specs = binary_specs()
    model = LinearBinary()
    logger = RecordingLogger()
    loader = binary_loader(n=32, batch_size=8)
    trainer = fit(
        net=model,
        loaders=TrainOnlyLoaders(train=loader),
        steps=steps(),
        metrics=train_only_metrics(specs),
        optimization=OptimizationConfig(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
            scheduler=OptimizationConfig.ABSENT,
        ),
        trainer=cpu_trainer(
            max_epochs=2,
            accumulate_grad_batches=1,
            early_stopping=EarlyStoppingConfig.none(),
        ),
        checkpointer=checkpointer(
            tmp_path / "run",
            max_epochs=2,
            monitor_key=loss_monitor_key(split="train"),
        ),
        logger=logger,
        run_log=run_log(),
    )
    assert trainer.current_epoch >= 1
    assert trainer.val_dataloaders is None
    callbacks = trainer.callbacks
    assert any(isinstance(cb, TrainEpochCheckpointCallback) for cb in callbacks)
    assert not any(isinstance(cb, InspectableCheckpointCallback) for cb in callbacks)
    assert not any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
    checkpoints = tmp_path / "run" / "checkpoints"
    assert (checkpoints / "catalog.json").is_file()
    ckpt_files = list((checkpoints / "live").glob("*.ckpt"))
    assert ckpt_files
    assert logger.setup_calls == [(LogStream.FIT, LogExistence.CREATE)]
    assert logger.close_calls >= 1
    assert logger.params
    assert logger.texts[0][0] == "run_config"
    assert logger.scalars
    logged_keys = logger.scalars[0][0]
    assert any(key.startswith("loss/train/") for key in logged_keys)
    assert not any("/val/" in key for key in logged_keys)


def test_fit_live_scheduler_and_patience(tmp_path: Path) -> None:
    specs = [(binary_bce_loss_metric(), (), ("mean",))]
    model = LinearBinary()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    loader = binary_loader(n=16, batch_size=8)
    trainer = fit(
        net=model,
        loaders=TrainOnlyLoaders(train=loader),
        steps=steps(),
        metrics=train_only_metrics(specs),
        optimization=OptimizationConfig(optimizer=optimizer, scheduler=scheduler),
        trainer=cpu_trainer(
            max_epochs=1,
            accumulate_grad_batches=2,
            early_stopping=EarlyStoppingConfig(
                name=EarlyStoppingConfig.PATIENCE, patience=3, min_delta=1e-5
            ),
        ),
        checkpointer=checkpointer(
            tmp_path / "run",
            max_epochs=1,
            monitor_key=loss_monitor_key(split="train"),
        ),
        logger=RecordingLogger(),
        run_log=run_log(),
    )
    assert trainer.accumulate_grad_batches == 2
    assert trainer.val_dataloaders is None


def test_fit_rejects_val_monitor_key(tmp_path: Path) -> None:
    loss_metric = binary_bce_loss_metric()
    specs = [(loss_metric, (), ("mean",))]
    model = LinearBinary()
    loader = binary_loader(n=8, batch_size=8)
    with pytest.raises(FitError, match="monitor.key"):
        fit(
            net=model,
            loaders=TrainOnlyLoaders(train=loader),
            steps=steps(),
            metrics=TrainOnlyMetrics(
                train_metrics=specs,
                monitor=MonitorConfig(
                    key=metric_log_name("loss", "val", LEVEL_EPOCH, "mean"),
                    mode="min",
                ),
            ),
            optimization=OptimizationConfig(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
                scheduler=OptimizationConfig.ABSENT,
            ),
            trainer=cpu_trainer(
                max_epochs=1,
                accumulate_grad_batches=1,
                early_stopping=EarlyStoppingConfig.none(),
            ),
            checkpointer=checkpointer(
                tmp_path / "run",
                max_epochs=1,
                monitor_key=loss_monitor_key(split="train"),
            ),
            logger=RecordingLogger(),
            run_log=run_log(),
        )
