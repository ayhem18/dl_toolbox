from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from dl_toolbox.dl_engine.loggers.logger import EXISTENCE_CREATE, STREAM_FIT
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
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import (
    TrainValLoaders,
    TrainValMetrics,
)
from dl_toolbox.dl_engine.runs.train_val_run.train_val_fit import fit
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
    train_val_metrics,
)


def test_fit_has_no_defaults() -> None:
    params = inspect.signature(fit).parameters
    for name, param in params.items():
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_fit_cpu_end_to_end(tmp_path: Path) -> None:
    specs = binary_specs()
    model = LinearBinary()
    logger = RecordingLogger()
    loader = binary_loader(n=32, batch_size=8)
    trainer = fit(
        net=model,
        loaders=TrainValLoaders(train=loader, val=loader),
        steps=steps(),
        metrics=train_val_metrics(specs),
        optimization=OptimizationConfig(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
            scheduler=OptimizationConfig.ABSENT,
        ),
        trainer=cpu_trainer(
            max_epochs=2,
            accumulate_grad_batches=1,
            early_stopping=EarlyStoppingConfig(
                name=EarlyStoppingConfig.PATIENCE, patience=3, min_delta=1e-5
            ),
        ),
        checkpointer=checkpointer(
            tmp_path / "run",
            max_epochs=2,
            monitor_key=loss_monitor_key(split="val"),
        ),
        logger=logger,
        run_log=run_log(),
    )
    assert trainer.current_epoch >= 1
    checkpoints = tmp_path / "run" / "checkpoints"
    assert (checkpoints / "catalog.json").is_file()
    ckpt_files = list((checkpoints / "live").glob("*.ckpt"))
    assert ckpt_files
    assert logger.setup_calls == [(STREAM_FIT, EXISTENCE_CREATE)]
    assert logger.close_calls >= 1
    assert logger.params
    assert logger.texts[0][0] == "run_config"
    assert logger.scalars


def test_fit_accumulate_grad_batches_and_live_scheduler(tmp_path: Path) -> None:
    specs = [(binary_bce_loss_metric(), (), ("mean",))]
    model = LinearBinary()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    logger = RecordingLogger()
    loader = binary_loader(n=16, batch_size=8)
    trainer = fit(
        net=model,
        loaders=TrainValLoaders(train=loader, val=loader),
        steps=steps(),
        metrics=train_val_metrics(specs),
        optimization=OptimizationConfig(optimizer=optimizer, scheduler=scheduler),
        trainer=cpu_trainer(
            max_epochs=1,
            accumulate_grad_batches=2,
            early_stopping=EarlyStoppingConfig.none(),
        ),
        checkpointer=checkpointer(
            tmp_path / "run",
            max_epochs=1,
            monitor_key=loss_monitor_key(split="val"),
        ),
        logger=logger,
        run_log=run_log(),
    )
    assert trainer.accumulate_grad_batches == 2
    assert logger.setup_calls == [(STREAM_FIT, EXISTENCE_CREATE)]


def test_fit_rejects_monitor_missing_from_val_metrics(tmp_path: Path) -> None:
    loss_metric = binary_bce_loss_metric()
    specs = [(loss_metric, (), ("mean",))]
    model = LinearBinary()
    loader = binary_loader(n=8, batch_size=8)
    with pytest.raises(FitError, match="monitor.key"):
        fit(
            net=model,
            loaders=TrainValLoaders(train=loader, val=loader),
            steps=steps(),
            metrics=TrainValMetrics(
                train_metrics=specs,
                val_metrics=specs,
                monitor=MonitorConfig(
                    key=metric_log_name("accuracy", "val", LEVEL_EPOCH, "acc"),
                    mode="max",
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
                monitor_key=loss_monitor_key(split="val"),
            ),
            logger=RecordingLogger(),
            run_log=run_log(),
        )
