"""Shared loop helpers. Epoch keys, device move, dual log, optimizer pair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import lightning as L
from torch import Tensor
from torch.optim import Optimizer

from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.metrics.metric_object import (
    LEVEL_EPOCH,
    MetricSpec,
    metric_log_name,
)
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig

# Epoch-end self.log is the only distributed sync. MetricAccumulator is
# rank-local; sync_dist=True averages those rank-local epoch scalars
# (mean-of-means). That equals a true global mean when each rank saw the
# same number of items (DistributedSampler + drop_last). The dl_toolbox
# logger writes on rank 0 only. Do not self.log per batch.
TRACK_PROG_BAR = {
    "on_step": False,
    "on_epoch": True,
    "prog_bar": True,
    "logger": False,
    "sync_dist": True,
}
TRACK_SILENT = {
    "on_step": False,
    "on_epoch": True,
    "prog_bar": False,
    "logger": False,
    "sync_dist": True,
}


def epoch_metric_keys(specs: Sequence[MetricSpec], *, split: str) -> list[str]:
    keys: list[str] = []
    n_specs = len(specs)
    for i in range(n_specs):
        metric, _batch_reducers, epoch_reducers = specs[i]
        n_reducers = len(epoch_reducers)
        # TODO: replace with .extend()
        for j in range(n_reducers):
            keys.append(
                metric_log_name(metric.name, split, LEVEL_EPOCH, epoch_reducers[j])
            )
    return keys


def configured_optimizers(
    optimization: OptimizationConfig,
) -> Optimizer | dict[str, Any]:
    optimizer = optimization.optimizer
    if optimization.scheduler is OptimizationConfig.ABSENT:
        return optimizer

    # TODO: it is not made explicit that only epoch-level learning rate schedulers are supported. This is buried deep in the code.
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": optimization.scheduler,
            "interval": "epoch",
        },
    }


def log_epoch_row(
    module: L.LightningModule,
    row: Mapping[str, float],
    *,
    track: Mapping[str, Any],
    expected_keys: Sequence[str],
    epoch_logger: Logger,
) -> None:
    epoch = int(module.current_epoch)
    keys = list(expected_keys)
    n_keys = len(keys)
    for i in range(n_keys):
        key = keys[i]
        value = row.get(key)
        if value is None:
            value = 0.0
        module.log(key, float(value), **track)
    if n_keys == 0 or not module.trainer.is_global_zero:
        return
    logged: dict[str, float] = {}
    metrics = module.trainer.callback_metrics
    for i in range(n_keys):
        key = keys[i]
        metric = metrics.get(key) # .get() 
        if metric is None:
            logged[key] = float(row.get(key, 0.0)) # I see .get() with default values ... this means either the contract is not clear: or the validation is missing
        else:
            logged[key] = float(metric.detach().cpu()) 
    epoch_logger.log_scalars(logged, step=epoch)


def prepare_batch(
    module: L.LightningModule,
    raw_batch: Any,
    *,
    prepare_tensors_fn: Any,
) -> dict[str, Any]:
    batch_as_dict = prepare_tensors_fn(raw_batch)
    if not isinstance(batch_as_dict, dict):
        raise TypeError(
            f"prepare_tensors_fn must return a dictionary, got {type(batch_as_dict)}"
        )
    prepared: dict[str, Any] = {}
    keys = list(batch_as_dict)
    n_keys = len(keys)
    for i in range(n_keys):
        key = keys[i]
        value = batch_as_dict[key]
        if isinstance(value, Tensor):
            prepared[key] = value.to(module.device)
        else:
            prepared[key] = value
    return prepared
