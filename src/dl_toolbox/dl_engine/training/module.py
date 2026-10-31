from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dl_toolbox.dl_engine.metrics.accumulator import MetricAccumulator
from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.dl_engine.loggers.base_logger import Logger, flatten_mapping

# Epoch-end self.log is the only distributed sync. MetricAccumulator is
# rank-local; sync_dist=True averages those rank-local epoch scalars
# (mean-of-means). That equals a true global mean when each rank saw the
# same number of items (DistributedSampler + drop_last). The dl_toolbox
# logger writes on rank 0 only. Do not self.log per batch.
_TRACK_PROG_BAR = {
    "on_step": False,
    "on_epoch": True,
    "prog_bar": True,
    "logger": False,
    "sync_dist": True,
}
_TRACK_SILENT = {
    "on_step": False,
    "on_epoch": True,
    "prog_bar": False,
    "logger": False,
    "sync_dist": True,
}


class ClassificationLightningModule(L.LightningModule):
    """
    Let's start by explaining the input to this class:

    1. prepare_tensors_fn: converts the raw batch into a dictionary with specific keys.
    2. forward_fn: computes the model output that is supposed to be passed to the loss function and metrics
    3. loss_fn: the callable responsible for computing the scalar tensor on which the backward pass is computed.
    4. build_metric_inputs_fn: Metric might require different inputs; maybe computed overall logits / actual predictions.

    Now, let's explain how the final training loop would look like:

    1. on_fit_start: open the logger and log the configuration
    2. on_train_epoch_start: initialize the train accumulator: a MetricAccumulator object stores the different predictions + targets across batches
    3. training_step: compute the loss and update the train accumulator
    4. on_train_epoch_end: log the train metrics

    5. on_validation_epoch_start: initialize the validation accumulator
    6. validation_step: compute the validation metrics
    7. on_validation_epoch_end: log the validation metrics

    8. on_fit_end: close the logger. Lightning usually calls this; fit also closes in a finally block.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        epoch_logger: Logger,
        *,
        prepare_tensors_fn: Callable[[Any], Mapping[str, Any]],
        forward_fn: Callable[[nn.Module, Mapping[str, Any]], Tensor],
        loss_fn: Callable[[Tensor, Mapping[str, Any]], Tensor],
        build_metric_inputs_fn: Callable[[Tensor, Mapping[str, Any]], Mapping[str, Tensor]],
        train_metrics: Sequence[MetricSpec],
        val_metrics: Sequence[MetricSpec],
        monitor_key: str,
        run_config: Mapping[str, Any],
        run_config_filename: str,
        lr_scheduler_factory: Callable[[Optimizer], LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(epoch_logger, Logger):
            raise TypeError(
                f"epoch_logger must be a dl_toolbox Logger, got {type(epoch_logger)!r}"
            )
        self.net = net
        self._optimizer = optimizer
        self.epoch_logger = epoch_logger
        self.prepare_tensors_fn = prepare_tensors_fn
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.build_metric_inputs_fn = build_metric_inputs_fn
        self.train_metrics = list(train_metrics)
        self.val_metrics = list(val_metrics)
        self.monitor_key = monitor_key
        self._run_config = dict(run_config)
        self._run_config_filename = run_config_filename
        self._lr_scheduler_factory = lr_scheduler_factory
        self._train_accumulator: MetricAccumulator | None = None
        self._val_accumulator: MetricAccumulator | None = None

    ################################## Training Loop functions ##################################

    ### 1. on_fit_start: log the configuration
    def on_fit_start(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self.epoch_logger.setup()
        self.epoch_logger.log_params(flatten_mapping(self._run_config))
        self.epoch_logger.log_text(
            self._run_config_filename,
            json.dumps(self._run_config, indent=2, ensure_ascii=True),
            step=0,
        )

    ################################### Training Epoch functions ##################################

    def on_train_epoch_start(self) -> None:
        self._train_accumulator = MetricAccumulator(self.train_metrics, split="train")

    def training_step(self, raw_batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        batch = self._prepare(raw_batch)
        output = self.forward_fn(self.net, batch)
        loss = self.loss_fn(output, batch)

        # the question is why would self._train_accumulator be None?
        # if it is initialized in on_train_epoch_start, it should not be None here.
        # the original code was:
        #
        # if self._train_accumulator is not None:
        #     with torch.no_grad():
        #         self._train_accumulator.update(self.build_metric_inputs_fn(output, batch))
        #
        # this is unacceptable: because if the accumulator is None, the training must stop.
        if self._train_accumulator is None:
            raise ValueError("train accumulator is None in training_step")

        with torch.no_grad():
            self._train_accumulator.update(self.build_metric_inputs_fn(output, batch))

        return loss

    def on_train_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if self._train_accumulator is None:
            raise ValueError("train accumulator is None in on_train_epoch_end")
        self._log_epoch_row(
            self._train_accumulator.epoch_scalars(),
            track=_TRACK_SILENT,
        )

    ################################### Validation Epoch functions ##################################
    def on_validation_epoch_start(self) -> None:
        self._val_accumulator = MetricAccumulator(self.val_metrics, split="val")

    def validation_step(self, raw_batch: Any, batch_idx: int) -> None:
        del batch_idx
        batch = self._prepare(raw_batch)
        with torch.inference_mode():
            output = self.forward_fn(self.net, batch)
        if self._val_accumulator is None:
            raise ValueError("val accumulator is None in validation_step")
        self._val_accumulator.update(self.build_metric_inputs_fn(output, batch))

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if self._val_accumulator is None:
            raise ValueError("val accumulator is None in on_validation_epoch_end")
        val_row = self._val_accumulator.epoch_scalars()

        # maybe this needs to be made configurable. I do not see the problem with logging
        # multiple metrics instead of just the monitor key.
        self._log_epoch_row(val_row, track=_TRACK_PROG_BAR)

        if self.monitor_key not in val_row:
            raise ValueError(
                f"monitor_key {self.monitor_key!r} missing from val epoch metrics: {sorted(val_row)}"
            )

    ################################### Fit End functions ##################################
    def on_fit_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self.epoch_logger.close()

    ################################## optimizer related functions ##################################

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        if self._lr_scheduler_factory is None:
            return self._optimizer
        scheduler = self._lr_scheduler_factory(self._optimizer)
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    ################################## helper functions ##################################

    def _log_epoch_row(
        self,
        row: Mapping[str, float],
        *,
        track: Mapping[str, Any],
    ) -> None:
        epoch = int(self.current_epoch)
        for key, value in row.items():
            self.log(key, value, **track)
        if not row or not self.trainer.is_global_zero:
            return
        logged: dict[str, float] = {}
        metrics = self.trainer.callback_metrics
        for key, value in row.items():
            metric = metrics.get(key)
            if metric is None:
                logged[key] = float(value)
            else:
                logged[key] = float(metric.detach().cpu())
        self.epoch_logger.log_scalars(logged, step=epoch)

    def _prepare(self, raw_batch: Any) -> dict[str, Any]:
        # the raw batch might be a tuple, list, raw torch.Tensor... The caller is expected to own the logic
        # for converting to a dictionary with the expected keys.
        batch_as_dict = self.prepare_tensors_fn(raw_batch)

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
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared
