from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from dl_toolbox.dl_engine.metrics.accumulator import MetricAccumulator
from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.dl_engine.loggers.base_logger import Logger, flatten_mapping

_TRACK_PROG_BAR = {"on_step": False, "on_epoch": True, "prog_bar": True, "logger": False}
_TRACK_SILENT = {"on_step": False, "on_epoch": True, "prog_bar": False, "logger": False}


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

    8. on_fit_end: close the logger. Lightning usually calls this; fit_classification also closes in a finally block.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_logger: Logger,
        *,
        prepare_tensors_fn: Callable[[Any], Mapping[str, Tensor]],
        forward_fn: Callable[[nn.Module, Mapping[str, Tensor]], Tensor],
        loss_fn: Callable[[Tensor, Mapping[str, Tensor]], Tensor],
        build_metric_inputs_fn: Callable[[Tensor, Mapping[str, Tensor]], Mapping[str, Tensor]],
        train_metrics: Sequence[MetricSpec],
        val_metrics: Sequence[MetricSpec],
        monitor_key: str,
        run_config: Mapping[str, Any],
        run_config_filename: str,
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
        self._train_accumulator: MetricAccumulator | None = None
        self._val_accumulator: MetricAccumulator | None = None

    ################################## Training Loop functions ##################################

    ### 1. on_fit_start: log the configuration
    def on_fit_start(self) -> None:
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
        epoch = int(self.current_epoch)
        row = self._train_accumulator.epoch_scalars()
        for key, value in row.items():
            self.log(key, value, **_TRACK_SILENT)
        if row:
            self.epoch_logger.log_scalars(row, step=epoch)

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
        epoch = int(self.current_epoch)
        val_row = self._val_accumulator.epoch_scalars()

        # maybe this needs to be made configurable. I do not see the problem with logging
        # multiple metrics instead of just the monitor key.
        for key, value in val_row.items():
            self.log(key, value, **_TRACK_PROG_BAR)
        if val_row:
            self.epoch_logger.log_scalars(val_row, step=epoch)

        if self.monitor_key not in val_row:
            raise ValueError(
                f"monitor_key {self.monitor_key!r} missing from val epoch metrics: {sorted(val_row)}"
            )

    ################################### Fit End functions ##################################
    def on_fit_end(self) -> None:
        self.epoch_logger.close()

    ################################## optimizer related functions ##################################

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # later it needs to be changed to work with learning rate schedulers.
        # the class will later be updated to accept a configuration of some sort.
        return self._optimizer

    ################################## helper functions ##################################

    def _prepare(self, raw_batch: Any) -> dict[str, Tensor]:
        # the raw batch might be a tuple, list, raw torch.Tensor... The caller is expected to own the logic
        # for converting to a dictionary with the expected keys.
        batch_as_dict = self.prepare_tensors_fn(raw_batch)

        if not isinstance(batch_as_dict, dict):
            raise TypeError(
                f"prepare_tensors_fn must return a dictionary, got {type(batch_as_dict)}"
            )

        return {key: value.to(self.device) for key, value in batch_as_dict.items()}
