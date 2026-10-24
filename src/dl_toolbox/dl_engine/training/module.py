from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from dl_toolbox.dl_engine.metrics.accumulator import MetricAccumulator
from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.loggers.base import BaseLogger

_TRACK_PROG_BAR = {"on_step": False, "on_epoch": True, "prog_bar": True, "logger": False}
_TRACK_SILENT = {"on_step": False, "on_epoch": True, "prog_bar": False, "logger": False}


class ClassificationLightningModule(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_logger: BaseLogger,
        *,
        prepare_tensors: Callable[[Any], Mapping[str, Tensor]],
        forward_fn: Callable[[nn.Module, Mapping[str, Tensor]], Tensor],
        loss_fn: Callable[[Tensor, Mapping[str, Tensor]], Tensor],
        build_metric_inputs: Callable[[Tensor, Mapping[str, Tensor]], Mapping[str, Tensor]],
        train_metrics: Sequence[MetricSpec],
        val_metrics: Sequence[MetricSpec],
        monitor_key: str,
        run_config: Mapping[str, Any],
        run_config_filename: str,
    ) -> None:
        super().__init__()
        self.net = net
        self._optimizer = optimizer
        self.epoch_logger = epoch_logger
        self.prepare_tensors = prepare_tensors
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.build_metric_inputs = build_metric_inputs
        self.train_metrics = list(train_metrics)
        self.val_metrics = list(val_metrics)
        self.monitor_key = monitor_key
        self._run_config = dict(run_config)
        self._run_config_filename = run_config_filename
        self._train_accumulator: MetricAccumulator | None = None
        self._val_accumulator: MetricAccumulator | None = None

    def _prepare(self, raw_batch: Any) -> dict[str, Tensor]:
        batch = self.prepare_tensors(raw_batch)
        return {key: value.to(self.device) for key, value in batch.items()}

    def training_step(self, raw_batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        batch = self._prepare(raw_batch)
        output = self.forward_fn(self.net, batch)
        loss = self.loss_fn(output, batch)
        if self._train_accumulator is not None:
            with torch.no_grad():
                self._train_accumulator.update(self.build_metric_inputs(output, batch))
        return loss

    def validation_step(self, raw_batch: Any, batch_idx: int) -> None:
        del batch_idx
        batch = self._prepare(raw_batch)
        with torch.inference_mode():
            output = self.forward_fn(self.net, batch)
        if self._val_accumulator is not None:
            self._val_accumulator.update(self.build_metric_inputs(output, batch))

    def on_fit_start(self) -> None:
        self.epoch_logger.log_config(self._run_config, self._run_config_filename)

    def on_fit_end(self) -> None:
        self.epoch_logger.close()

    def on_train_epoch_start(self) -> None:
        self._train_accumulator = MetricAccumulator(self.train_metrics, split="train")

    def on_validation_epoch_start(self) -> None:
        self._val_accumulator = MetricAccumulator(self.val_metrics, split="val")

    def on_train_epoch_end(self) -> None:
        if self.trainer.sanity_checking or self._train_accumulator is None:
            return
        epoch = int(self.current_epoch)
        for key, value in self._train_accumulator.epoch_scalars().items():
            self._record_scalar(epoch, key, value)
            self.log(key, value, **_TRACK_SILENT)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking or self._val_accumulator is None:
            return
        epoch = int(self.current_epoch)
        val_row = self._val_accumulator.epoch_scalars()
        for key, value in val_row.items():
            self._record_scalar(epoch, key, value)
            track = _TRACK_PROG_BAR if key == self.monitor_key else _TRACK_SILENT
            self.log(key, value, **track)
        if self.monitor_key not in val_row:
            raise ValueError(
                f"monitor_key {self.monitor_key!r} missing from val epoch metrics: {sorted(val_row)}"
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer

    def _record_scalar(self, step: int, tag: str, value: float) -> None:
        self.epoch_logger.log_scalar(step, tag, float(value))
        writer = getattr(self.epoch_logger, "writer", None)
        flush = getattr(writer, "flush", None)
        if callable(flush):
            flush()
