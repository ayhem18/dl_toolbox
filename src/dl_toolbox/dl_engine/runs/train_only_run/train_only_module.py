"""Train-only LightningModule: the loop as methods. No val methods."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import lightning as L
from torch import Tensor, nn
from torch.optim import Optimizer

from dl_toolbox.base.input_output.json_io import dump_object, flatten_object, JsonTextForm
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream, Logger
from dl_toolbox.dl_engine.metrics.accumulator import MetricAccumulator
from dl_toolbox.dl_engine.runs.common.run_config import (
    FitSteps,
    OptimizationConfig,
    RunLogConfig,
)
from dl_toolbox.dl_engine.runs.common.run_loop import (
    TRACK_PROG_BAR,
    configured_optimizers,
    epoch_metric_keys,
    log_epoch_row,
    prepare_batch,
)
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import TrainOnlyMetrics


class TrainOnlyLightningModule(L.LightningModule):
    def __init__(
        self,
        *,
        net: nn.Module,
        steps: FitSteps,
        metrics: TrainOnlyMetrics,
        optimization: OptimizationConfig,
        logger: Logger,
        run_log: RunLogConfig,
    ) -> None:
        super().__init__()
        if not isinstance(net, nn.Module):
            raise TypeError(f"net must be nn.Module, got {type(net)!r}")
        if not isinstance(steps, FitSteps):
            raise TypeError(f"steps must be FitSteps, got {type(steps)!r}")
        if not isinstance(metrics, TrainOnlyMetrics):
            raise TypeError(f"metrics must be TrainOnlyMetrics, got {type(metrics)!r}")
        if not isinstance(optimization, OptimizationConfig):
            raise TypeError(
                f"optimization must be OptimizationConfig, got {type(optimization)!r}"
            )
        if not isinstance(logger, Logger):
            raise TypeError(f"logger must be a dl_toolbox Logger, got {type(logger)!r}")
        if not isinstance(run_log, RunLogConfig):
            raise TypeError(f"run_log must be RunLogConfig, got {type(run_log)!r}")
        self.net = net
        self._steps = steps
        self._metrics = metrics
        self._optimization = optimization
        self._epoch_logger = logger
        self._run_log = run_log
        self._train_accumulator: MetricAccumulator | None = None

    def on_fit_start(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self._epoch_logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        self._epoch_logger.log_params(flatten_object(self._run_log.payload))
        self._epoch_logger.log_text(
            self._run_log.filename,
            dump_object(self._run_log.payload, form=JsonTextForm.ASCII_PRETTY),
            step=0,
        )

    def on_train_epoch_start(self) -> None:
        self._train_accumulator = MetricAccumulator(
            self._metrics.train_metrics, split="train"
        )

    def training_step(self, raw_batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        batch = prepare_batch(
            self, raw_batch, prepare_tensors_fn=self._steps.prepare_tensors_fn
        )
        output = self._steps.forward_fn(self.net, batch)
        loss = self._steps.loss_fn(output, batch)
        if self._train_accumulator is None:
            raise ValueError("train accumulator is None in training_step")
        with torch.no_grad():
            self._train_accumulator.update(
                self._steps.build_metric_inputs_fn(output, batch)
            )
        return loss

    def on_train_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if self._train_accumulator is None:
            raise ValueError("train accumulator is None in on_train_epoch_end")
        expected = epoch_metric_keys(self._metrics.train_metrics, split="train")
        log_epoch_row(
            self,
            self._train_accumulator.epoch_scalars(),
            track=TRACK_PROG_BAR,
            expected_keys=expected,
            epoch_logger=self._epoch_logger,
        )
        if self._metrics.monitor.key not in expected:
            raise ValueError(
                f"monitor.key {self._metrics.monitor.key!r} missing from train "
                f"epoch metrics: {sorted(expected)}"
            )

    def on_fit_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self._epoch_logger.close()

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        return configured_optimizers(self._optimization)

    @classmethod
    def load_net(
        cls,
        *,
        net: nn.Module,
        file: Path,
        map_location: torch.device,
    ) -> nn.Module:
        del cls
        if not isinstance(net, nn.Module):
            raise TypeError(f"net must be nn.Module, got {type(net)!r}")
        if not isinstance(file, Path):
            raise TypeError(f"file must be Path, got {type(file)!r}")
        if not file.is_absolute():
            raise ValueError(f"file must be an absolute Path, got {file}")
        if not isinstance(map_location, torch.device):
            raise TypeError(
                f"map_location must be torch.device, got {type(map_location)!r}"
            )
        payload = torch.load(file, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(
                "checkpoint must be a dict with key 'state_dict', "
                f"got {type(payload)!r}"
            )
        state = payload["state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError(
                f"state_dict must be a mapping, got {type(state)!r}"
            )
        shell = nn.Module()
        shell.add_module("net", net)
        try:
            incompatible = shell.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"checkpoint state_dict does not match net: {exc}"
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise ValueError(
                "checkpoint state_dict does not match net: "
                f"missing={missing} unexpected={unexpected}"
            )
        return net
