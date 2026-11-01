"""Return object of ExperimentConfigManager.build. Every field required."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.metrics.metric_object import MetricSpec
from dl_toolbox.dl_engine.training.config import MonitorConfig


class ExperimentBuild:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, dict[str, Tensor]], Tensor],
        prepare_tensors_fn: Callable[[Any], dict[str, Tensor]],
        forward_fn: Callable[[nn.Module, dict[str, Tensor]], Tensor],
        build_metric_inputs_fn: Callable[[Tensor, dict[str, Tensor]], dict[str, Tensor]],
        train_metrics: Sequence[MetricSpec],
        val_metrics: Sequence[MetricSpec],
        monitor: MonitorConfig,
        run_config: Mapping[str, Any],
        lr_scheduler_factory: Callable[[Optimizer], LRScheduler] | None = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be nn.Module, got {type(model)!r}")
        if not isinstance(monitor, MonitorConfig):
            raise TypeError(f"monitor must be MonitorConfig, got {type(monitor)!r}")
        if not isinstance(run_config, Mapping):
            raise TypeError(f"run_config must be a mapping, got {type(run_config)!r}")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prepare_tensors_fn = prepare_tensors_fn
        self.forward_fn = forward_fn
        self.build_metric_inputs_fn = build_metric_inputs_fn
        self.train_metrics = list(train_metrics)
        self.val_metrics = list(val_metrics)
        self.monitor = monitor
        if accumulate_grad_batches < 1:
            raise ValueError(
                f"accumulate_grad_batches must be >= 1, got {accumulate_grad_batches}"
            )
        self.lr_scheduler_factory = lr_scheduler_factory
        self.accumulate_grad_batches = accumulate_grad_batches
        self.run_config = dict(run_config)
