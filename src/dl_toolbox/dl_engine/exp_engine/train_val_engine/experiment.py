"""Return types of experiment.build_fit / build_eval. Not JSON."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torch import nn

from dl_toolbox.dl_engine.runs.common.run_config import FitSteps, OptimizationConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalContext,
    EvalLoaders,
    LogItem,
)
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import (
    TrainValLoaders,
    TrainValMetrics,
)


class TrainValFitParts:
    def __init__(
        self,
        *,
        net: nn.Module,
        loaders: TrainValLoaders,
        steps: FitSteps,
        metrics: TrainValMetrics,
        optimization: OptimizationConfig,
    ) -> None:
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
        self.net = net
        self.loaders = loaders
        self.steps = steps
        self.metrics = metrics
        self.optimization = optimization


class TrainValEvalParts:
    def __init__(
        self,
        *,
        loaders: EvalLoaders,
        evaluate_fn: Callable[[EvalContext], Sequence[LogItem]],
    ) -> None:
        if not isinstance(loaders, EvalLoaders):
            raise TypeError(f"loaders must be EvalLoaders, got {type(loaders)!r}")
        if not callable(evaluate_fn):
            raise TypeError("evaluate_fn must be callable")
        self.loaders = loaders
        self.evaluate_fn = evaluate_fn
