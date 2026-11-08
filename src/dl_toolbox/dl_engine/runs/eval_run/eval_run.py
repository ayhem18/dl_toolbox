"""evaluate() for a bound checkpoint.

evaluate_fn(EvalContext) returns a sequence of LogItem. emit writes each
item. Setup uses EvalLoggerConfig.stream / existence. flatten_object /
dump_object log the RunLogConfig payload, then close.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from dl_toolbox.base.input_output.json_io import dump_object, flatten_object, JsonTextForm
from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.runs.common.run_config import RunLogConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalContext,
    EvalLoaders,
    EvalLoggerConfig,
    LogItem,
)


class EvalError(ValueError):
    pass


def evaluate(
    *,
    net: nn.Module,
    loaders: EvalLoaders,
    evaluate_fn: Callable[[EvalContext], Sequence[LogItem]],
    device: torch.device,
    logger: Logger,
    logger_config: EvalLoggerConfig,
    run_log: RunLogConfig,
) -> None:
    if not isinstance(net, nn.Module):
        raise TypeError(f"net must be nn.Module, got {type(net)!r}")
    if not isinstance(loaders, EvalLoaders):
        raise TypeError(f"loaders must be EvalLoaders, got {type(loaders)!r}")
    if not callable(evaluate_fn):
        raise TypeError("evaluate_fn must be callable")
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be torch.device, got {type(device)!r}")
    if not isinstance(logger, Logger):
        raise TypeError(f"logger must be a dl_toolbox Logger, got {type(logger)!r}")
    if not isinstance(logger_config, EvalLoggerConfig):
        raise TypeError(
            f"logger_config must be EvalLoggerConfig, got {type(logger_config)!r}"
        )
    if not isinstance(run_log, RunLogConfig):
        raise TypeError(f"run_log must be RunLogConfig, got {type(run_log)!r}")

    net.eval()
    net.to(device)

    logger.setup(stream=logger_config.stream, existence=logger_config.existence)
    try:
        logger.log_params(flatten_object(run_log.payload))
        logger.log_text(
            run_log.filename,
            dump_object(run_log.payload, form=JsonTextForm.ASCII_PRETTY),
            step=0,
        )
        context = EvalContext(
            net=net,
            loaders=loaders,
            device=device,
            step=logger_config.step,
        )
        items = evaluate_fn(context)
        _require_log_items(items)
        emit(logger, items)
    finally:
        logger.close()


def emit(logger: Logger, items: Sequence[LogItem]) -> None:
    n_items = len(items)
    for i in range(n_items):
        item = items[i]
        if item.kind == LogItem.SCALARS:
            logger.log_scalars(item.values, step=item.step)
            continue
        if item.kind == LogItem.IMAGE:
            logger.log_image(item.tag, item.image, step=item.step)
            continue
        if item.kind == LogItem.TEXT:
            logger.log_text(item.tag, item.text, step=item.step)
            continue
        if item.kind == LogItem.TABLE:
            logger.log_table(item.tag, item.rows, step=item.step)
            continue
        raise EvalError(f"unknown LogItem kind {item.kind!r}")


def _require_log_items(items: object) -> None:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise EvalError(
            f"evaluate_fn must return a sequence of LogItem, got {type(items)!r}"
        )
    n_items = len(items)
    for i in range(n_items):
        item = items[i]
        if not isinstance(item, LogItem):
            raise EvalError(
                f"evaluate_fn items[{i}] must be LogItem, got {type(item)!r}"
            )
