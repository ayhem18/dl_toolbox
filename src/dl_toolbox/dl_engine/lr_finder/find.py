"""Public entry: one request object, log the curve, write artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.dl_engine.loggers.base_logger import Logger, flatten_mapping
from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError
from dl_toolbox.dl_engine.lr_finder.loop import RangeTestInput, run_range_test
from dl_toolbox.dl_engine.lr_finder.lr_finder_loop_full_ds import run_full_ds_range_test
from dl_toolbox.dl_engine.lr_finder.plot import plot_lr_curve
from dl_toolbox.dl_engine.lr_finder.suggest import suggest_lrs, suggest_min_loss_lr

CURVE_FILENAME = "lr_curve.json"
PLOT_FILENAME = "lr_curve.png"


@dataclass
class LrFinderRequest:
    inputs: RangeTestInput
    logger: Logger
    run_dir: Path
    run_config: Mapping[str, Any]


@dataclass(frozen=True)
class LrFinderResult:
    lrs: tuple[float, ...]
    losses: tuple[float, ...]
    suggested: dict[str, float]
    curve_path: Path
    plot_path: Path | None


def find_lr(request: LrFinderRequest) -> LrFinderResult:
    return _find(
        request,
        run_fn=run_range_test,
        suggest_fn=suggest_lrs,
        ylabel="smoothed loss",
    )


def find_lr_full_ds(request: LrFinderRequest) -> LrFinderResult:
    return _find(
        request,
        run_fn=run_full_ds_range_test,
        suggest_fn=suggest_min_loss_lr,
        ylabel="subset loss",
    )


def _find(
    request: LrFinderRequest,
    *,
    run_fn: Callable[
        [RangeTestInput], tuple[Sequence[float], Sequence[float]]
    ],
    suggest_fn: Callable[
        [Sequence[float], Sequence[float]], dict[str, float]
    ],
    ylabel: str,
) -> LrFinderResult:
    logger = request.logger
    logger.setup()
    try:
        logger.log_params(flatten_mapping(dict(request.run_config)))
        lrs, losses = run_fn(request.inputs)
        curve_lrs = tuple(lrs)
        curve_losses = tuple(losses)
        _log_curve(logger, curve_lrs, curve_losses)
        curve_path = request.run_dir / CURVE_FILENAME
        payload: dict[str, Any] = {
            "lrs": list(curve_lrs),
            "losses": list(curve_losses),
        }
        try:
            suggested = suggest_fn(curve_lrs, curve_losses)
        except LrFinderError:
            write_json(curve_path, payload)
            raise
        payload["suggested"] = dict(suggested)
        write_json(curve_path, payload)
        last_step = len(curve_lrs) - 1 if curve_lrs else 0
        logger.log_scalars(_suggested_scalars(suggested), step=last_step)
        logger.log_table(
            "lr_finder/curve",
            _curve_rows(curve_lrs, curve_losses),
            step=last_step,
        )
        plot_path = plot_lr_curve(
            request.run_dir / PLOT_FILENAME,
            curve_lrs,
            curve_losses,
            suggested,
            ylabel=ylabel,
        )
        return LrFinderResult(
            lrs=curve_lrs,
            losses=curve_losses,
            suggested=dict(suggested),
            curve_path=curve_path,
            plot_path=plot_path,
        )
    finally:
        logger.close()


def _suggested_scalars(suggested: Mapping[str, float]) -> dict[str, float]:
    keys = list(suggested)
    n = len(keys)
    logged: dict[str, float] = {}
    for i in range(n):
        key = keys[i]
        logged[f"suggested/{key}"] = float(suggested[key])
    return logged


def _log_curve(
    logger: Logger, lrs: tuple[float, ...], losses: tuple[float, ...]
) -> None:
    n = len(lrs)
    for i in range(n):
        logger.log_scalars(
            {
                "lr_finder/lr": float(lrs[i]),
                "lr_finder/loss": float(losses[i]),
            },
            step=i,
        )


def _curve_rows(
    lrs: tuple[float, ...], losses: tuple[float, ...]
) -> list[dict[str, float]]:
    n = len(lrs)
    rows: list[dict[str, float]] = []
    for i in range(n):
        rows.append({"lr": float(lrs[i]), "loss": float(losses[i])})
    return rows
