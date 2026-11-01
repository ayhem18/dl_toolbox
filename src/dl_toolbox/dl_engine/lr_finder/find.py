"""Public entry: one request object, log the curve, write artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.dl_engine.loggers.base_logger import Logger, flatten_mapping
from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError
from dl_toolbox.dl_engine.lr_finder.loop import RangeTestInput, run_range_test
from dl_toolbox.dl_engine.lr_finder.plot import plot_lr_curve
from dl_toolbox.dl_engine.lr_finder.suggest import suggest_lrs

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
    logger = request.logger
    logger.setup()
    try:
        logger.log_params(flatten_mapping(dict(request.run_config)))
        lrs, losses = run_range_test(request.inputs)
        _log_curve(logger, lrs, losses)
        curve_path = request.run_dir / CURVE_FILENAME
        payload: dict[str, Any] = {"lrs": list(lrs), "losses": list(losses)}
        try:
            suggested = suggest_lrs(lrs, losses)
        except LrFinderError:
            write_json(curve_path, payload)
            raise
        payload["suggested"] = dict(suggested)
        write_json(curve_path, payload)
        last_step = len(lrs) - 1 if lrs else 0
        logger.log_scalars(
            {
                "suggested/steepest": float(suggested["steepest"]),
                "suggested/before_min": float(suggested["before_min"]),
                "suggested/min_loss_lr": float(suggested["min_loss_lr"]),
            },
            step=last_step,
        )
        logger.log_table(
            "lr_finder/curve",
            _curve_rows(lrs, losses),
            step=last_step,
        )
        plot_path = plot_lr_curve(
            request.run_dir / PLOT_FILENAME, lrs, losses, suggested
        )
        return LrFinderResult(
            lrs=lrs,
            losses=losses,
            suggested=dict(suggested),
            curve_path=curve_path,
            plot_path=plot_path,
        )
    finally:
        logger.close()


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
