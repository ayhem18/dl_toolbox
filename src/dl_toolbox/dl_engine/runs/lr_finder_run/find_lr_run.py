"""find_lr: snapshot, sweep, restore, suggest, write the curve and plot, log, close."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory, DirectorySpec
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import (
    JsonTextForm,
    dump_object,
    flatten_object,
    write_object,
)
from dl_toolbox.base.packaging import require
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of
from dl_toolbox.dl_engine.dl_engine_utils.trainable_snapshot import (
    restore_trainable,
    snapshot_trainable,
)
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream, Logger
from dl_toolbox.dl_engine.runs.common.run_config import RunLogConfig
from dl_toolbox.dl_engine.runs.lr_finder_run.range_test import (
    run_full_ds_range_test,
    run_range_test,
    suggest_lrs,
    suggest_min_loss_lr,
)
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrError,
    FindLrRuntime,
    FindLrSpec,
    FindLrSteps,
    resolve_device,
)


class FindLrRun:
    CURVE_FILENAME = "lr_curve.json"
    PLOT_FILENAME = "lr_curve.png"
    TAG_LR = "lr_finder/lr"
    TAG_LOSS = "lr_finder/loss"
    TAG_TABLE = "lr_finder/curve"
    TAG_PLOT = "lr_finder/plot"
    SUGGESTED_PREFIX = "suggested/"


class FindLrResult:
    def __init__(
        self,
        *,
        kind: str,
        lrs: tuple[float, ...],
        losses: tuple[float, ...],
        suggested: Mapping[str, float],
        curve: File,
        plot: File,
    ) -> None:
        self.kind = parse_one_of(
            kind,
            allowed=FindLrSpec.KINDS,
            label="kind",
            error_type=FindLrError,
            error_message=one_of_message,
        )
        if not isinstance(lrs, tuple) or not isinstance(losses, tuple):
            raise TypeError("lrs and losses must be tuples of float")
        if len(lrs) != len(losses):
            raise FindLrError("lrs and losses must be the same length")
        if not isinstance(suggested, Mapping):
            raise TypeError(f"suggested must be a mapping, got {type(suggested)!r}")
        if not isinstance(curve, File):
            raise TypeError(f"curve must be File, got {type(curve)!r}")
        if not isinstance(plot, File):
            raise TypeError(f"plot must be File, got {type(plot)!r}")
        self.lrs = lrs
        self.losses = losses
        self.suggested = dict(suggested)
        self.curve = curve
        self.plot = plot


def plot_lr_curve(
    *,
    artifacts: Directory,
    lrs: Sequence[float],
    losses: Sequence[float],
    suggested: Mapping[str, float],
    ylabel: str) -> tuple[File, object]:
    if not isinstance(artifacts, Directory):
        raise TypeError(f"artifacts must be Directory, got {type(artifacts)!r}")
    if artifacts.spec != DirectorySpec.EXISTS:
        raise ValueError(f"artifacts Directory must exist: {artifacts.path}")
    if not isinstance(ylabel, str) or not ylabel:
        raise ValueError(f"ylabel must be a non-empty str, got {ylabel!r}")
    require(package="matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    axes.semilogx(list(lrs), list(losses))
    axes.set_xlabel("learning rate")
    axes.set_ylabel(ylabel)
    names = list(suggested)
    n_names = len(names)
    styles = {
        "steepest": ("C1", "--"),
        "before_min": ("C2", ":"),
        "min_loss_lr": ("C3", "-."),
    }
    for i in range(n_names):
        name = names[i]
        color, linestyle = styles[name] if name in styles else ("C7", "-")
        axes.axvline(
            float(suggested[name]),
            color=color,
            linestyle=linestyle,
            label=name,
        )
    if n_names > 0:
        axes.legend()
    path = artifacts.path / FindLrRun.PLOT_FILENAME
    figure.savefig(path)
    canvas.draw()
    buffer = canvas.buffer_rgba()
    import numpy as np

    image = np.asarray(buffer)[:, :, :3]
    figure.clear()
    return File.from_absolute(path=path), image


def find_lr(
    *,
    net: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    steps: FindLrSteps,
    spec: FindLrSpec,
    runtime: FindLrRuntime,
    logger: Logger,
    artifacts: Directory,
    run_log: RunLogConfig,
) -> FindLrResult:
    if not isinstance(net, nn.Module):
        raise TypeError(f"net must be nn.Module, got {type(net)!r}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            f"train_loader must be DataLoader, got {type(train_loader)!r}"
        )
    if not isinstance(steps, FindLrSteps):
        raise TypeError(f"steps must be FindLrSteps, got {type(steps)!r}")
    if not isinstance(spec, FindLrSpec):
        raise TypeError(f"spec must be FindLrSpec, got {type(spec)!r}")
    if not isinstance(runtime, FindLrRuntime):
        raise TypeError(f"runtime must be FindLrRuntime, got {type(runtime)!r}")
    if not isinstance(logger, Logger):
        raise TypeError(f"logger must be a dl_toolbox Logger, got {type(logger)!r}")
    if not isinstance(artifacts, Directory):
        raise TypeError(f"artifacts must be Directory, got {type(artifacts)!r}")
    if artifacts.spec != DirectorySpec.EXISTS:
        raise ValueError(f"artifacts Directory must exist: {artifacts.path}")
    if not isinstance(run_log, RunLogConfig):
        raise TypeError(f"run_log must be RunLogConfig, got {type(run_log)!r}")
    kind = parse_one_of(
        spec.kind,
        allowed=FindLrSpec.KINDS,
        label="kind",
        error_type=FindLrError,
        error_message=one_of_message,
    )
    require(package="matplotlib")
    device = resolve_device(runtime)
    logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
    try:
        logger.log_params(flatten_object(run_log.payload))
        logger.log_text(
            run_log.filename,
            dump_object(run_log.payload, form=JsonTextForm.ASCII_PRETTY),
            step=0,
        )
        net.to(device)
        snapshot = snapshot_trainable(model=net, optimizer=optimizer)
        try:
            if kind == FindLrSpec.RANGE:
                lrs, losses = run_range_test(
                    net=net,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    steps=steps,
                    spec=spec,
                    device=device,
                    runtime=runtime,
                )
                suggested = suggest_lrs(lrs, losses)
                ylabel = "smoothed loss"
            else:
                lrs, losses = run_full_ds_range_test(
                    net=net,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    steps=steps,
                    spec=spec,
                    device=device,
                    runtime=runtime,
                    snapshot=snapshot,
                )
                suggested = suggest_min_loss_lr(lrs, losses)
                ylabel = "eval loss"
        finally:
            restore_trainable(
                model=net, optimizer=optimizer, snapshot=snapshot, device=device
            )
        n_points = len(lrs)
        for i in range(n_points):
            logger.log_scalars(
                {FindLrRun.TAG_LR: float(lrs[i]), FindLrRun.TAG_LOSS: float(losses[i])},
                step=i,
            )
        plot_file, plot_image = plot_lr_curve(
            artifacts=artifacts,
            lrs=lrs,
            losses=losses,
            suggested=suggested,
            ylabel=ylabel,
        )
        logger.log_image(FindLrRun.TAG_PLOT, plot_image, step=0)
        curve = write_object(
            directory=artifacts,
            name=FindLrRun.CURVE_FILENAME,
            payload={
                "kind": kind,
                "lrs": list(lrs),
                "losses": list(losses),
                "suggested": dict(suggested),
            },
            form=JsonTextForm.DOCUMENT,
        )
        last = n_points - 1
        suggested_scalars: dict[str, float] = {}
        suggested_keys = list(suggested)
        n_suggested = len(suggested_keys)
        for i in range(n_suggested):
            key = suggested_keys[i]
            suggested_scalars[FindLrRun.SUGGESTED_PREFIX + key] = float(suggested[key])
        logger.log_scalars(suggested_scalars, step=last)
        rows: list[dict[str, float]] = []
        for i in range(n_points):
            rows.append({"lr": float(lrs[i]), "loss": float(losses[i])})
        logger.log_table(FindLrRun.TAG_TABLE, rows, step=last)
    finally:
        logger.close()
    return FindLrResult(
        kind=kind,
        lrs=lrs,
        losses=losses,
        suggested=suggested,
        curve=curve,
        plot=plot_file,
    )
