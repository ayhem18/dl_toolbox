"""Optional semilog LR curve. Matplotlib is not a core dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def plot_lr_curve(
    path: Path,
    lrs: Sequence[float],
    losses: Sequence[float],
    suggested: Mapping[str, float],
) -> Path | None:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return None
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.semilogx(list(lrs), list(losses))
    axes.set_xlabel("learning rate")
    axes.set_ylabel("smoothed loss")
    axes.axvline(suggested["steepest"], color="C1", linestyle="--", label="steepest")
    axes.axvline(suggested["before_min"], color="C2", linestyle=":", label="before_min")
    axes.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.clear()
    return path
