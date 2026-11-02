"""Optional semilog LR curve. Matplotlib is not a core dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


_VLINE_STYLES = {
    "steepest": ("C1", "--"),
    "before_min": ("C2", ":"),
    "min_loss_lr": ("C3", "-."),
}


def plot_lr_curve(
    path: Path,
    lrs: Sequence[float],
    losses: Sequence[float],
    suggested: Mapping[str, float],
    *,
    ylabel: str = "smoothed loss",
) -> Path | None:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return None
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.semilogx(list(lrs), list(losses))
    axes.set_xlabel("learning rate")
    axes.set_ylabel(ylabel)
    names = list(suggested)
    n_names = len(names)
    for i in range(n_names):
        name = names[i]
        color, style = _VLINE_STYLES.get(name, ("C7", "-"))
        axes.axvline(
            float(suggested[name]), color=color, linestyle=style, label=name
        )
    if n_names > 0:
        axes.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.clear()
    return path
