"""Labeled 2D scatter. Matplotlib is optional."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dl_toolbox.base.packaging import require

SPLIT_TRAIN = 0
SPLIT_VAL = 1
_MARKERS = {SPLIT_TRAIN: "o", SPLIT_VAL: "x"}
_SPLIT_LABELS = {SPLIT_TRAIN: "train", SPLIT_VAL: "val"}


def plot_labeled_scatter(
    path: Path,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: tuple[str, ...],
    splits: np.ndarray | None = None,
    title: str = "",
) -> Path | None:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {tuple(points.shape)}")
    y = np.asarray(labels).reshape(-1)
    if y.shape[0] != points.shape[0]:
        raise ValueError(
            f"labels length {y.shape[0]} != xy rows {points.shape[0]}"
        )
    try:
        require(package="matplotlib")
    except ImportError:
        return None
    from matplotlib.figure import Figure

    n_classes = len(class_names)
    fig = Figure(figsize=(7.0, 6.0))
    axes = fig.add_subplot(1, 1, 1)
    if splits is None:
        split_codes = np.zeros(points.shape[0], dtype=np.int64)
    else:
        split_codes = np.asarray(splits).reshape(-1)
        if split_codes.shape[0] != points.shape[0]:
            raise ValueError(
                f"splits length {split_codes.shape[0]} != xy rows {points.shape[0]}"
            )
    split_values = (SPLIT_TRAIN, SPLIT_VAL)
    for class_i in range(n_classes):
        for s in range(len(split_values)):
            split = split_values[s]
            mask = (y == class_i) & (split_codes == split)
            if not np.any(mask):
                continue
            axes.scatter(
                points[mask, 0],
                points[mask, 1],
                marker=_MARKERS[split],
                label=f"{class_names[class_i]} / {_SPLIT_LABELS[split]}",
                alpha=0.8,
                s=28,
            )
    axes.set_title(title)
    axes.set_xlabel("dim 1")
    axes.set_ylabel("dim 2")
    axes.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    fig.clear()
    return path
