"""Query | neighbor grayscale strip. Matplotlib is optional."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dl_toolbox.dl_engine_utils.extras import require_matplotlib
from dl_toolbox.dl_engine_utils.images.gray import as_gray2d


def neighbor_sheet(
    path: Path,
    query: object,
    neighbors: Sequence[object],
    *,
    query_title: str,
    neighbor_titles: Sequence[str],
) -> Path | None:
    n_neighbors = len(neighbors)
    if n_neighbors != len(neighbor_titles):
        raise ValueError(
            f"neighbors {n_neighbors} != neighbor_titles {len(neighbor_titles)}"
        )
    try:
        require_matplotlib()
    except ImportError:
        return None
    from matplotlib.figure import Figure

    n_panels = 1 + n_neighbors
    titles: list[str] = [""] * n_panels
    planes: list[np.ndarray] = [np.empty((1, 1))] * n_panels
    titles[0] = query_title
    planes[0] = as_gray2d(query)
    for i in range(n_neighbors):
        titles[i + 1] = neighbor_titles[i]
        planes[i + 1] = as_gray2d(neighbors[i])
    fig = Figure(figsize=(2.2 * n_panels, 3.2))
    axes = fig.subplots(1, n_panels)
    if n_panels == 1:
        axis_list = [axes]
    else:
        axis_list = list(axes)
    for i in range(n_panels):
        axis = axis_list[i]
        axis.imshow(planes[i], cmap="gray")
        axis.set_title(titles[i], fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    fig.clear()
    return path
