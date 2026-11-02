"""Rank val queries by mean distance to their k train neighbors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.errors import (
    RepresentationEvalError,
)


@dataclass(frozen=True)
class CrowdingExtremes:
    top: tuple[int, ...]
    bottom: tuple[int, ...]


def rank_crowding_extremes(
    mean_k_distance: np.ndarray, *, n_extremes: int
) -> CrowdingExtremes:
    distances = np.asarray(mean_k_distance).reshape(-1)
    n = int(distances.shape[0])
    if n < 1:
        raise RepresentationEvalError("mean_k_distance is empty")
    if n_extremes < 1:
        raise RepresentationEvalError(
            f"n_extremes must be >= 1, got {n_extremes}"
        )
    take = n_extremes if n_extremes <= n else n
    order = np.argsort(distances, kind="mergesort")
    top: list[int] = [0] * take
    bottom: list[int] = [0] * take
    for i in range(take):
        top[i] = int(order[i])
        bottom[i] = int(order[n - 1 - i])
    return CrowdingExtremes(top=tuple(top), bottom=tuple(bottom))


def crowding_filename(kind: str, rank: int) -> str:
    if kind not in ("top", "bottom"):
        raise RepresentationEvalError(f"kind must be 'top' or 'bottom', got {kind!r}")
    if rank < 1:
        raise RepresentationEvalError(f"rank must be >= 1, got {rank}")
    return f"{kind}_{rank}_crowded_sample.png"
