"""Knobs for one frozen-representation inspect pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationProbeConfig:
    k: int = 20
    crowding_extremes: int = 3
    tsne_max_points: int = 2000
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000
    write_embeddings: bool = True
    random_state: int = 0
