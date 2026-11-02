"""Cosine kNN classifier. Train is the gallery, val is the query set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.errors import (
    RepresentationEvalError,
)
from dl_toolbox.dl_engine.representation_eval.common.feature_bank import (
    FeatureBank,
    l2_normalize,
)
from dl_toolbox.dl_engine_utils.classification.pred_metrics import (
    classification_pred_metrics,
    pred_metrics_as_scalars,
)


@dataclass(frozen=True)
class KnnResult:
    k: int
    pred: np.ndarray
    neighbor_index: np.ndarray
    neighbor_distance: np.ndarray
    mean_k_distance: np.ndarray
    scalars: dict[str, float]


def knn_probe(train: FeatureBank, val: FeatureBank, *, k: int) -> KnnResult:
    if train.class_names != val.class_names:
        raise RepresentationEvalError("train and val class_names must match")
    train.require_all_classes(split="train")
    val.require_all_classes(split="val")
    n_gallery = train.n_samples
    if k < 1:
        raise RepresentationEvalError(f"k must be >= 1, got {k}")
    used_k = k if k <= n_gallery else n_gallery
    gallery = l2_normalize(train.features)
    query = l2_normalize(val.features)
    similarity = query @ gallery.T
    top_sim, neighbor_index = _topk_similarity(similarity, used_k)
    neighbor_distance = np.clip(1.0 - top_sim, 0.0, 2.0)
    mean_k_distance = neighbor_distance.mean(axis=1)
    neighbor_labels = train.labels[neighbor_index]
    pred = _distance_weighted_vote(
        neighbor_labels,
        neighbor_distance,
        n_classes=len(train.class_names),
    )
    metrics = classification_pred_metrics(
        val.labels, pred, class_names=train.class_names
    )
    scalars = pred_metrics_as_scalars(
        metrics, class_names=train.class_names, include_precision=False
    )
    scalars["mean_neighbor_distance"] = float(mean_k_distance.mean())
    return KnnResult(
        k=used_k,
        pred=pred.astype(np.int64, copy=False),
        neighbor_index=neighbor_index.astype(np.int64, copy=False),
        neighbor_distance=neighbor_distance.astype(np.float64, copy=False),
        mean_k_distance=mean_k_distance.astype(np.float64, copy=False),
        scalars=scalars,
    )


def _topk_similarity(sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_gallery = int(sim.shape[1])
    if k > n_gallery:
        raise RepresentationEvalError(f"k={k} exceeds gallery size {n_gallery}")
    if k == n_gallery:
        index = np.argsort(-sim, axis=1)
        taken = np.take_along_axis(sim, index, axis=1)
        return taken, index
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    part_sim = np.take_along_axis(sim, part, axis=1)
    within = np.argsort(-part_sim, axis=1)
    index = np.take_along_axis(part, within, axis=1)
    taken = np.take_along_axis(part_sim, within, axis=1)
    return taken, index


def _distance_weighted_vote(
    neighbor_labels: np.ndarray,
    neighbor_distance: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    weights = 1.0 / np.clip(neighbor_distance, 1.0e-8, None)
    n_query = neighbor_labels.shape[0]
    pred = np.zeros(n_query, dtype=np.int64)
    for i in range(n_query):
        scores = np.zeros(n_classes, dtype=np.float64)
        k = neighbor_labels.shape[1]
        for j in range(k):
            scores[int(neighbor_labels[i, j])] += float(weights[i, j])
        pred[i] = int(np.argmax(scores))
    return pred
