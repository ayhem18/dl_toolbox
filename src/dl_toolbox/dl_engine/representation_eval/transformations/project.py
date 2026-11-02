"""Unsupervised 2D projections of frozen features."""

from __future__ import annotations

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.errors import (
    RepresentationEvalError,
)
from dl_toolbox.dl_engine.representation_eval.common.feature_bank import (
    FeatureBank,
    l2_normalize,
)
from dl_toolbox.dl_engine_utils.extras import require_sklearn
from dl_toolbox.dl_engine_utils.images.scatter import SPLIT_TRAIN, SPLIT_VAL

METHOD_PCA = "pca"
METHOD_TSNE = "tsne"


def concat_train_val(
    train: FeatureBank, val: FeatureBank
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.class_names != val.class_names:
        raise RepresentationEvalError("train and val class_names must match")
    features = np.concatenate(
        (l2_normalize(train.features), l2_normalize(val.features)), axis=0
    )
    labels = np.concatenate((train.labels, val.labels), axis=0)
    splits = np.concatenate(
        (
            np.full(train.n_samples, SPLIT_TRAIN, dtype=np.int64),
            np.full(val.n_samples, SPLIT_VAL, dtype=np.int64),
        ),
        axis=0,
    )
    return features, labels, splits


def project_2d(
    features: np.ndarray,
    *,
    method: str,
    random_state: int = 0,
    tsne_perplexity: float = 30.0,
    tsne_max_iter: int = 1000,
    max_points: int | None = None,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise RepresentationEvalError(
            f"features must have shape (N, D), got {tuple(array.shape)}"
        )
    n = int(array.shape[0])
    keep = np.arange(n, dtype=np.int64)
    if max_points is not None and n > max_points:
        if labels is None:
            raise RepresentationEvalError("labels are required to subsample")
        keep = _stratified_subsample(
            np.asarray(labels).reshape(-1), max_points, random_state=random_state
        )
        array = array[keep]
        n = int(array.shape[0])
    require_sklearn()
    if method == METHOD_PCA:
        from sklearn.decomposition import PCA

        n_components = 2 if n >= 2 else 1
        xy = PCA(n_components=n_components, random_state=random_state).fit_transform(
            array
        )
        if n_components == 1:
            xy = np.concatenate((xy, np.zeros((n, 1), dtype=xy.dtype)), axis=1)
        return np.asarray(xy, dtype=np.float64), keep
    if method == METHOD_TSNE:
        from sklearn.manifold import TSNE

        perplexity = float(tsne_perplexity)
        max_perp = float(max(n - 1, 1))
        if perplexity >= max_perp:
            perplexity = max(max_perp - 1.0e-3, 1.0)
        kwargs = {
            "n_components": 2,
            "perplexity": perplexity,
            "init": "pca",
            "random_state": random_state,
            "learning_rate": "auto",
        }
        try:
            reducer = TSNE(max_iter=tsne_max_iter, **kwargs)
        except TypeError:
            reducer = TSNE(n_iter=tsne_max_iter, **kwargs)
        xy = reducer.fit_transform(array)
        return np.asarray(xy, dtype=np.float64), keep
    raise RepresentationEvalError(f"unknown projection method {method!r}")


def _stratified_subsample(
    labels: np.ndarray, max_points: int, *, random_state: int
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    n = int(labels.shape[0])
    if max_points >= n:
        return np.arange(n, dtype=np.int64)
    classes = np.unique(labels)
    n_classes = int(classes.shape[0])
    chosen: list[int] = []
    remaining = max_points
    for i in range(n_classes):
        cls = int(classes[i])
        members = np.flatnonzero(labels == cls)
        leftover_classes = n_classes - i
        take = remaining // leftover_classes
        if take < 1:
            take = 1
        if take > members.shape[0]:
            take = int(members.shape[0])
        picked = rng.choice(members, size=take, replace=False)
        n_picked = int(picked.shape[0])
        for j in range(n_picked):
            chosen.append(int(picked[j]))
        remaining = max_points - len(chosen)
        if remaining <= 0:
            break
    return np.asarray(chosen, dtype=np.int64)
