from __future__ import annotations

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from dl_toolbox.dl_engine.representation_eval.knn.classify import knn_probe
from dl_toolbox.dl_engine.representation_eval.knn.crowding import rank_crowding_extremes
from tests.dl_engine.representation_eval.support import CLASS_NAMES, two_gaussians


def test_knn_separates_two_gaussians() -> None:
    train, val = two_gaussians()
    result = knn_probe(train, val, k=5)
    assert result.k == 5
    assert float(result.scalars["accuracy"]) == 1.0
    assert float(result.scalars["balanced_accuracy"]) == 1.0
    assert result.pred.shape == (val.n_samples,)
    assert result.neighbor_index.shape == (val.n_samples, 5)


def test_crowding_rank_puts_near_cluster_on_top() -> None:
    train_x = np.zeros((10, 2), dtype=np.float32)
    train_x[:, 1] = 0.1
    train_x[9] = np.array([20.0, 0.1], dtype=np.float32)
    train_y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    val_x = np.asarray([[0.0, 0.1], [40.0, 0.1]], dtype=np.float32)
    val_y = np.asarray([0, 1], dtype=np.int64)
    train = FeatureBank(
        features=train_x,
        labels=train_y,
        class_names=CLASS_NAMES,
        ids=tuple(range(10)),
    )
    val = FeatureBank(
        features=val_x,
        labels=val_y,
        class_names=CLASS_NAMES,
        ids=(0, 1),
    )
    result = knn_probe(train, val, k=3)
    extremes = rank_crowding_extremes(result.mean_k_distance, n_extremes=1)
    assert extremes.top == (0,)
    assert extremes.bottom == (1,)
    assert result.mean_k_distance[0] < result.mean_k_distance[1]


def test_crowding_allows_overlap_when_n_lt_six() -> None:
    distances = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    extremes = rank_crowding_extremes(distances, n_extremes=3)
    assert extremes.top == (0, 1, 2)
    assert extremes.bottom == (4, 3, 2)
