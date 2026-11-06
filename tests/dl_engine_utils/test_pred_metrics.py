# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import numpy as np

from dl_toolbox.dl_engine_utils.classification.pred_metrics import (
    classification_pred_metrics,
    pred_metrics_as_scalars,
)


def test_perfect_binary_predictions() -> None:
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.asarray([0, 0, 1, 1], dtype=np.int64)
    names = ("large", "non_large")
    metrics = classification_pred_metrics(y_true, y_pred, class_names=names)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.chance_accuracy == 0.5
    scalars = pred_metrics_as_scalars(
        metrics, class_names=names, include_precision=True
    )
    assert scalars["recall_large"] == 1.0
    assert scalars["precision_non_large"] == 1.0


def test_chance_is_majority_class_rate() -> None:
    y_true = np.asarray([0, 0, 0, 1], dtype=np.int64)
    y_pred = np.asarray([0, 0, 0, 0], dtype=np.int64)
    names = ("a", "b")
    metrics = classification_pred_metrics(y_true, y_pred, class_names=names)
    assert metrics.chance_accuracy == 0.75
    assert metrics.accuracy == 0.75
    assert metrics.recall["a"] == 1.0
    assert metrics.recall["b"] == 0.0
