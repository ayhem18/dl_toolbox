"""Prediction metrics from integer labels. No sklearn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredMetrics:
    accuracy: float
    balanced_accuracy: float
    chance_accuracy: float
    recall: dict[str, float]
    precision: dict[str, float]


def classification_pred_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: tuple[str, ...],
) -> PredMetrics:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape[0] == 0:
        raise ValueError("y_true is empty")
    if true.shape[0] != pred.shape[0]:
        raise ValueError(
            f"y_true length {true.shape[0]} != y_pred length {pred.shape[0]}"
        )
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError(f"need at least 2 class_names, got {n_classes}")
    accuracy = float(np.mean(true == pred))
    counts = np.bincount(true, minlength=n_classes)
    majority = int(np.max(counts))
    chance = float(majority) / float(true.shape[0])
    recall: dict[str, float] = {}
    precision: dict[str, float] = {}
    recalls: list[float] = [0.0] * n_classes
    for i in range(n_classes):
        name = class_names[i]
        support = int(np.sum(true == i))
        predicted = int(np.sum(pred == i))
        correct = int(np.sum((true == i) & (pred == i)))
        if support == 0:
            rec = 0.0
        else:
            rec = float(correct) / float(support)
        if predicted == 0:
            prec = 0.0
        else:
            prec = float(correct) / float(predicted)
        recall[name] = rec
        precision[name] = prec
        recalls[i] = rec
    balanced = float(np.mean(np.asarray(recalls, dtype=np.float64)))
    return PredMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        chance_accuracy=chance,
        recall=recall,
        precision=precision,
    )


def pred_metrics_as_scalars(
    metrics: PredMetrics,
    *,
    class_names: tuple[str, ...],
    include_precision: bool,
) -> dict[str, float]:
    values: dict[str, float] = {
        "accuracy": float(metrics.accuracy),
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "chance_accuracy": float(metrics.chance_accuracy),
    }
    n = len(class_names)
    for i in range(n):
        name = class_names[i]
        values[f"recall_{name}"] = float(metrics.recall[name])
        if include_precision:
            values[f"precision_{name}"] = float(metrics.precision[name])
    return values
