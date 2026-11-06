"""Frozen-feature logistic regression. Folder name on disk is linear_regression."""

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
from dl_toolbox.base.packaging import require


@dataclass(frozen=True)
class LinearProbeResult:
    pred: np.ndarray
    proba: np.ndarray
    confusion: np.ndarray
    scalars: dict[str, float]


def linear_probe(
    train: FeatureBank,
    val: FeatureBank,
    *,
    max_iter: int = 1000,
    random_state: int = 0,
) -> LinearProbeResult:
    if train.class_names != val.class_names:
        raise RepresentationEvalError("train and val class_names must match")
    train.require_all_classes(split="train")
    val.require_all_classes(split="val")
    require(package="sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    x_train = l2_normalize(train.features)
    x_val = l2_normalize(val.features)
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(x_train, train.labels)
    pred = np.asarray(clf.predict(x_val), dtype=np.int64)
    proba = np.asarray(clf.predict_proba(x_val), dtype=np.float64)
    metrics = classification_pred_metrics(
        val.labels, pred, class_names=train.class_names
    )
    scalars = pred_metrics_as_scalars(
        metrics, class_names=train.class_names, include_precision=True
    )
    n_classes = len(train.class_names)
    confusion = _confusion(val.labels, pred, n_classes=n_classes)
    scalars["log_loss"] = float(
        log_loss(val.labels, proba, labels=list(range(n_classes)))
    )
    if n_classes == 2:
        scalars["auroc"] = float(roc_auc_score(val.labels, proba[:, 1]))
    else:
        scalars["auroc"] = float(
            roc_auc_score(val.labels, proba, multi_class="ovr", labels=list(range(n_classes)))
        )
    return LinearProbeResult(
        pred=pred,
        proba=proba,
        confusion=confusion,
        scalars=scalars,
    )


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    n = int(y_true.shape[0])
    for i in range(n):
        matrix[int(y_true[i]), int(y_pred[i])] += 1
    return matrix
