"""Frozen embeddings plus integer labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.errors import (
    RepresentationEvalError,
)


@dataclass(frozen=True)
class FeatureBank:
    features: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]
    ids: tuple[object, ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        labels = np.asarray(self.labels).reshape(-1)
        if features.ndim != 2:
            raise RepresentationEvalError(
                f"features must have shape (N, D), got {tuple(features.shape)}"
            )
        n_samples = int(features.shape[0])
        if n_samples == 0:
            raise RepresentationEvalError("FeatureBank is empty")
        if int(features.shape[1]) < 1:
            raise RepresentationEvalError("feature dim must be >= 1")
        if labels.shape[0] != n_samples:
            raise RepresentationEvalError(
                f"labels length {labels.shape[0]} != n_samples {n_samples}"
            )
        if len(self.ids) != n_samples:
            raise RepresentationEvalError(
                f"ids length {len(self.ids)} != n_samples {n_samples}"
            )
        if len(self.class_names) < 2:
            raise RepresentationEvalError("class_names must have at least 2 names")
        object.__setattr__(self, "features", np.asarray(features, dtype=np.float32))
        object.__setattr__(self, "labels", np.asarray(labels, dtype=np.int64))

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def require_all_classes(self, *, split: str) -> None:
        n_classes = len(self.class_names)
        for i in range(n_classes):
            if int(np.sum(self.labels == i)) == 0:
                raise RepresentationEvalError(
                    f"{split} is missing class {self.class_names[i]!r}"
                )


def l2_normalize(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise RepresentationEvalError(
            f"features must have shape (N, D), got {tuple(array.shape)}"
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1.0e-12, None)
