"""Persist FeatureBanks for later re-plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.errors import (
    RepresentationEvalError,
)
from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from dl_toolbox.dl_engine.representation_eval.common.json_ids import jsonable_id

_NPZ_KEYS = (
    "train_features",
    "train_labels",
    "val_features",
    "val_labels",
    "class_names",
    "train_ids",
    "val_ids",
)


def write_embeddings_npz(
    path: Path, *, train: FeatureBank, val: FeatureBank
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_ids = [jsonable_id(train.ids[i]) for i in range(train.n_samples)]
    val_ids = [jsonable_id(val.ids[i]) for i in range(val.n_samples)]
    np.savez(
        path,
        train_features=train.features,
        train_labels=train.labels,
        val_features=val.features,
        val_labels=val.labels,
        class_names=np.asarray(train.class_names, dtype=object),
        train_ids=np.asarray(train_ids, dtype=object),
        val_ids=np.asarray(val_ids, dtype=object),
    )
    return path


def read_embeddings_npz(path: Path) -> tuple[FeatureBank, FeatureBank]:
    if not path.is_file():
        raise FileNotFoundError(f"embeddings not found: {path}")
    with np.load(path, allow_pickle=True) as payload:
        n_keys = len(_NPZ_KEYS)
        for i in range(n_keys):
            key = _NPZ_KEYS[i]
            if key not in payload:
                raise RepresentationEvalError(
                    f"{path} is missing embeddings key {key!r}"
                )
        class_names = _string_tuple(payload["class_names"])
        train = FeatureBank(
            features=payload["train_features"],
            labels=payload["train_labels"],
            class_names=class_names,
            ids=_ids_from_saved(payload["train_ids"]),
        )
        val = FeatureBank(
            features=payload["val_features"],
            labels=payload["val_labels"],
            class_names=class_names,
            ids=_ids_from_saved(payload["val_ids"]),
        )
    if train.class_names != val.class_names:
        raise RepresentationEvalError("train and val class_names must match")
    return train, val


def _string_tuple(raw: np.ndarray) -> tuple[str, ...]:
    values = np.asarray(raw, dtype=object).reshape(-1)
    n = int(values.shape[0])
    names: list[str] = [""] * n
    for i in range(n):
        names[i] = str(values[i])
    return tuple(names)


def _ids_from_saved(raw: np.ndarray) -> tuple[object, ...]:
    array = np.asarray(raw, dtype=object)
    if array.ndim == 0:
        raise RepresentationEvalError("ids array is scalar")
    n = int(array.shape[0])
    ids: list[object] = [None] * n
    for i in range(n):
        ids[i] = _restore_id(array[i])
    return tuple(ids)


def _restore_id(sample_id: object) -> object:
    if isinstance(sample_id, np.ndarray):
        n = int(sample_id.shape[0])
        parts: list[object] = [None] * n
        for i in range(n):
            parts[i] = _restore_id(sample_id[i])
        return tuple(parts)
    if isinstance(sample_id, (list, tuple)):
        n = len(sample_id)
        parts: list[object] = [None] * n
        for i in range(n):
            parts[i] = _restore_id(sample_id[i])
        return tuple(parts)
    if isinstance(sample_id, np.generic):
        return sample_id.item()
    return sample_id
