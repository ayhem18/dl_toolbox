from __future__ import annotations

from pathlib import Path

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.embeddings import (
    read_embeddings_npz,
    write_embeddings_npz,
)
from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from tests.dl_engine.representation_eval.support import CLASS_NAMES, two_gaussians


def test_embeddings_roundtrip_restores_tuple_ids(tmp_path: Path) -> None:
    train, val = two_gaussians(n_train=6, n_val=4, dim=3)
    train = FeatureBank(
        features=train.features,
        labels=train.labels,
        class_names=CLASS_NAMES,
        ids=_frame_ids("train", train.n_samples),
    )
    val = FeatureBank(
        features=val.features,
        labels=val.labels,
        class_names=CLASS_NAMES,
        ids=_frame_ids("val", val.n_samples),
    )
    path = write_embeddings_npz(tmp_path / "embeddings.npz", train=train, val=val)
    loaded_train, loaded_val = read_embeddings_npz(path)
    np.testing.assert_allclose(loaded_train.features, train.features)
    np.testing.assert_array_equal(loaded_train.labels, train.labels)
    np.testing.assert_allclose(loaded_val.features, val.features)
    np.testing.assert_array_equal(loaded_val.labels, val.labels)
    assert loaded_train.class_names == CLASS_NAMES
    assert loaded_val.class_names == CLASS_NAMES
    assert loaded_train.n_samples == train.n_samples
    assert loaded_val.n_samples == val.n_samples
    assert loaded_train.ids == train.ids
    assert loaded_val.ids == val.ids


def _frame_ids(split: str, n: int) -> tuple[object, ...]:
    ids: list[object] = [None] * n
    for i in range(n):
        ids[i] = (f"{split}.dcm", str(i))
    return tuple(ids)
