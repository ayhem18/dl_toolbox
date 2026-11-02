"""Persist FeatureBanks for later re-plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from dl_toolbox.dl_engine.representation_eval.common.json_ids import jsonable_id


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
