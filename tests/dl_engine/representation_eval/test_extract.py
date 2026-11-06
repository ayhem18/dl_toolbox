# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import torch

from dl_toolbox.dl_engine.representation_eval.common.extract import extract_features
from dl_toolbox.dl_engine.representation_eval.common.loader import loader_for_extract
from tests.dl_engine.representation_eval.support import (
    CLASS_NAMES,
    TinyPool,
    encode_pool,
    labels_from_batch,
    make_image_loader,
)


def test_loader_for_extract_keeps_every_item() -> None:
    source = make_image_loader(n=8, batch_size=4)
    rebuilt = loader_for_extract(source)
    assert rebuilt.drop_last is False
    n_seen = 0
    for batch in rebuilt:
        n_seen += int(batch["image"].shape[0])
    assert n_seen == 8


def test_extract_features_matches_dataset_size() -> None:
    loader = make_image_loader(n=8, batch_size=4)
    model = TinyPool()
    bank = extract_features(
        model,
        loader,
        encode_pool,
        labels_from_batch,
        device=torch.device("cpu"),
        class_names=CLASS_NAMES,
    )
    assert bank.n_samples == 8
    assert bank.n_features == 4
    assert bank.labels.shape == (8,)
    assert len(bank.ids) == 8
    norms = (bank.features ** 2).sum(axis=1) ** 0.5
    assert float(norms.min()) > 0.99
    assert float(norms.max()) < 1.01
