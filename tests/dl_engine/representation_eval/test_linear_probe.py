from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.representation_eval.linear_probe.probe import linear_probe
from tests.dl_engine.representation_eval.support import two_gaussians

pytest.importorskip("sklearn")


def test_linear_probe_separates_two_gaussians() -> None:
    train, val = two_gaussians()
    result = linear_probe(train, val)
    assert float(result.scalars["accuracy"]) == 1.0
    assert float(result.scalars["balanced_accuracy"]) == 1.0
    assert float(result.scalars["auroc"]) == 1.0
    assert result.pred.shape == (val.n_samples,)
    assert result.proba.shape == (val.n_samples, 2)
    assert int(result.confusion.sum()) == val.n_samples
    assert "recall_large" in result.scalars
    assert "precision_non_large" in result.scalars
