# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from pathlib import Path

import pytest

from dl_toolbox.dl_engine.representation_eval.transformations.project import (
    METHOD_PCA,
    METHOD_TSNE,
    concat_train_val,
    project_2d,
)
from dl_toolbox.dl_engine_utils.images.scatter import plot_labeled_scatter
from tests.dl_engine.representation_eval.support import two_gaussians

pytest.importorskip("sklearn")


def test_pca_returns_2d_for_all_points() -> None:
    train, val = two_gaussians(n_train=16, n_val=8)
    features, labels, splits = concat_train_val(train, val)
    xy, keep = project_2d(features, method=METHOD_PCA, random_state=0)
    assert xy.shape == (24, 2)
    assert keep.shape == (24,)
    assert splits.shape == (24,)
    assert labels.shape == (24,)


def test_tsne_subsample_keeps_requested_cap(tmp_path: Path) -> None:
    train, val = two_gaussians(n_train=20, n_val=10, dim=4)
    features, labels, splits = concat_train_val(train, val)
    xy, keep = project_2d(
        features,
        method=METHOD_TSNE,
        random_state=0,
        tsne_perplexity=5.0,
        tsne_max_iter=250,
        max_points=12,
        labels=labels,
    )
    assert xy.shape == (12, 2)
    assert keep.shape == (12,)
    path = plot_labeled_scatter(
        tmp_path / "scatter.png",
        xy,
        labels[keep],
        class_names=train.class_names,
        splits=splits[keep],
        title="t-SNE",
    )
    if path is not None:
        assert path.is_file()
