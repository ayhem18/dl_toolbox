from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dl_toolbox.dl_engine.representation_eval.common.config import (
    RepresentationProbeConfig,
)
from dl_toolbox.dl_engine.representation_eval.common.run import (
    run_representation_probes,
)
from tests.dl_engine.representation_eval.support import two_gaussians

pytest.importorskip("sklearn")


def test_run_writes_structured_artifacts_and_sheets(tmp_path: Path) -> None:
    train, val = two_gaussians(n_train=16, n_val=8, dim=4)
    result = run_representation_probes(
        train,
        val,
        staging=tmp_path,
        config=RepresentationProbeConfig(
            k=5,
            crowding_extremes=3,
            tsne_max_points=24,
            tsne_perplexity=5.0,
            tsne_max_iter=250,
            random_state=0,
        ),
        load_train_image=_gray,
        load_val_image=_gray,
    )
    assert result.knn_scalars["accuracy"] == 1.0
    assert result.linear_scalars["accuracy"] == 1.0
    assert (tmp_path / "probe" / "linear_regression" / "confusion.json").is_file()
    assert (tmp_path / "probe" / "linear_regression" / "predictions.json").is_file()
    assert (tmp_path / "probe" / "knn" / "crowding.json").is_file()
    assert (tmp_path / "probe" / "embeddings.npz").is_file()
    assert not (tmp_path / "probe" / "knn" / "accuracy.json").exists()
    for kind in ("top", "bottom"):
        for rank in (1, 2, 3):
            stem = f"{kind}_{rank}_crowded_sample"
            assert stem in result.paths
            assert result.paths[stem].is_file()
    assert len(result.knn_table) == 8
    assert len(result.linear_table) == 8


def _gray(index: int) -> np.ndarray:
    plane = np.full((12, 16), float(index % 5) / 4.0, dtype=np.float32)
    return plane
