"""Write inspect artifacts. Returns scalar maps; the caller logs them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.dl_engine.representation_eval.common.config import (
    RepresentationProbeConfig,
)
from dl_toolbox.dl_engine.representation_eval.common.embeddings import (
    write_embeddings_npz,
)
from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from dl_toolbox.dl_engine.representation_eval.common.json_ids import jsonable_id
from dl_toolbox.dl_engine.representation_eval.knn.classify import KnnResult, knn_probe
from dl_toolbox.dl_engine.representation_eval.knn.crowding import rank_crowding_extremes
from dl_toolbox.dl_engine.representation_eval.knn.sheets import write_crowding_sheets
from dl_toolbox.dl_engine.representation_eval.linear_probe.probe import (
    LinearProbeResult,
    linear_probe,
)
from dl_toolbox.dl_engine.representation_eval.transformations.project import (
    METHOD_PCA,
    METHOD_TSNE,
    concat_train_val,
    project_2d,
)
from dl_toolbox.dl_engine_utils.images.scatter import plot_labeled_scatter

LINEAR_DIRNAME = "linear_regression"
KNN_DIRNAME = "knn"
PCA_DIRNAME = "pca"
TSNE_DIRNAME = "tsne"
PROBE_DIRNAME = "probe"
LoadIndexImage = Callable[[int], object]


@dataclass(frozen=True)
class RepresentationProbeResult:
    knn: KnnResult
    linear: LinearProbeResult
    knn_scalars: dict[str, float]
    linear_scalars: dict[str, float]
    knn_table: tuple[dict[str, Any], ...]
    linear_table: tuple[dict[str, Any], ...]
    paths: dict[str, Path]


def run_representation_probes(
    train: FeatureBank,
    val: FeatureBank,
    *,
    staging: Path,
    config: RepresentationProbeConfig | None = None,
    load_train_image: LoadIndexImage | None = None,
    load_val_image: LoadIndexImage | None = None,
) -> RepresentationProbeResult:
    probe_config = RepresentationProbeConfig() if config is None else config
    knn = knn_probe(train, val, k=probe_config.k)
    linear = linear_probe(
        train, val, random_state=probe_config.random_state
    )
    probe_root = staging / PROBE_DIRNAME
    linear_dir = probe_root / LINEAR_DIRNAME
    knn_dir = probe_root / KNN_DIRNAME
    linear_dir.mkdir(parents=True, exist_ok=True)
    knn_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if probe_config.write_embeddings:
        paths["embeddings"] = write_embeddings_npz(
            probe_root / "embeddings.npz", train=train, val=val
        )
    linear_rows = _linear_prediction_rows(val, linear)
    confusion_path = write_json(
        linear_dir / "confusion.json",
        {
            "class_names": list(train.class_names),
            "matrix": linear.confusion.tolist(),
        },
    )
    predictions_path = write_json(
        linear_dir / "predictions.json", {"rows": linear_rows}
    )
    paths["linear_confusion"] = confusion_path
    paths["linear_predictions"] = predictions_path
    crowding_rows = _crowding_rows(val, train, knn)
    crowding_path = write_json(
        knn_dir / "crowding.json",
        {"k": knn.k, "rows": crowding_rows},
    )
    paths["knn_crowding"] = crowding_path
    extremes = rank_crowding_extremes(
        knn.mean_k_distance, n_extremes=probe_config.crowding_extremes
    )
    if load_train_image is not None and load_val_image is not None:
        sheet_paths = write_crowding_sheets(
            knn_dir,
            knn=knn,
            extremes=extremes,
            train=train,
            val=val,
            load_train_image=load_train_image,
            load_val_image=load_val_image,
        )
        sheet_keys = list(sheet_paths)
        n_sheets = len(sheet_keys)
        for i in range(n_sheets):
            key = sheet_keys[i]
            paths[key] = sheet_paths[key]
    features, labels, splits = concat_train_val(train, val)
    pca_xy, _pca_keep = project_2d(
        features,
        method=METHOD_PCA,
        random_state=probe_config.random_state,
    )
    pca_path = plot_labeled_scatter(
        probe_root / PCA_DIRNAME / "scatter.png",
        pca_xy,
        labels,
        class_names=train.class_names,
        splits=splits,
        title="PCA",
    )
    if pca_path is not None:
        paths["pca_scatter"] = pca_path
    tsne_xy, tsne_keep = project_2d(
        features,
        method=METHOD_TSNE,
        random_state=probe_config.random_state,
        tsne_perplexity=probe_config.tsne_perplexity,
        tsne_max_iter=probe_config.tsne_max_iter,
        max_points=probe_config.tsne_max_points,
        labels=labels,
    )
    tsne_path = plot_labeled_scatter(
        probe_root / TSNE_DIRNAME / "scatter.png",
        tsne_xy,
        labels[tsne_keep],
        class_names=train.class_names,
        splits=splits[tsne_keep],
        title="t-SNE",
    )
    if tsne_path is not None:
        paths["tsne_scatter"] = tsne_path
    return RepresentationProbeResult(
        knn=knn,
        linear=linear,
        knn_scalars=dict(knn.scalars),
        linear_scalars=dict(linear.scalars),
        knn_table=tuple(crowding_rows),
        linear_table=tuple(linear_rows),
        paths=paths,
    )


def _linear_prediction_rows(
    val: FeatureBank, linear: LinearProbeResult
) -> list[dict[str, Any]]:
    n = val.n_samples
    n_classes = len(val.class_names)
    rows: list[dict[str, Any]] = [{}] * n
    for i in range(n):
        proba: dict[str, float] = {}
        for c in range(n_classes):
            proba[val.class_names[c]] = float(linear.proba[i, c])
        rows[i] = {
            "id": jsonable_id(val.ids[i]),
            "y": val.class_names[int(val.labels[i])],
            "pred": val.class_names[int(linear.pred[i])],
            "proba": proba,
        }
    return rows


def _crowding_rows(
    val: FeatureBank, train: FeatureBank, knn: KnnResult
) -> list[dict[str, Any]]:
    n = val.n_samples
    k = knn.k
    rows: list[dict[str, Any]] = [{}] * n
    for i in range(n):
        neighbor_ids: list[object] = [None] * k
        neighbor_distances: list[float] = [0.0] * k
        for j in range(k):
            gallery_i = int(knn.neighbor_index[i, j])
            neighbor_ids[j] = jsonable_id(train.ids[gallery_i])
            neighbor_distances[j] = float(knn.neighbor_distance[i, j])
        rows[i] = {
            "id": jsonable_id(val.ids[i]),
            "mean_k_distance": float(knn.mean_k_distance[i]),
            "y": val.class_names[int(val.labels[i])],
            "pred": val.class_names[int(knn.pred[i])],
            "neighbor_ids": neighbor_ids,
            "neighbor_distances": neighbor_distances,
        }
    return rows
