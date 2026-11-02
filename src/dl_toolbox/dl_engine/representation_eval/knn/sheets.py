"""Write top/bottom crowded-sample neighbor sheets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from dl_toolbox.dl_engine.representation_eval.knn.classify import KnnResult
from dl_toolbox.dl_engine.representation_eval.knn.crowding import (
    CrowdingExtremes,
    crowding_filename,
)
from dl_toolbox.dl_engine_utils.images.neighbor_sheet import neighbor_sheet

LoadIndexImage = Callable[[int], object]


def write_crowding_sheets(
    dest_dir: Path,
    *,
    knn: KnnResult,
    extremes: CrowdingExtremes,
    train: FeatureBank,
    val: FeatureBank,
    load_train_image: LoadIndexImage,
    load_val_image: LoadIndexImage,
) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    _write_group(
        paths,
        dest_dir,
        kind="top",
        indices=extremes.top,
        knn=knn,
        train=train,
        val=val,
        load_train_image=load_train_image,
        load_val_image=load_val_image,
    )
    _write_group(
        paths,
        dest_dir,
        kind="bottom",
        indices=extremes.bottom,
        knn=knn,
        train=train,
        val=val,
        load_train_image=load_train_image,
        load_val_image=load_val_image,
    )
    return paths


def _write_group(
    paths: dict[str, Path],
    dest_dir: Path,
    *,
    kind: str,
    indices: tuple[int, ...],
    knn: KnnResult,
    train: FeatureBank,
    val: FeatureBank,
    load_train_image: LoadIndexImage,
    load_val_image: LoadIndexImage,
) -> None:
    n = len(indices)
    for rank in range(n):
        query_i = indices[rank]
        filename = crowding_filename(kind, rank + 1)
        stem = filename[: -len(".png")]
        path = dest_dir / filename
        neighbor_idx = knn.neighbor_index[query_i]
        neighbor_dist = knn.neighbor_distance[query_i]
        k = int(neighbor_idx.shape[0])
        neighbor_images: list[object] = [None] * k
        neighbor_titles: list[str] = [""] * k
        for j in range(k):
            gallery_i = int(neighbor_idx[j])
            neighbor_images[j] = load_train_image(gallery_i)
            neighbor_name = train.class_names[int(train.labels[gallery_i])]
            neighbor_titles[j] = (
                f"n{j + 1} d={float(neighbor_dist[j]):.3f} {neighbor_name}"
            )
        true_name = val.class_names[int(val.labels[query_i])]
        pred_name = val.class_names[int(knn.pred[query_i])]
        mean_d = float(knn.mean_k_distance[query_i])
        written = neighbor_sheet(
            path,
            load_val_image(query_i),
            neighbor_images,
            query_title=f"q y={true_name} pred={pred_name} mean_d={mean_d:.3f}",
            neighbor_titles=neighbor_titles,
        )
        if written is not None:
            paths[stem] = written
