"""Batched frozen-feature extract. Project supplies encode_fn and label mapping."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_toolbox.dl_engine.representation_eval.common.errors import (
    RepresentationEvalError,
)
from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank
from dl_toolbox.dl_engine.representation_eval.common.loader import loader_for_extract
from dl_toolbox.dl_engine.dl_engine_utils.cuda import is_cuda_oom

EncodeFn = Callable[[nn.Module, Mapping[str, Any], torch.device], Tensor]
LabelsFrom = Callable[[Mapping[str, Any]], Sequence[int]]
IdsFrom = Callable[[Mapping[str, Any]], Sequence[object]]


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    encode_fn: EncodeFn,
    labels_from: LabelsFrom,
    *,
    device: torch.device,
    class_names: tuple[str, ...],
    ids_from: IdsFrom | None = None,
    progress_desc: str = "extract",
) -> FeatureBank:
    batch_size = int(loader.batch_size or 1)
    while True:
        try:
            extract_loader = loader_for_extract(loader, batch_size=batch_size)
            return _extract_once(
                model,
                extract_loader,
                encode_fn,
                labels_from,
                device=device,
                class_names=class_names,
                ids_from=ids_from,
                progress_desc=progress_desc,
            )
        except Exception as exc:
            if not is_cuda_oom(exc) or batch_size <= 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = batch_size // 2
            print(
                f"extract: CUDA OOM; retrying with batch_size={batch_size}",
                flush=True,
            )


def _extract_once(
    model: nn.Module,
    loader: DataLoader,
    encode_fn: EncodeFn,
    labels_from: LabelsFrom,
    *,
    device: torch.device,
    class_names: tuple[str, ...],
    ids_from: IdsFrom | None,
    progress_desc: str,
) -> FeatureBank:
    model.eval()
    model.to(device)
    n_dataset = len(loader.dataset)
    if n_dataset < 1:
        raise RepresentationEvalError("extract loader is empty")
    rows: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[object] = []
    seen = 0
    progress = tqdm(
        total=n_dataset,
        desc=progress_desc,
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
        ascii=True,
    )
    try:
        with torch.inference_mode():
            for batch in loader:
                encoded = encode_fn(model, batch, device)
                if not isinstance(encoded, Tensor):
                    raise TypeError(
                        f"encode_fn must return a Tensor, got {type(encoded)!r}"
                    )
                if encoded.ndim != 2:
                    raise RepresentationEvalError(
                        f"encode_fn must return (B, D), got {tuple(encoded.shape)}"
                    )
                cpu = encoded.detach().to(device="cpu", dtype=torch.float32).numpy()
                batch_labels = list(labels_from(batch))
                if len(batch_labels) != cpu.shape[0]:
                    raise RepresentationEvalError(
                        f"labels_from length {len(batch_labels)} != batch {cpu.shape[0]}"
                    )
                if ids_from is None:
                    batch_ids: list[object] = [seen + i for i in range(cpu.shape[0])]
                else:
                    batch_ids = list(ids_from(batch))
                    if len(batch_ids) != cpu.shape[0]:
                        raise RepresentationEvalError(
                            f"ids_from length {len(batch_ids)} != batch {cpu.shape[0]}"
                        )
                rows.append(cpu)
                n_batch = cpu.shape[0]
                for i in range(n_batch):
                    labels.append(int(batch_labels[i]))
                    ids.append(batch_ids[i])
                seen += n_batch
                progress.update(n_batch)
    finally:
        progress.close()
    if seen != n_dataset:
        raise RepresentationEvalError(
            f"extracted {seen} samples, dataset has {n_dataset}"
        )
    features = np.concatenate(tuple(rows), axis=0)
    return FeatureBank(
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
        ids=tuple(ids),
    )
