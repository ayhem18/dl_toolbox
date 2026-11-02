"""Tiny fixtures for representation_eval tests. Not collected by pytest."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dl_toolbox.dl_engine.representation_eval.common.feature_bank import FeatureBank

CLASS_NAMES = ("large", "non_large")


class TinyPool(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).mean(dim=(-2, -1))


class DictImageDataset(Dataset):
    def __init__(self, images: Tensor, labels: Tensor) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        return {
            "image": self._images[index],
            "label": int(self._labels[index].item()),
            "id": index,
        }


def two_gaussians(
    *,
    n_train: int = 40,
    n_val: int = 20,
    dim: int = 8,
    seed: int = 0,
) -> tuple[FeatureBank, FeatureBank]:
    rng = np.random.RandomState(seed)
    train_y = np.asarray([i % 2 for i in range(n_train)], dtype=np.int64)
    val_y = np.asarray([i % 2 for i in range(n_val)], dtype=np.int64)
    train_x = rng.normal(size=(n_train, dim))
    val_x = rng.normal(size=(n_val, dim))
    train_x[train_y == 0, 0] += 6.0
    train_x[train_y == 1, 0] -= 6.0
    val_x[val_y == 0, 0] += 6.0
    val_x[val_y == 1, 0] -= 6.0
    train = FeatureBank(
        features=train_x.astype(np.float32),
        labels=train_y,
        class_names=CLASS_NAMES,
        ids=tuple(range(n_train)),
    )
    val = FeatureBank(
        features=val_x.astype(np.float32),
        labels=val_y,
        class_names=CLASS_NAMES,
        ids=tuple(range(n_val)),
    )
    return train, val


def encode_pool(model: nn.Module, batch: dict, device: torch.device) -> Tensor:
    image = batch["image"].to(device)
    pooled = model(image)
    return torch.nn.functional.normalize(pooled, dim=1)


def labels_from_batch(batch: dict) -> list[int]:
    target = batch["label"]
    if isinstance(target, Tensor):
        n = int(target.shape[0])
        out: list[int] = [0] * n
        for i in range(n):
            out[i] = int(target[i].item())
        return out
    return [int(value) for value in target]


def make_image_loader(*, n: int = 8, batch_size: int = 4) -> DataLoader:
    images = torch.randn(n, 1, 8, 8)
    labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
    return DataLoader(
        DictImageDataset(images, labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
