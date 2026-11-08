"""Shared DDP fit harness for inspectable checkpoint tests. Not collected by pytest."""

from __future__ import annotations

import random
import sys
from collections.abc import Sequence
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.strategies import DDPStrategy
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.callback import (
    InspectableCheckpointCallback,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CheckpointMonitor,
    CheckpointPolicy,
    CheckpointRequest,
)

MONITOR_KEY = "scripted"
N_DDP_TRIALS = 8
LOCK_TIMEOUT_SECONDS = 120.0
N_SAMPLES = 16
BATCH_SIZE = 4


def cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def ddp_runtime() -> tuple[str, int, str]:
    n_cuda = cuda_device_count()
    if n_cuda < 2:
        raise RuntimeError(
            f"DDP checkpoint tests require at least 2 CUDA devices, got {n_cuda}"
        )
    backend = "gloo" if sys.platform == "win32" else "nccl"
    return "gpu", 2, backend


class ScriptedMonitorModule(L.LightningModule):
    """Logs ``scores[current_epoch]`` as the monitor. One Linear net to dump."""

    def __init__(self, *, scores: tuple[float, ...], monitor_key: str) -> None:
        super().__init__()
        n = len(scores)
        if n < 1:
            raise ValueError(f"scores must be non-empty, got length {n}")
        self.scores = scores
        self.monitor_key = monitor_key
        self.net = nn.Linear(4, 1)

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        del batch_idx
        features, target = batch
        return F.mse_loss(self.net(features), target)

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        del batch_idx
        features, target = batch
        loss = F.mse_loss(self.net(features), target)
        epoch = int(self.current_epoch)
        score = self.scores[epoch]
        self.log(
            self.monitor_key,
            score,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=int(features.size(0)),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.05)


class DdpFitConfig:
    def __init__(
        self,
        *,
        run_artifacts_dir: Path,
        scores: tuple[float, ...],
        save_top_k: int,
        save_last: bool,
        mode: str,
        lock_timeout_seconds: float,
    ) -> None:
        self.run_artifacts_dir = run_artifacts_dir
        self.scores = scores
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.mode = mode
        self.lock_timeout_seconds = lock_timeout_seconds


def make_loaders() -> tuple[DataLoader, DataLoader]:
    features = torch.zeros(N_SAMPLES, 4)
    target = torch.zeros(N_SAMPLES, 1)
    dataset = TensorDataset(features, target)
    train = DataLoader(dataset, batch_size=BATCH_SIZE)
    val = DataLoader(dataset, batch_size=BATCH_SIZE)
    return train, val


def make_checkpointer(config: DdpFitConfig) -> InspectableCheckpointer:
    return InspectableCheckpointer(
        run_artifacts_dir=config.run_artifacts_dir,
        policy=CheckpointPolicy(
            save_top_k=config.save_top_k, save_last=config.save_last
        ),
        monitor=CheckpointMonitor(key=MONITOR_KEY, mode=config.mode),
        max_epochs=len(config.scores),
        lock_timeout_seconds=config.lock_timeout_seconds,
        lock_poll_seconds=0.05,
    )


def run_ddp_fit(config: DdpFitConfig) -> InspectableCheckpointer:
    accelerator, n_devices, backend = ddp_runtime()
    checkpointer = make_checkpointer(config)
    module = ScriptedMonitorModule(
        scores=config.scores, monitor_key=MONITOR_KEY
    )
    train_loader, val_loader = make_loaders()
    trainer = L.Trainer(
        max_epochs=len(config.scores),
        accelerator=accelerator,
        devices=n_devices,
        strategy=DDPStrategy(
            start_method="spawn", process_group_backend=backend
        ),
        precision=32,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[InspectableCheckpointCallback(checkpointer=checkpointer)],
    )
    trainer.fit(
        module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    return checkpointer


def live_ckpt_paths(run_artifacts_dir: Path) -> list[Path]:
    live = run_artifacts_dir / "checkpoints" / "live"
    if not live.is_dir():
        return []
    paths = list(live.glob("*.ckpt"))
    kept: list[Path] = []
    n = len(paths)
    for i in range(n):
        if paths[i].name != ".ddp_join.ckpt":
            kept.append(paths[i])
    return kept


def lock_dir(run_artifacts_dir: Path) -> Path:
    return run_artifacts_dir / "checkpoints" / "catalog.lock"


def catalog_path(run_artifacts_dir: Path) -> Path:
    return run_artifacts_dir / "checkpoints" / "catalog.json"


def best_epoch(*, scores: Sequence[float], mode: str) -> int:
    """Target: the first epoch with the unique-best score (ties keep the earliest)."""
    best_i = 0
    best = scores[0]
    n = len(scores)
    for i in range(1, n):
        if mode == "min" and scores[i] < best:
            best = scores[i]
            best_i = i
        elif mode == "max" and scores[i] > best:
            best = scores[i]
            best_i = i
    return best_i


def unique_scores(*, rng: random.Random, n_epochs: int) -> tuple[float, ...]:
    values: list[float] = []
    seen: set[float] = set()
    while len(values) < n_epochs:
        raw = rng.random()
        key = round(raw, 8)
        if key in seen:
            continue
        seen.add(key)
        values.append(float(raw))
    return tuple(values)


def fetch_rank1(checkpointer: InspectableCheckpointer, *, dest: Path):
    return checkpointer.fetch(CheckpointRequest.parse("rank=1"), dest=dest)


def fetch_epoch(
    checkpointer: InspectableCheckpointer, *, epoch: int, dest: Path
):
    return checkpointer.fetch(
        CheckpointRequest.parse(f"epoch={epoch}"), dest=dest
    )


def fetch_last(checkpointer: InspectableCheckpointer, *, dest: Path):
    return checkpointer.fetch(CheckpointRequest.parse("last"), dest=dest)
