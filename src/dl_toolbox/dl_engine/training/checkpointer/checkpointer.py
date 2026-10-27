"""Run façade: engine and fit call this. Talks only to CheckpointManager."""

from __future__ import annotations

from pathlib import Path

import lightning as L

from dl_toolbox.dl_engine.training.checkpointer.manager import CheckpointerBinding
from dl_toolbox.dl_engine.training.checkpointer.types import (
    CheckpointRecord,
    CheckpointRequest,
    CheckpointSaveError,
    FitRunState,
)


class Checkpointer:
    def __init__(self, *, binding: CheckpointerBinding) -> None:
        self._binding = binding

    def as_callback(self) -> L.Callback:
        return _OwnedCheckpoint(self)

    def save(self, trainer: L.Trainer) -> None:
        epoch = int(trainer.current_epoch)
        score = _monitor_score(trainer, self._binding.monitor_key)
        self._binding.save_new(
            epoch=epoch,
            score=score,
            write=trainer.save_checkpoint,
        )

    def fetch(self, request: CheckpointRequest, *, dest: Path) -> CheckpointRecord:
        return self._binding.fetch(request, dest=dest)

    def fit_state(self) -> FitRunState:
        return self._binding.fit_state()

    def mark_finished(self) -> None:
        self._binding.mark_finished()


class _OwnedCheckpoint(L.Callback):
    def __init__(self, checkpointer: Checkpointer) -> None:
        super().__init__()
        self._checkpointer = checkpointer

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        self._checkpointer.save(trainer)


def _monitor_score(trainer: L.Trainer, key: str) -> float:
    metrics = getattr(trainer, "callback_metrics", None) or {}
    if key not in metrics:
        raise CheckpointSaveError(f"monitor metric {key!r} is missing from callback_metrics")
    raw = metrics[key]
    if hasattr(raw, "item"):
        return float(raw.item())
    return float(raw)
