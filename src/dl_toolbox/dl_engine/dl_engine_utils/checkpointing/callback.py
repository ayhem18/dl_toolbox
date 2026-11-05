"""Lightning callback for inspectability under DDP.

Rank 0 calls ``InspectableCheckpointer.save`` with
``write_function=trainer.save_checkpoint``. Non-zero ranks still enter
``Trainer.save_checkpoint`` so they hit ``strategy.barrier``. If rank 0
skips the write, it calls ``strategy.barrier`` instead so the collective
still matches.
"""

from __future__ import annotations

import lightning as L

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CheckpointSaveError,
)


class InspectableCheckpointCallback(L.Callback):
    def __init__(self, *, checkpointer: InspectableCheckpointer) -> None:
        super().__init__()
        self._checkpointer = checkpointer

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        score = _monitor_score(trainer, self._checkpointer.monitor_key)
        if trainer.is_global_zero:
            kept = self._checkpointer.save(
                epoch=epoch,
                score=score,
                write_function=trainer.save_checkpoint,
            )
            if not kept:
                trainer.strategy.barrier()
            return
        trainer.save_checkpoint(self._checkpointer.ddp_join_path())


def _monitor_score(trainer: L.Trainer, key: str) -> float:
    metrics = getattr(trainer, "callback_metrics", None) or {}
    if key not in metrics:
        raise CheckpointSaveError(
            f"monitor metric {key!r} is missing from callback_metrics"
        )
    raw = metrics[key]
    if hasattr(raw, "item"):
        return float(raw.item())
    return float(raw)
