# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from pathlib import Path

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


class _FakeStrategy:
    def __init__(self) -> None:
        self.barrier_calls = 0

    def barrier(self) -> None:
        self.barrier_calls += 1


class _FakeTrainer:
    def __init__(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        is_global_zero: bool,
    ) -> None:
        self.current_epoch = epoch
        self.callback_metrics = metrics
        self.sanity_checking = False
        self.is_global_zero = is_global_zero
        self.strategy = _FakeStrategy()
        self.save_checkpoint_paths: list[Path] = []

    def save_checkpoint(self, path: Path) -> None:
        self.save_checkpoint_paths.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(f"epoch-{self.current_epoch}".encode("utf-8"))


def _checkpointer(run_artifacts_dir: Path) -> InspectableCheckpointer:
    return InspectableCheckpointer(
        run_artifacts_dir=run_artifacts_dir,
        policy=CheckpointPolicy(save_top_k=-1, save_last=True),
        monitor=CheckpointMonitor(key="loss", mode="min"),
        max_epochs=10,
        lock_timeout_seconds=5.0,
    )


def test_callback_saves_on_global_zero(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir)
    trainer = _FakeTrainer(
        epoch=1, metrics={"loss": 0.2}, is_global_zero=True
    )
    InspectableCheckpointCallback(checkpointer=checkpointer).on_validation_end(
        trainer, None
    )
    assert (run_artifacts_dir / "checkpoints" / "catalog.json").is_file()
    record = checkpointer.fetch(
        CheckpointRequest.parse("rank=1"), dest=tmp_path / "snap"
    )
    assert record.epoch == 1
    assert trainer.strategy.barrier_calls == 0
    assert len(trainer.save_checkpoint_paths) == 1


def test_callback_skips_catalog_write_on_non_zero_rank(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir)
    trainer = _FakeTrainer(
        epoch=1, metrics={"loss": 0.2}, is_global_zero=False
    )
    InspectableCheckpointCallback(checkpointer=checkpointer).on_validation_end(
        trainer, None
    )
    assert not (run_artifacts_dir / "checkpoints" / "catalog.json").exists()
    assert len(trainer.save_checkpoint_paths) == 1
    assert trainer.save_checkpoint_paths[0].name == ".ddp_join.ckpt"


def test_callback_skip_uses_barrier_on_global_zero(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = InspectableCheckpointer(
        run_artifacts_dir=run_artifacts_dir,
        policy=CheckpointPolicy(save_top_k=1, save_last=False),
        monitor=CheckpointMonitor(key="loss", mode="min"),
        max_epochs=10,
        lock_timeout_seconds=5.0,
    )
    callback = InspectableCheckpointCallback(checkpointer=checkpointer)
    first = _FakeTrainer(epoch=0, metrics={"loss": 0.1}, is_global_zero=True)
    callback.on_validation_end(first, None)
    second = _FakeTrainer(epoch=1, metrics={"loss": 0.9}, is_global_zero=True)
    callback.on_validation_end(second, None)
    assert second.strategy.barrier_calls == 1
    assert second.save_checkpoint_paths == []
    assert checkpointer.fit_state().latest_epoch == 1
    assert checkpointer.fit_state().current_best_epoch == 0
