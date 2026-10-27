from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.dl_engine.training.checkpointer.checkpointer import Checkpointer
from dl_toolbox.dl_engine.training.checkpointer.manager import CheckpointManager
from dl_toolbox.dl_engine.training.checkpointer.types import (
    FIT_FINISHED,
    FIT_RUNNING,
    CheckpointLookupError,
    CheckpointPolicy,
    CheckpointRequest,
)
from dl_toolbox.dl_engine.training.config import MonitorConfig


class _FakeTrainer:
    def __init__(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.current_epoch = epoch
        self.callback_metrics = metrics
        self.sanity_checking = False

    def save_checkpoint(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(f"epoch-{self.current_epoch}".encode("utf-8"))


def _checkpointer(run_dir: Path, *, save_top_k: int, save_last: bool) -> Checkpointer:
    manager = CheckpointManager(lock_timeout_seconds=5.0)
    binding = manager.bind(
        run_dir=run_dir,
        policy=CheckpointPolicy(save_top_k=save_top_k, save_last=save_last),
        monitor=MonitorConfig(key="loss", mode="min"),
        max_epochs=10,
    )
    return Checkpointer(binding=binding)


def test_save_and_fetch_by_epoch_rank_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpointer = _checkpointer(run_dir, save_top_k=-1, save_last=True)
    checkpointer.save(_FakeTrainer(epoch=3, metrics={"loss": 0.2}))
    dest = tmp_path / "snap"
    by_epoch = checkpointer.fetch(CheckpointRequest.parse("epoch=3"), dest=dest / "e")
    assert by_epoch.epoch == 3
    assert by_epoch.path.is_file()
    assert by_epoch.path.read_bytes() == b"epoch-3"
    by_rank = checkpointer.fetch(CheckpointRequest.parse("rank=1"), dest=dest / "r")
    assert by_rank.epoch == 3
    by_last = checkpointer.fetch(CheckpointRequest.parse("last"), dest=dest / "l")
    assert by_last.epoch == 3
    state = checkpointer.fit_state()
    assert state.status == FIT_RUNNING
    assert state.latest_epoch == 3
    assert state.current_best_epoch == 3


def test_prune_keeps_best_rank_now(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpointer = _checkpointer(run_dir, save_top_k=1, save_last=False)
    checkpointer.save(_FakeTrainer(epoch=0, metrics={"loss": 0.5}))
    checkpointer.save(_FakeTrainer(epoch=1, metrics={"loss": 0.1}))
    dest = tmp_path / "snap"
    best = checkpointer.fetch(CheckpointRequest.parse("rank=1"), dest=dest)
    assert best.epoch == 1
    with pytest.raises(CheckpointLookupError, match="epoch=0"):
        checkpointer.fetch(CheckpointRequest.parse("epoch=0"), dest=tmp_path / "gone")
    live = list((run_dir / "checkpoints" / "live").glob("*.ckpt"))
    assert len(live) == 1


def test_save_last_keeps_file_outside_top_k(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpointer = _checkpointer(run_dir, save_top_k=1, save_last=True)
    checkpointer.save(_FakeTrainer(epoch=0, metrics={"loss": 0.1}))
    checkpointer.save(_FakeTrainer(epoch=1, metrics={"loss": 0.9}))
    dest = tmp_path / "snap"
    best = checkpointer.fetch(CheckpointRequest.parse("rank=1"), dest=dest / "best")
    assert best.epoch == 0
    last = checkpointer.fetch(CheckpointRequest.parse("last"), dest=dest / "last")
    assert last.epoch == 1
    with pytest.raises(CheckpointLookupError, match="epoch=1"):
        checkpointer.fetch(CheckpointRequest.parse("epoch=1"), dest=dest / "e1")


def test_fetch_missing_catalog_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpointer = _checkpointer(run_dir, save_top_k=1, save_last=True)
    with pytest.raises(CheckpointLookupError, match="catalog does not exist"):
        checkpointer.fetch(CheckpointRequest.parse("rank=1"), dest=tmp_path / "snap")


def test_mark_finished(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpointer = _checkpointer(run_dir, save_top_k=1, save_last=True)
    checkpointer.save(_FakeTrainer(epoch=0, metrics={"loss": 0.2}))
    checkpointer.mark_finished()
    assert checkpointer.fit_state().status == FIT_FINISHED
