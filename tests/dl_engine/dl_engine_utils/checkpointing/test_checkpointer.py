# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from pathlib import Path

import pytest
import torch

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    FIT_FINISHED,
    FIT_RUNNING,
    CheckpointLookupError,
    CheckpointMonitor,
    CheckpointPolicy,
    CheckpointRequest,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.write import (
    weights_writer,
    write_weights,
)


def _checkpointer(
    run_artifacts_dir: Path, *, save_top_k: int, save_last: bool
) -> InspectableCheckpointer:
    return InspectableCheckpointer(
        run_artifacts_dir=run_artifacts_dir,
        policy=CheckpointPolicy(save_top_k=save_top_k, save_last=save_last),
        monitor=CheckpointMonitor(key="loss", mode="min"),
        max_epochs=10,
        lock_timeout_seconds=5.0,
        lock_poll_seconds=0.05,
    )


def _write_epoch(epoch: int):
    def write_function(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"epoch-{epoch}".encode("utf-8"))

    return write_function


def test_save_and_fetch_by_epoch_rank_last(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=-1, save_last=True)
    wrote = checkpointer.save(
        epoch=3, score=0.2, write_function=_write_epoch(3)
    )
    assert wrote is True
    dest = tmp_path / "snap"
    by_epoch = checkpointer.fetch(
        CheckpointRequest.parse("epoch=3"), dest=dest / "e"
    )
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
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=1, save_last=False)
    checkpointer.save(epoch=0, score=0.5, write_function=_write_epoch(0))
    checkpointer.save(epoch=1, score=0.1, write_function=_write_epoch(1))
    dest = tmp_path / "snap"
    best = checkpointer.fetch(CheckpointRequest.parse("rank=1"), dest=dest)
    assert best.epoch == 1
    with pytest.raises(CheckpointLookupError, match="epoch=0"):
        checkpointer.fetch(
            CheckpointRequest.parse("epoch=0"), dest=tmp_path / "gone"
        )
    live = list((run_artifacts_dir / "checkpoints" / "live").glob("*.ckpt"))
    assert len(live) == 1


def test_save_last_keeps_file_outside_top_k(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=1, save_last=True)
    checkpointer.save(epoch=0, score=0.1, write_function=_write_epoch(0))
    checkpointer.save(epoch=1, score=0.9, write_function=_write_epoch(1))
    dest = tmp_path / "snap"
    best = checkpointer.fetch(
        CheckpointRequest.parse("rank=1"), dest=dest / "best"
    )
    assert best.epoch == 0
    last = checkpointer.fetch(CheckpointRequest.parse("last"), dest=dest / "last")
    assert last.epoch == 1
    with pytest.raises(CheckpointLookupError, match="epoch=1"):
        checkpointer.fetch(
            CheckpointRequest.parse("epoch=1"), dest=dest / "e1"
        )


def test_worse_score_skips_write(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=1, save_last=False)
    checkpointer.save(epoch=0, score=0.1, write_function=_write_epoch(0))
    calls: list[Path] = []

    def write_function(path: Path) -> None:
        calls.append(path)
        raise AssertionError("write must not run when the score is skipped")

    wrote = checkpointer.save(epoch=1, score=0.9, write_function=write_function)
    assert wrote is False
    assert calls == []
    live = list((run_artifacts_dir / "checkpoints" / "live").glob("*.ckpt"))
    assert len(live) == 1
    state = checkpointer.fit_state()
    assert state.latest_epoch == 1
    assert state.current_best_epoch == 0
    with pytest.raises(CheckpointLookupError, match="epoch=1"):
        checkpointer.fetch(
            CheckpointRequest.parse("epoch=1"), dest=tmp_path / "gone"
        )


def test_fetch_missing_catalog_fails_closed(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=1, save_last=True)
    with pytest.raises(CheckpointLookupError, match="catalog does not exist"):
        checkpointer.fetch(
            CheckpointRequest.parse("rank=1"), dest=tmp_path / "snap"
        )


def test_mark_finished(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=1, save_last=True)
    checkpointer.save(epoch=0, score=0.2, write_function=_write_epoch(0))
    checkpointer.mark_finished()
    assert checkpointer.fit_state().status == FIT_FINISHED


def test_write_weights_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "w.ckpt"
    state = {"net.weight": torch.tensor([1.5, 2.5])}
    write_weights(path=path, state_dict=state)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload.keys()) == {"state_dict"}
    loaded = payload["state_dict"]
    assert torch.equal(loaded["net.weight"], state["net.weight"])


def test_save_with_weights_writer(tmp_path: Path) -> None:
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir()
    checkpointer = _checkpointer(run_artifacts_dir, save_top_k=1, save_last=False)
    state = {"net.bias": torch.zeros(2)}
    wrote = checkpointer.save(
        epoch=0, score=0.3, write_function=weights_writer(state_dict=state)
    )
    assert wrote is True
    record = checkpointer.fetch(
        CheckpointRequest.parse("rank=1"), dest=tmp_path / "snap"
    )
    payload = torch.load(record.path, map_location="cpu", weights_only=False)
    assert torch.equal(payload["state_dict"]["net.bias"], state["net.bias"])


def test_checkpoint_request_parse() -> None:
    last = CheckpointRequest.parse("last")
    assert last.kind == "last"
    epoch = CheckpointRequest.parse("epoch=3")
    assert epoch.kind == "epoch"
    assert epoch.epoch == 3
    with pytest.raises(CheckpointLookupError, match="rank must be >= 1"):
        CheckpointRequest.parse("rank=0")
    with pytest.raises(CheckpointLookupError, match="selector must be"):
        CheckpointRequest.parse("best")
