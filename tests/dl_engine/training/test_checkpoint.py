from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.dl_engine.training.checkpoint import (
    CheckpointLookupError,
    CheckpointManager,
)
from dl_toolbox.dl_engine.training.config import MonitorConfig


def _manager(run_dir: Path) -> CheckpointManager:
    return CheckpointManager(
        save_top_k=1,
        save_last=True,
        run_dir=run_dir,
        monitor=MonitorConfig(key="loss", mode="min"),
    )


def _write_catalog(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    epoch_path = ckpt_dir / "epoch=003.ckpt"
    epoch_path.write_bytes(b"weights")
    last_path = ckpt_dir / "last.ckpt"
    last_path.write_bytes(b"last")
    write_json(
        ckpt_dir / "catalog.json",
        {
            "monitor": {"key": "loss", "mode": "min"},
            "save_top_k": 1,
            "save_last": True,
            "files": [
                {
                    "epoch": 3,
                    "path": "epoch=003.ckpt",
                    "score": 0.1,
                    "rank": 1,
                }
            ],
            "last": {"epoch": 3, "path": "last.ckpt"},
        },
    )
    return epoch_path


def test_lookup_epoch_rank_and_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    epoch_path = _write_catalog(run_dir)
    manager = _manager(run_dir)
    by_epoch = manager.lookup("epoch=3")
    assert by_epoch.path == epoch_path
    assert by_epoch.rank == 1
    by_rank = manager.lookup("rank=1")
    assert by_rank.epoch == 3
    by_last = manager.lookup("last")
    assert by_last.path == run_dir / "checkpoints" / "last.ckpt"
    assert by_last.kind == "last"


def test_lookup_missing_epoch_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_catalog(run_dir)
    manager = _manager(run_dir)
    with pytest.raises(CheckpointLookupError, match="epoch=9"):
        manager.lookup("epoch=9")
    with pytest.raises(CheckpointLookupError, match="must be"):
        manager.lookup("best")
