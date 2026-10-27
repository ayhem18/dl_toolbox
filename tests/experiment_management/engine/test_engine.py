from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.training.checkpointer.manager import CheckpointManager
from dl_toolbox.dl_engine.training.checkpointer.types import FIT_FINISHED, FIT_RUNNING
from dl_toolbox.experiment_management.engine.core.engine import ExpEngine
from dl_toolbox.experiment_management.engine.core.paths import EnginePaths
from dl_toolbox.experiment_management.engine.core.session import RunSession
from dl_toolbox.experiment_management.toolbox.manager import DlToolboxConfigManager
from tests.experiment_management.em_support import (
    TinyExperimentConfigManager,
    placeholder_recipe,
)


class _FakeTrainer:
    def __init__(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.current_epoch = epoch
        self.callback_metrics = metrics
        self.sanity_checking = False

    def save_checkpoint(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(f"epoch-{self.current_epoch}".encode("utf-8"))


def _engine(tmp_path: Path) -> ExpEngine:
    records = tmp_path / "records"
    payload_root = tmp_path / "payload"
    records.mkdir()
    payload_root.mkdir()
    return ExpEngine(
        identity=HashManager(
            prefix_len=12,
            human_readable_dirname="human_readable",
            hashes_dirname="hashes",
        ),
        logging=PlaceholderManager(log_dir=tmp_path / "logs"),
        checkpoint=CheckpointManager(lock_timeout_seconds=5.0),
        experiment=TinyExperimentConfigManager(),
        paths=EnginePaths(
            project_root=tmp_path,
            records_dir=records,
            payload_root=payload_root,
        ),
        compose_toolbox=DlToolboxConfigManager,
        tracking_backend="placeholder",
    )


def _note(session: RunSession, record, staging: Path) -> None:
    del session, record
    (staging / "note.txt").write_text("ok", encoding="utf-8")


def test_claim_and_open_match(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_alpha", seed=7, run_name="alpha")
    session = engine.claim(recipe)
    assert session.human_readable_name == "run_alpha"
    assert session.run_dir == tmp_path / "payload" / "run_alpha"
    assert session.run_dir.is_dir()
    records = read_json(tmp_path / "records" / "human_readable" / "run_alpha.json")
    assert records["name"] == "run_alpha"
    assert records["hash"] == session.hash
    opened = engine.open(recipe)
    assert opened.hash == session.hash
    assert opened.run_dir == session.run_dir
    with pytest.raises(FileExistsError, match="already claimed"):
        engine.claim(recipe)


def test_open_before_claim_fails(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_alpha")
    with pytest.raises(FileNotFoundError, match="has not been claimed"):
        engine.open(recipe)


def test_inspect_rank_one_exclusive_epoch_folder(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_a", run_name="tiny_run")
    session = engine.claim(recipe)
    key = session.fit_config.monitor.key
    session.checkpointer.save(_FakeTrainer(epoch=3, metrics={key: 0.2}))
    dest = engine.inspect(
        recipe,
        {"selector": "rank=1"},
        inspect_fn=_note,
    )
    assert dest == session.run_dir / "inspection_runs" / "epoch=3"
    assert (dest / "note.txt").read_text(encoding="utf-8") == "ok"
    manifest = read_json(dest / "manifest.json")
    assert manifest["epoch"] == 3
    published = tmp_path / "logs" / "tiny_run" / "inspection" / "epoch=3" / "note.txt"
    assert published.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        engine.inspect(recipe, {"selector": "rank=1"}, inspect_fn=_note)


def test_fit_state_from_catalog(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_a")
    session = engine.claim(recipe)
    claimed = engine.fit_state(session)
    assert claimed.status == "claimed"
    assert claimed.latest_epoch is None
    key = session.fit_config.monitor.key
    session.checkpointer.save(_FakeTrainer(epoch=2, metrics={key: 0.4}))
    running = engine.fit_state(session)
    assert running.status == FIT_RUNNING
    assert running.latest_epoch == 2
    assert running.current_best_epoch == 2
    session.checkpointer.mark_finished()
    assert engine.fit_state(session).status == FIT_FINISHED


def test_train_writes_finished_catalog(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_fit", seed=0, run_name="fit_run")
    trainer = engine.train(recipe)
    assert trainer.current_epoch >= 0
    session = engine.open(recipe)
    state = engine.fit_state(session)
    assert state.status == FIT_FINISHED
    assert state.latest_epoch is not None
