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
    run_identities_root = tmp_path / "run_identities"
    run_artifacts_root = tmp_path / "run_artifacts"
    run_identities_root.mkdir()
    run_artifacts_root.mkdir()
    return ExpEngine(
        identity_manager=HashManager(
            prefix_len=12,
            human_readable_dirname="human_readable",
            hashes_dirname="hashes",
        ),
        logging_manager=PlaceholderManager(log_dir=tmp_path / "logs"),
        checkpoint_manager=CheckpointManager(lock_timeout_seconds=5.0),
        experiment_manager=TinyExperimentConfigManager(),
        paths=EnginePaths(
            project_root=tmp_path,
            run_identities_root=run_identities_root,
            run_artifacts_root=run_artifacts_root,
        ),
        compose_toolbox=DlToolboxConfigManager,
        tracking_backend="placeholder",
    )


def _note(session: RunSession, record, staging: Path) -> None:
    del session, record
    (staging / "note.txt").write_text("ok", encoding="utf-8")


def test_get_new_run_and_get_existing_run_match(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_alpha", seed=7, run_name="alpha")
    session = engine.get_new_run_or_overwrite(recipe)
    assert session.human_readable_name == "run_alpha"
    assert session.run_dir == tmp_path / "run_artifacts" / "run_alpha"
    assert session.run_dir.is_dir()
    records = read_json(tmp_path / "run_identities" / "human_readable" / "run_alpha.json")
    assert records["name"] == "run_alpha"
    assert records["hash"] == session.hash
    existing = engine.get_existing_run(recipe)
    assert existing.hash == session.hash
    assert existing.run_dir == session.run_dir
    with pytest.raises(FileExistsError, match="already exists with this configuration hash"):
        engine.get_new_run_or_overwrite(recipe)
    with pytest.raises(FileExistsError, match="already exists with this configuration hash"):
        engine.get_new_run_or_overwrite(recipe, force=True)
    assert session.run_dir.is_dir()
    assert (tmp_path / "run_identities" / "human_readable" / "run_alpha.json").is_file()


def test_hash_taken_refuses_even_with_force(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    first = placeholder_recipe(name="run_alpha", seed=1, run_name="shared")
    second = placeholder_recipe(name="run_beta", seed=1, run_name="shared")
    engine.get_new_run_or_overwrite(first)
    with pytest.raises(FileExistsError, match="already saved as name"):
        engine.get_new_run_or_overwrite(second)
    with pytest.raises(FileExistsError, match="already saved as name"):
        engine.get_new_run_or_overwrite(second, force=True)
    assert (tmp_path / "run_artifacts" / "run_alpha").is_dir()
    assert not (tmp_path / "run_artifacts" / "run_beta").exists()


def test_name_taken_without_force_errors(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    first = placeholder_recipe(name="run_alpha", seed=1)
    second = placeholder_recipe(name="run_alpha", seed=2)
    first_session = engine.get_new_run_or_overwrite(first)
    marker = first_session.run_dir / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different configuration hash"):
        engine.get_new_run_or_overwrite(second)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_orphan_artifacts_dir_blocks_both_free(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_alpha")
    orphan = tmp_path / "run_artifacts" / "run_alpha"
    orphan.mkdir()
    (orphan / "leftover.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="run artifacts directory already exists"):
        engine.get_new_run_or_overwrite(recipe)
    with pytest.raises(FileExistsError, match="run artifacts directory already exists"):
        engine.get_new_run_or_overwrite(recipe, force=True)
    assert (orphan / "leftover.txt").is_file()


def test_get_existing_run_before_create_fails(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_alpha")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        engine.get_existing_run(recipe)


def test_inspect_rank_one_exclusive_epoch_folder(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    recipe = placeholder_recipe(name="run_a", run_name="tiny_run")
    session = engine.get_new_run_or_overwrite(recipe)
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
    session = engine.get_new_run_or_overwrite(recipe)
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
    session = engine.get_existing_run(recipe)
    state = engine.fit_state(session)
    assert state.status == FIT_FINISHED
    assert state.latest_epoch is not None
