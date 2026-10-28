from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.dl_engine.training.checkpointer.types import FIT_FINISHED
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import (
    PlaceholderManager,
)
from dl_toolbox.experiment_management.default_layout.layout.discover import (
    read_tracking_backend,
)
from dl_toolbox.experiment_management.default_layout.layout.names import (
    BACKEND_MLFLOW,
    BACKEND_PLACEHOLDER,
    CHECKPOINT_CONFIG_FILENAME,
    DEFAULT_CHECKPOINT_CONFIG,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
)
from dl_toolbox.experiment_management.default_layout.layout.scaffold import setup_project
from dl_toolbox.experiment_management.engine.core.session import RunSession
from dl_toolbox.experiment_management.engine.default.engine import DefaultExpEngine
from tests.experiment_management.em_support import (
    TinyExperimentConfigManager,
    placeholder_recipe,
)
from tests.experiment_management.names import random_project_name


class _FakeTrainer:
    def __init__(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.current_epoch = epoch
        self.callback_metrics = metrics
        self.sanity_checking = False

    def save_checkpoint(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(f"epoch-{self.current_epoch}".encode("utf-8"))


def _scaffold(
    tmp_path: Path, rng: random.Random
) -> tuple[str, Path, Path]:
    parent = tmp_path / "parent"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent, tracking_backend=BACKEND_PLACEHOLDER)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    return name, root, start


def _engine(
    name: str, start: Path, *, tracking_backend: str | None = None
) -> DefaultExpEngine:
    return DefaultExpEngine(
        name,
        start=start,
        experiment=TinyExperimentConfigManager(),
        tracking_backend=tracking_backend,
    )


def _note(session: RunSession, record, staging: Path) -> None:
    (staging / "note.txt").write_text("ok", encoding="utf-8")
    session.logger.log_text("inspection/note", "ok", step=record.epoch)


def test_default_engine_discovers_project_and_default_checkpoint_json(
    tmp_path: Path, rng: random.Random
) -> None:
    name, root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    assert engine.project_name == name
    assert engine.project_root == root
    assert engine.tracking_backend == BACKEND_PLACEHOLDER
    checkpoint = read_json(
        root / PROJECT_LEVEL_CONFIGS_DIRNAME / CHECKPOINT_CONFIG_FILENAME
    )
    assert checkpoint == DEFAULT_CHECKPOINT_CONFIG
    assert engine._checkpoint_manager.lock_timeout_seconds == checkpoint["lock_timeout_seconds"]


def test_default_engine_reads_custom_lock_timeout(
    tmp_path: Path, rng: random.Random
) -> None:
    name, root, start = _scaffold(tmp_path, rng)
    write_json(
        root / PROJECT_LEVEL_CONFIGS_DIRNAME / CHECKPOINT_CONFIG_FILENAME,
        {"lock_timeout_seconds": 12.5},
    )
    engine = _engine(name, start)
    assert engine._checkpoint_manager.lock_timeout_seconds == 12.5


def test_default_engine_cli_backend_overrides_file_without_rewriting_it(
    tmp_path: Path, rng: random.Random
) -> None:
    name, root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start, tracking_backend=BACKEND_MLFLOW)
    assert engine.tracking_backend == BACKEND_MLFLOW
    assert isinstance(engine._logging_manager, MlflowManager)
    assert read_tracking_backend(root) == BACKEND_PLACEHOLDER
    file_engine = _engine(name, start)
    assert file_engine.tracking_backend == BACKEND_PLACEHOLDER
    assert isinstance(file_engine._logging_manager, PlaceholderManager)


def test_default_engine_rejects_unknown_backend_override(
    tmp_path: Path, rng: random.Random
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    with pytest.raises(ValueError, match="tracking_backend must be one of"):
        _engine(name, start, tracking_backend="not-a-backend")


def test_default_engine_rejects_mismatched_project_name(
    tmp_path: Path, rng: random.Random
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    del name
    with pytest.raises(ValueError, match="does not match"):
        _engine("other_project", start)


def test_default_engine_rejects_relative_start(
    tmp_path: Path, rng: random.Random
) -> None:
    name, _root, _start = _scaffold(tmp_path, rng)
    with pytest.raises(ValueError, match="absolute"):
        DefaultExpEngine(
            name,
            start=Path("relative_start"),
            experiment=TinyExperimentConfigManager(),
        )


def test_default_engine_create_existing_and_inspect(
    tmp_path: Path, rng: random.Random
) -> None:
    name, root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    recipe = placeholder_recipe(name="run_alpha", seed=7, run_name="alpha")
    session = engine.get_new_run_or_overwrite(recipe)
    assert session.human_readable_name == "run_alpha"
    assert session.run_dir == root / "storage" / "run_artifacts" / "run_alpha"
    assert session.run_dir.is_dir()
    records = read_json(
        root / "storage" / "runs" / "human_readable" / "run_alpha.json"
    )
    assert records["name"] == "run_alpha"
    assert records["hash"] == session.hash
    existing = engine.get_existing_run(recipe)
    assert existing.hash == session.hash
    assert existing.run_dir == session.run_dir
    with pytest.raises(FileExistsError, match="already exists with this configuration hash"):
        engine.get_new_run_or_overwrite(recipe)
    with pytest.raises(FileExistsError, match="already exists with this configuration hash"):
        engine.get_new_run_or_overwrite(recipe, force=True)
    key = session.fit_config.monitor.key
    session.checkpointer.save(_FakeTrainer(epoch=3, metrics={key: 0.2}))
    dest = engine.inspect(recipe, {"selector": "rank=1"}, inspect_fn=_note)
    assert dest == session.run_dir / "inspection_runs" / "epoch=3"
    assert (dest / "note.txt").read_text(encoding="utf-8") == "ok"
    logged = (
        root / "storage" / "placeholder" / "alpha" / "inspection" / "note_step3.txt"
    )
    assert logged.read_text(encoding="utf-8") == "ok"


def test_default_engine_force_overwrites_name_and_artifacts(
    tmp_path: Path, rng: random.Random
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    first = placeholder_recipe(name="run_alpha", seed=1, run_name="alpha")
    second = placeholder_recipe(name="run_alpha", seed=2, run_name="alpha")
    first_session = engine.get_new_run_or_overwrite(first)
    marker = first_session.run_dir / "marker.txt"
    marker.write_text("old", encoding="utf-8")
    first_hash = first_session.hash
    forced = engine.get_new_run_or_overwrite(second, force=True)
    assert forced.hash != first_hash
    assert forced.run_dir == first_session.run_dir
    assert forced.run_dir.is_dir()
    assert not marker.exists()
    existing = engine.get_existing_run(second)
    assert existing.hash == forced.hash
    with pytest.raises(FileExistsError, match="different configuration hash"):
        engine.get_existing_run(first)


def test_default_engine_train_writes_finished_catalog(
    tmp_path: Path, rng: random.Random
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    recipe = placeholder_recipe(name="run_fit", seed=0, run_name="fit_run")
    trainer = engine.train(recipe)
    assert trainer.current_epoch >= 0
    session = engine.get_existing_run(recipe)
    state = engine.fit_state(session)
    assert state.status == FIT_FINISHED
    assert state.latest_epoch is not None
    assert (session.run_dir / "checkpoints" / "catalog.json").is_file()
