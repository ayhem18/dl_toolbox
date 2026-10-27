from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager
from dl_toolbox.dl_engine.loggers.wandb.wandb_manager import WandbManager
from dl_toolbox.experiment_management.project.default_run_manager import DefaultRunManager
from dl_toolbox.experiment_management.default_layout.layout.names import (
    BACKEND_MLFLOW,
    BACKEND_PLACEHOLDER,
    BACKEND_WANDB,
    LOGGING_CONFIGS_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    TRACKING_BACKEND_FILENAME,
)
from dl_toolbox.experiment_management.default_layout.layout.scaffold import setup_project
from dl_toolbox.experiment_management.project.abstract_run_manager import AbstractRunManager

from tests.experiment_management.project.pl_support import (
    make_layout_run_manager,
    random_package_name,
    random_project_name,
)


def test_default_manager_is_a_run_manager(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "abc"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    manager = make_layout_run_manager(name, start=start)
    assert isinstance(manager, AbstractRunManager)
    assert isinstance(manager, DefaultRunManager)
    assert manager.project_name == name
    assert manager.project_runs_dir() == (root / "storage" / "runs").resolve()
    assert manager.tracking_backend() == "tensorboard"
    logging_manager = manager.make_logging_manager()
    assert isinstance(logging_manager, TensorBoardManager)
    assert logging_manager.log_dir == (root / "storage" / "tensorboard").resolve()
    hash_manager = manager.make_hash_manager()
    assert hash_manager.prefix_len == 12
    assert hash_manager.human_readable_dirname == "human_readable"
    assert hash_manager.hashes_dirname == "hashes"
    assert manager.run_payload_dir("run_a") == (
        root / "storage" / "run_artifacts" / "run_a"
    )


def test_store_json_has_no_wandb_project_or_mlflow_experiment(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "store_keys"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent)
    logging_dir = root / PROJECT_LEVEL_CONFIGS_DIRNAME / LOGGING_CONFIGS_DIRNAME
    wandb = read_json(logging_dir / "wandb.json")
    assert wandb == {"dir": "storage/wandb"}
    mlflow = read_json(logging_dir / "mlflow.json")
    assert "experiment" not in mlflow
    assert set(mlflow) == {"backend_store", "artifact_store"}


def test_make_logging_manager_follows_the_selector(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "selector"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(
        name, parent=parent, tracking_backend=BACKEND_PLACEHOLDER
    )
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    manager = make_layout_run_manager(name, start=start)
    logging_manager = manager.make_logging_manager()
    assert isinstance(logging_manager, PlaceholderManager)
    assert logging_manager.log_dir == (root / "storage" / "placeholder").resolve()


def test_switching_selector_builds_the_matching_store(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "switch"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    (root / TRACKING_BACKEND_FILENAME).write_text(
        BACKEND_WANDB + "\n", encoding="utf-8"
    )
    wandb_manager = make_layout_run_manager(name, start=start).make_logging_manager()
    assert isinstance(wandb_manager, WandbManager)
    assert wandb_manager.dir == (root / "storage" / "wandb").resolve()
    assert not hasattr(wandb_manager, "project")
    (root / TRACKING_BACKEND_FILENAME).write_text(
        BACKEND_MLFLOW + "\n", encoding="utf-8"
    )
    mlflow_manager = make_layout_run_manager(name, start=start).make_logging_manager()
    assert isinstance(mlflow_manager, MlflowManager)
    assert mlflow_manager.artifact_root == (
        root / "storage" / "mlflow" / "artifacts"
    ).resolve()


def test_wrong_project_name_is_rejected(tmp_path: Path, rng: random.Random) -> None:
    parent = tmp_path / "mismatch"
    parent.mkdir()
    name = random_project_name(rng)
    other = random_project_name(rng)
    if other == name:
        other = name + "_x"
    root = setup_project(name, parent=parent)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        make_layout_run_manager(other, start=start)


def test_package_field_is_exposed(tmp_path: Path, rng: random.Random) -> None:
    parent = tmp_path / "with_pkg"
    parent.mkdir()
    name = random_project_name(rng)
    package = random_package_name(rng)
    root = setup_project(name, parent=parent, package=package)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    manager = make_layout_run_manager(name, start=start)
    assert manager.package_name == package
    plain_parent = tmp_path / "plain"
    plain_parent.mkdir()
    plain_name = random_project_name(rng)
    plain_root = setup_project(plain_name, parent=plain_parent)
    plain_start = plain_root / "scripts" / "train.py"
    plain_start.write_text("x = 1\n", encoding="utf-8")
    plain = make_layout_run_manager(plain_name, start=plain_start)
    assert plain.package_name is None


def test_relative_backend_path_cannot_escape_the_project(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "escape"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent)
    with pytest.raises(ValueError, match="stay under"):
        resolve_under_project(root.resolve(), "../outside", label="log_dir")
    with pytest.raises(ValueError, match="relative"):
        resolve_under_project(root.resolve(), str(tmp_path / "abs"), label="log_dir")
