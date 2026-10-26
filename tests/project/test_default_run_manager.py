from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.project.default_run_manager import DefaultRunManager
from dl_toolbox.project.layout import (
    BACKEND_PLACEHOLDER,
    TRACKING_BACKEND_FILENAME,
    resolve_under_project,
)
from dl_toolbox.project.scaffold import setup_project

from _support import N_TRIALS, random_package_name, random_project_name


def test_default_manager_resolves_relative_backend_paths(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        parent = tmp_path / f"mgr_{i}"
        parent.mkdir()
        name = random_project_name(rng)
        root = setup_project(name, parent=parent)
        start = root / "scripts" / "train.py"
        start.write_text("x = 1\n", encoding="utf-8")
        manager = DefaultRunManager(name, start=start)
        assert manager.project_root() == root.resolve()
        assert manager.project_runs_dir() == (root / "storage" / "runs").resolve()
        assert manager.tracking_backend() == "tensorboard"
        tensorboard = manager.get_tensorboard_global_config()
        assert Path(tensorboard["log_dir"]) == (root / "storage" / "tensorboard").resolve()
        assert Path(tensorboard["log_dir"]).is_absolute()
        placeholder = manager.get_placeholder_global_config()
        assert Path(placeholder["log_dir"]) == (root / "storage" / "placeholder").resolve()
        wandb = manager.get_wandb_global_config()
        assert wandb["project"] == name
        assert Path(wandb["dir"]) == (root / "storage" / "wandb").resolve()
        mlflow = manager.get_mlflow_global_config()
        assert Path(mlflow["backend_store"]["path"]) == (
            root / "storage" / "mlflow" / "tracking.db"
        ).resolve()
        assert Path(mlflow["artifact_store"]["root"]) == (
            root / "storage" / "mlflow" / "artifacts"
        ).resolve()
        assert manager.mlflow_experiment() == name
        selected = manager.selected_backend_setup()
        assert selected == tensorboard


def test_selector_file_chooses_the_backend(tmp_path: Path, rng: random.Random) -> None:
    parent = tmp_path / "selector"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent)
    (root / TRACKING_BACKEND_FILENAME).write_text(
        BACKEND_PLACEHOLDER + "\n", encoding="utf-8"
    )
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    manager = DefaultRunManager(name, start=start)
    assert manager.tracking_backend() == BACKEND_PLACEHOLDER
    assert manager.selected_backend_setup() == manager.get_placeholder_global_config()


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
        DefaultRunManager(other, start=start)


def test_package_field_is_exposed(tmp_path: Path, rng: random.Random) -> None:
    parent = tmp_path / "with_pkg"
    parent.mkdir()
    name = random_project_name(rng)
    package = random_package_name(rng)
    root = setup_project(name, parent=parent, package=package)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    manager = DefaultRunManager(name, start=start)
    assert manager.package_name == package
    plain_parent = tmp_path / "plain"
    plain_parent.mkdir()
    plain_name = random_project_name(rng)
    plain_root = setup_project(plain_name, parent=plain_parent)
    plain_start = plain_root / "scripts" / "train.py"
    plain_start.write_text("x = 1\n", encoding="utf-8")
    plain = DefaultRunManager(plain_name, start=plain_start)
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
