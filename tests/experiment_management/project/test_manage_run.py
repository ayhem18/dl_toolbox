from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.experiment_management.loops.run_experiment import run_experiment
from dl_toolbox.experiment_management.project.default_run_manager import DefaultRunManager
from dl_toolbox.experiment_management.project.layout import BACKEND_PLACEHOLDER
from dl_toolbox.experiment_management.project.scaffold import setup_project
from tests.experiment_management.em_support import (
    TinyExperimentConfigManager,
    placeholder_recipe,
)
from tests.experiment_management.project.pl_support import random_project_name


def _project(tmp_path: Path, rng: random.Random) -> tuple[Path, DefaultRunManager]:
    parent = tmp_path / "parent"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent, tracking_backend=BACKEND_PLACEHOLDER)
    start = root / "scripts" / "train.py"
    start.write_text("x = 1\n", encoding="utf-8")
    manager = DefaultRunManager(
        name,
        start=start,
        experiment_config_manager=TinyExperimentConfigManager(),
    )
    return root, manager


def test_manage_run_claims_and_open_run_matches(
    tmp_path: Path, rng: random.Random
) -> None:
    root, manager = _project(tmp_path, rng)
    recipe = placeholder_recipe(name="run_alpha", seed=7, run_name="alpha")
    signature = manager.manage_run(recipe)
    assert signature.human_readable_name == "run_alpha"
    assert signature.run_dir == root / "storage" / "run_artifacts" / "run_alpha"
    assert signature.run_dir.is_dir()
    records = read_json(
        root / "storage" / "runs" / "human_readable" / "run_alpha.json"
    )
    assert records["name"] == "run_alpha"
    assert records["hash"] == signature.hash
    opened = manager.open_run(recipe)
    assert opened.hash == signature.hash
    assert opened.run_dir == signature.run_dir
    with pytest.raises(FileExistsError, match="already claimed"):
        manager.manage_run(recipe)


def test_open_run_refuses_unclaimed_name(
    tmp_path: Path, rng: random.Random
) -> None:
    _root, manager = _project(tmp_path, rng)
    recipe = placeholder_recipe(name="missing_run")
    with pytest.raises(FileNotFoundError, match="has not been claimed"):
        manager.open_run(recipe)


def test_run_experiment_from_manage_run(
    tmp_path: Path, rng: random.Random
) -> None:
    _root, manager = _project(tmp_path, rng)
    recipe = placeholder_recipe(name="fit_run", seed=1, run_name="fit")
    signature = manager.manage_run(recipe)
    trainer = run_experiment(signature)
    assert trainer.current_epoch >= 0
    assert (signature.run_dir / "checkpoints" / "catalog.json").is_file()
