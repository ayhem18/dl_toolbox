from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.data_contract.json_io import read_json
from dl_toolbox.project.layout import (
    DEFAULT_TRACKING_BACKEND,
    EXPERIMENTS_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    PROJECT_MARKER_FILENAME,
    SCRIPTS_DIRNAME,
    STORAGE_DIRNAME,
    TRACKING_BACKEND_FILENAME,
)
from dl_toolbox.project.scaffold import setup_project

from _support import N_TRIALS, random_package_name, random_project_name


def test_setup_project_writes_marker_selector_and_backend_configs(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        parent = tmp_path / f"parent_{i}"
        parent.mkdir()
        name = random_project_name(rng)
        root = setup_project(name, parent=parent)
        assert root == parent / name
        assert (root / PROJECT_MARKER_FILENAME).is_file()
        assert (root / TRACKING_BACKEND_FILENAME).is_file()
        marker = read_json(root / PROJECT_MARKER_FILENAME)
        assert marker == {"name": name}
        backend = (root / TRACKING_BACKEND_FILENAME).read_text(encoding="utf-8")
        assert backend == DEFAULT_TRACKING_BACKEND + "\n"
        configs = root / PROJECT_LEVEL_CONFIGS_DIRNAME
        assert (configs / "tensorboard.json").is_file()
        assert (configs / "mlflow.json").is_file()
        assert (configs / "wandb.json").is_file()
        assert (configs / "placeholder.json").is_file()
        assert not (root / "pyproject.toml").exists()
        expected_readmes = (
            root / STORAGE_DIRNAME / "runs" / "README.md",
            root / STORAGE_DIRNAME / "tensorboard" / "README.md",
            root / STORAGE_DIRNAME / "mlflow" / "README.md",
            root / SCRIPTS_DIRNAME / "README.md",
            root / EXPERIMENTS_DIRNAME / "README.md",
        )
        n_readmes = len(expected_readmes)
        for j in range(n_readmes):
            assert expected_readmes[j].is_file()
            assert expected_readmes[j].read_text(encoding="utf-8").strip() != ""


def test_setup_project_refuses_an_existing_directory(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        parent = tmp_path / f"exists_{i}"
        parent.mkdir()
        name = random_project_name(rng)
        setup_project(name, parent=parent)
        with pytest.raises(FileExistsError, match="already exists"):
            setup_project(name, parent=parent)


def test_setup_project_rejects_relative_parent(
    tmp_path: Path, rng: random.Random
) -> None:
    del tmp_path
    for _i in range(N_TRIALS):
        with pytest.raises(ValueError, match="absolute"):
            setup_project(random_project_name(rng), parent=Path("relative_parent"))


def test_optional_package_gets_its_own_pyproject(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        parent = tmp_path / f"pkg_parent_{i}"
        parent.mkdir()
        name = random_project_name(rng)
        package = random_package_name(rng)
        root = setup_project(name, parent=parent, package=package)
        marker = read_json(root / PROJECT_MARKER_FILENAME)
        assert marker == {"name": name, "package": package}
        assert not (root / "pyproject.toml").exists()
        package_pyproject = root / package / "pyproject.toml"
        assert package_pyproject.is_file()
        text = package_pyproject.read_text(encoding="utf-8")
        assert f'name = "{package}"' in text
        assert (root / package / "src" / package / "__init__.py").is_file()
        assert (root / package / "src" / package / "__init__.py").read_text(
            encoding="utf-8"
        ) == ""
        run_manager = root / package / "src" / package / "run_manager.py"
        assert run_manager.is_file()
        source = run_manager.read_text(encoding="utf-8")
        assert "class " in source
        assert "DefaultRunManager" in source
        assert name in source


def test_package_name_cannot_be_a_reserved_folder(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "reserved"
    parent.mkdir()
    name = random_project_name(rng)
    with pytest.raises(ValueError, match="reserved"):
        setup_project(name, parent=parent, package="storage")
    with pytest.raises(ValueError, match="reserved"):
        setup_project(name, parent=parent, package="scripts")
    assert not (parent / name).exists()
