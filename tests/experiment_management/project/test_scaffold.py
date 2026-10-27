from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.experiment_management.project.layout import (
    BACKEND_MLFLOW,
    BACKEND_PLACEHOLDER,
    BACKEND_WANDB,
    DEFAULT_IDENTITY_CONFIG,
    DEFAULT_TRACKING_BACKEND,
    EXPERIMENTS_DIRNAME,
    IDENTITY_CONFIG_FILENAME,
    LOGGING_CONFIGS_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    PROJECT_MARKER_FILENAME,
    RUN_ARTIFACTS_DIRNAME,
    SCRIPTS_DIRNAME,
    STORAGE_DIRNAME,
    TRACKING_BACKEND_FILENAME,
    TRACKING_BACKENDS,
)
from dl_toolbox.experiment_management.project.scaffold import setup_project

from tests.experiment_management.project.pl_support import N_TRIALS, random_package_name, random_project_name

_UNUSED_BACKENDS = (BACKEND_MLFLOW, BACKEND_WANDB, BACKEND_PLACEHOLDER)


def test_setup_project_writes_all_logging_configs_and_selected_storage(
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
        logging_configs = (
            root / PROJECT_LEVEL_CONFIGS_DIRNAME / LOGGING_CONFIGS_DIRNAME
        )
        expected_backends = tuple(sorted(TRACKING_BACKENDS))
        n_backends = len(expected_backends)
        for j in range(n_backends):
            assert (logging_configs / f"{expected_backends[j]}.json").is_file()
        n_unused = len(_UNUSED_BACKENDS)
        for j in range(n_unused):
            unused = _UNUSED_BACKENDS[j]
            assert not (root / PROJECT_LEVEL_CONFIGS_DIRNAME / f"{unused}.json").exists()
            assert not (root / STORAGE_DIRNAME / unused).exists()
        assert not (root / "pyproject.toml").exists()
        identity = read_json(
            root / PROJECT_LEVEL_CONFIGS_DIRNAME / IDENTITY_CONFIG_FILENAME
        )
        assert identity == DEFAULT_IDENTITY_CONFIG
        assert (
            root / STORAGE_DIRNAME / RUN_ARTIFACTS_DIRNAME / "README.md"
        ).is_file()
        assert (
            root
            / STORAGE_DIRNAME
            / "runs"
            / identity["human_readable_dirname"]
        ).is_dir()
        assert (
            root / STORAGE_DIRNAME / "runs" / identity["hashes_dirname"]
        ).is_dir()
        expected_readmes = (
            root / STORAGE_DIRNAME / "runs" / "README.md",
            root / STORAGE_DIRNAME / RUN_ARTIFACTS_DIRNAME / "README.md",
            root / STORAGE_DIRNAME / "tensorboard" / "README.md",
            logging_configs / "README.md",
            root / SCRIPTS_DIRNAME / "README.md",
            root / EXPERIMENTS_DIRNAME / "README.md",
        )
        n_readmes = len(expected_readmes)
        for j in range(n_readmes):
            assert expected_readmes[j].is_file()
            assert expected_readmes[j].read_text(encoding="utf-8").strip() != ""


def test_setup_project_creates_storage_only_for_the_chosen_backend(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "mlflow_only"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent, tracking_backend=BACKEND_MLFLOW)
    assert (root / TRACKING_BACKEND_FILENAME).read_text(encoding="utf-8") == (
        BACKEND_MLFLOW + "\n"
    )
    logging_configs = root / PROJECT_LEVEL_CONFIGS_DIRNAME / LOGGING_CONFIGS_DIRNAME
    assert (logging_configs / "mlflow.json").is_file()
    assert (logging_configs / "tensorboard.json").is_file()
    assert (root / STORAGE_DIRNAME / "mlflow" / "artifacts" / "README.md").is_file()
    assert not (root / STORAGE_DIRNAME / "tensorboard").exists()
    assert not (root / STORAGE_DIRNAME / "wandb").exists()
    assert not (root / STORAGE_DIRNAME / "placeholder").exists()


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


def test_optional_package_is_an_empty_installable_tree(
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
        init_path = root / package / "src" / package / "__init__.py"
        assert init_path.is_file()
        assert init_path.read_text(encoding="utf-8") == ""
        src_dir = root / package / "src" / package
        names = sorted(path.name for path in src_dir.iterdir())
        assert names == ["__init__.py"]


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
