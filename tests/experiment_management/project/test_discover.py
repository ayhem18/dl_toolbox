from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.experiment_management.default_layout.layout.discover import (
    assert_package_name,
    find_project_root,
    load_project_document,
    read_tracking_backend,
)
from dl_toolbox.experiment_management.default_layout.layout.names import (
    DEFAULT_TRACKING_BACKEND,
    PROJECT_MARKER_FILENAME,
    TRACKING_BACKEND_FILENAME,
)
from dl_toolbox.experiment_management.default_layout.layout.scaffold import setup_project

from tests.experiment_management.project.pl_support import N_TRIALS, random_project_name


def test_project_root_is_the_marker_file_parent(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        parent = tmp_path / f"walk_{i}"
        parent.mkdir()
        name = random_project_name(rng)
        root = setup_project(name, parent=parent)
        nested = root / "scripts" / "nested" / "deeper"
        nested.mkdir(parents=True)
        start_file = nested / "train.py"
        start_file.write_text("x = 1\n", encoding="utf-8")
        found = find_project_root(start=start_file)
        assert found == root.resolve()
        document = load_project_document(found)
        assert document["name"] == name
        assert read_tracking_backend(found) == DEFAULT_TRACKING_BACKEND


def test_discovery_does_not_use_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "repo"
    parent.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    for i in range(N_TRIALS):
        name = random_project_name(rng)
        root = setup_project(name, parent=parent)
        start = root / "scripts" / "train.py"
        start.parent.mkdir(parents=True, exist_ok=True)
        start.write_text("x = 1\n", encoding="utf-8")
        assert find_project_root(start=start) == root.resolve()
        with pytest.raises(FileNotFoundError):
            find_project_root(start=other)


def test_relative_start_is_rejected(tmp_path: Path) -> None:
    (tmp_path / PROJECT_MARKER_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="absolute"):
        find_project_root(start=Path("relative"))


def test_tracking_backend_must_be_one_known_line(
    tmp_path: Path, rng: random.Random
) -> None:
    parent = tmp_path / "backend"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent)
    (root / TRACKING_BACKEND_FILENAME).write_text("not-a-backend\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be one of"):
        read_tracking_backend(root)
    (root / TRACKING_BACKEND_FILENAME).write_text("tensorboard\nmlflow\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one line"):
        read_tracking_backend(root)


def test_package_name_must_be_a_lowercase_identifier() -> None:
    with pytest.raises(ValueError, match="identifier"):
        assert_package_name("NotValid")
    with pytest.raises(ValueError, match="identifier"):
        assert_package_name("class")
    with pytest.raises(ValueError, match="identifier"):
        assert_package_name("pkg-name")
    assert assert_package_name("pkg_ok") == "pkg_ok"
