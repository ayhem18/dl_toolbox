from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.parent_walk import find_file_in_parents, find_named_directory

from tests.base.base_support import N_TRIALS, random_generating_process_name


def test_named_directory_is_found_from_nested_start(
    tmp_path: Path, rng: random.Random
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    for _ in range(N_TRIALS):
        depth = rng.randint(1, 5)
        parts = [random_generating_process_name(rng) for _ in range(depth)]
        nested = tmp_path.joinpath(*parts)
        nested.mkdir(parents=True, exist_ok=True)
        start_file = nested / "module.py"
        start_file.write_text("x = 1\n", encoding="utf-8")
        found = find_named_directory("storage", start=start_file)
        assert found == storage.resolve()


def test_named_file_is_found_from_nested_start(
    tmp_path: Path, rng: random.Random
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'x'\n", encoding="utf-8")
    for _ in range(N_TRIALS):
        nested = tmp_path.joinpath(
            *[random_generating_process_name(rng) for _ in range(rng.randint(1, 4))]
        )
        nested.mkdir(parents=True, exist_ok=True)
        found = find_file_in_parents("pyproject.toml", start=nested)
        assert found == pyproject.resolve()


def test_start_must_be_absolute(tmp_path: Path) -> None:
    (tmp_path / "storage").mkdir()
    with pytest.raises(ValueError, match="absolute"):
        find_named_directory("storage", start="storage")


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="storage"):
        find_named_directory("storage", start=tmp_path)


def test_does_not_use_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "repo" / "storage"
    storage.mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    for _ in range(N_TRIALS):
        found = find_named_directory("storage", start=tmp_path / "repo")
        assert found == storage.resolve()
        with pytest.raises(FileNotFoundError):
            find_named_directory("storage", start=other)


def test_basename_required(tmp_path: Path, rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        left = random_generating_process_name(rng)
        right = random_generating_process_name(rng)
        with pytest.raises(ValueError, match="basename"):
            find_named_directory(f"{left}/{right}", start=tmp_path)
