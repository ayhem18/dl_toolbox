from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.paths import abs_path, process_path

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def test_relative_paths_are_rejected(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        relative = f"logs_{rng.randint(0, 10_000)}"
        with pytest.raises(ValueError, match="absolute"):
            abs_path(relative)
        with pytest.raises(ValueError, match="absolute"):
            process_path(relative, dir_ok=True, file_ok=False, must_exist=False)


def test_absolute_directory_is_created(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        log_dir = tmp_path / f"run_{rng.randint(0, 10_000)}_{i}"
        created = process_path(log_dir, dir_ok=True, file_ok=False, must_exist=False)
        assert created.is_dir()
        assert created.is_absolute()


def test_does_not_use_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    for i in range(N_TRIALS):
        log_dir = tmp_path / f"logs_{i}"
        created = process_path(log_dir, dir_ok=True, file_ok=False, must_exist=False)
        assert created == log_dir
        assert (other / f"logs_{i}").exists() is False
        with pytest.raises(ValueError, match="absolute"):
            process_path("logs_relative", dir_ok=True, file_ok=False, must_exist=False)
