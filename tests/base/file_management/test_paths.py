# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.paths import (
    ManagedPath,
    PathPresence,
    PathRole,
    abs_path,
    process_path,
)

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
            process_path(
                relative,
                role=PathRole.DIRECTORY,
                presence=PathPresence.CREATE,
            )


def test_create_makes_absolute_directory(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        log_dir = tmp_path / f"run_{rng.randint(0, 10_000)}_{i}"
        managed = process_path(
            log_dir,
            role=PathRole.DIRECTORY,
            presence=PathPresence.CREATE,
        )
        assert isinstance(managed, ManagedPath)
        assert managed.role == PathRole.DIRECTORY
        assert managed.path.is_dir()
        assert managed.path.is_absolute()
        assert managed.path == log_dir


def test_create_does_not_use_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    for i in range(N_TRIALS):
        log_dir = tmp_path / f"logs_{i}"
        managed = process_path(
            log_dir,
            role=PathRole.DIRECTORY,
            presence=PathPresence.CREATE,
        )
        assert managed.path == log_dir
        assert (other / f"logs_{i}").exists() is False
        with pytest.raises(ValueError, match="absolute"):
            process_path(
                "logs_relative",
                role=PathRole.DIRECTORY,
                presence=PathPresence.CREATE,
            )


def test_file_create_is_illegal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CREATE is only valid"):
        process_path(
            tmp_path / "a.bin",
            role=PathRole.FILE,
            presence=PathPresence.CREATE,
        )


def test_must_exist_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        process_path(
            tmp_path,
            role=PathRole.FILE,
            presence=PathPresence.MUST_EXIST,
        )


def test_must_exist_directory_rejects_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    with pytest.raises(ValueError, match="must be a directory"):
        process_path(
            target,
            role=PathRole.DIRECTORY,
            presence=PathPresence.MUST_EXIST,
        )


def test_must_exist_file_accepts_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    managed = process_path(
        target,
        role=PathRole.FILE,
        presence=PathPresence.MUST_EXIST,
    )
    assert managed.path == target
    assert managed.role == PathRole.FILE


def test_allow_missing_file_does_not_create(tmp_path: Path) -> None:
    target = tmp_path / "missing.bin"
    managed = process_path(
        target,
        role=PathRole.FILE,
        presence=PathPresence.ALLOW_MISSING,
    )
    assert managed.path == target
    assert not target.exists()
