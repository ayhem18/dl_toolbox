from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from tests.base.base_support import N_TRIALS, random_generating_process_name


def test_create_makes_existing_directory(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        log_dir = tmp_path / f"run_{rng.randint(0, 10_000)}_{i}"
        found = Directory.from_absolute(
            path=log_dir, policy=DirectoryParsePolicy.CREATE
        )
        assert found.spec == DirectorySpec.EXISTS
        assert found.path.is_dir()
        assert found.path.is_absolute()
        assert found.path == log_dir


def test_create_does_not_use_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    for i in range(N_TRIALS):
        log_dir = tmp_path / f"logs_{i}"
        found = Directory.from_absolute(
            path=log_dir, policy=DirectoryParsePolicy.CREATE
        )
        assert found.path == log_dir
        assert (other / f"logs_{i}").exists() is False
        with pytest.raises(ValueError, match="absolute"):
            Directory.from_absolute(
                path="logs_relative", policy=DirectoryParsePolicy.CREATE
            )


def test_must_exist_accepts_directory(tmp_path: Path) -> None:
    found = Directory.from_absolute(
        path=tmp_path, policy=DirectoryParsePolicy.MUST_EXIST
    )
    assert found.spec == DirectorySpec.EXISTS
    assert found.path == tmp_path


def test_must_exist_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="must exist"):
        Directory.from_absolute(
            path=tmp_path / "ghost", policy=DirectoryParsePolicy.MUST_EXIST
        )


def test_must_exist_rejects_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    with pytest.raises(ValueError, match="must be a directory"):
        Directory.from_absolute(path=target, policy=DirectoryParsePolicy.MUST_EXIST)


def test_create_rejects_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    with pytest.raises(ValueError, match="must be a directory"):
        Directory.from_absolute(path=target, policy=DirectoryParsePolicy.CREATE)


def test_allow_missing_does_not_create(tmp_path: Path) -> None:
    target = tmp_path / "missing_dir"
    found = Directory.from_absolute(
        path=target, policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    assert found.spec == DirectorySpec.MISSING
    assert found.path == target
    assert not target.exists()


def test_allow_missing_on_existing_is_exists(tmp_path: Path) -> None:
    found = Directory.from_absolute(
        path=tmp_path, policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    assert found.spec == DirectorySpec.EXISTS


def test_create_is_not_stored_on_the_entity(tmp_path: Path) -> None:
    found = Directory.from_absolute(
        path=tmp_path / "made", policy=DirectoryParsePolicy.CREATE
    )
    assert found.spec == DirectorySpec.EXISTS
    assert not hasattr(found, "policy")


def test_from_under_create(
    tmp_path: Path, rng: random.Random
) -> None:
    root = Directory.from_absolute(path=tmp_path, policy=DirectoryParsePolicy.MUST_EXIST)
    for i in range(N_TRIALS):
        relative = f"out_{i}/{random_generating_process_name(rng)}"
        found = Directory.from_under(
            root=root,
            relative_path=relative,
            policy=DirectoryParsePolicy.CREATE,
        )
        assert found.spec == DirectorySpec.EXISTS
        assert found.path.is_dir()
        assert found.path.relative_to(root.path).as_posix() == relative


def test_from_under_allow_missing(tmp_path: Path) -> None:
    root = Directory.from_absolute(path=tmp_path, policy=DirectoryParsePolicy.MUST_EXIST)
    found = Directory.from_under(
        root=root,
        relative_path="cache/run",
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )
    assert found.spec == DirectorySpec.MISSING
    assert not found.path.exists()


def test_from_under_requires_existing_root(tmp_path: Path) -> None:
    missing = Directory.from_absolute(
        path=tmp_path / "ghost",
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )
    with pytest.raises(ValueError, match="root Directory must exist"):
        Directory.from_under(
            root=missing,
            relative_path="out",
            policy=DirectoryParsePolicy.CREATE,
        )


def test_from_under_rejects_escape(tmp_path: Path) -> None:
    root = Directory.from_absolute(
        path=tmp_path / "run", policy=DirectoryParsePolicy.CREATE
    )
    with pytest.raises(ValueError, match="stay under"):
        Directory.from_under(
            root=root,
            relative_path="../secret",
            policy=DirectoryParsePolicy.ALLOW_MISSING,
        )


def test_from_parent_walk(
    tmp_path: Path, rng: random.Random
) -> None:
    storage = Directory.from_absolute(
        path=tmp_path / "storage", policy=DirectoryParsePolicy.CREATE
    )
    for _ in range(N_TRIALS):
        depth = rng.randint(1, 5)
        parts = [random_generating_process_name(rng) for _ in range(depth)]
        nested = Directory.from_absolute(
            path=tmp_path.joinpath(*parts), policy=DirectoryParsePolicy.CREATE
        )
        start_file_path = nested.path / "module.py"
        start_file_path.write_text("x = 1\n", encoding="utf-8")
        start_file = File.from_absolute(path=start_file_path)
        from_file = Directory.from_parent_walk(
            start=start_file, basename="storage"
        )
        from_dir = Directory.from_parent_walk(start=nested, basename="storage")
        assert from_file.path == storage.path.resolve()
        assert from_dir.path == storage.path.resolve()
        assert from_file.spec == DirectorySpec.EXISTS


def test_from_parent_walk_rejects_path_start(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="File or Directory"):
        Directory.from_parent_walk(start=tmp_path, basename="storage")


def test_mismatched_spec_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist but spec"):
        Directory(path=tmp_path / "ghost", spec=DirectorySpec.EXISTS)
    with pytest.raises(ValueError, match="exists as a directory but spec"):
        Directory(path=tmp_path, spec=DirectorySpec.MISSING)
