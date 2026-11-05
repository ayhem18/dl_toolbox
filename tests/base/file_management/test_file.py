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


def _existing_directory(path: Path) -> Directory:
    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.CREATE)


def test_from_absolute_accepts_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    found = File.from_absolute(path=target)
    assert found.path == target
    assert found.path.is_file()
    assert found.path.is_absolute()


def test_from_absolute_rejects_relative(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        relative = f"logs_{rng.randint(0, 10_000)}.bin"
        with pytest.raises(ValueError, match="absolute"):
            File.from_absolute(path=relative)


def test_from_absolute_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        File.from_absolute(path=tmp_path)


def test_from_absolute_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="must exist"):
        File.from_absolute(path=tmp_path / "missing.bin")


def test_from_under_resolves_existing_file(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        root = _existing_directory(tmp_path / f"run_{i}")
        name = f"{random_generating_process_name(rng)}.bin"
        target = root.path / name
        target.write_bytes(rng.randbytes(8))
        found = File.from_under(root=root, relative_path=name)
        assert found.path == target.resolve()
        found_slash = File.from_under(
            root=root, relative_path=name.replace("/", "\\")
        )
        assert found_slash.path == target.resolve()


def test_from_under_nested_file(tmp_path: Path, rng: random.Random) -> None:
    root = _existing_directory(tmp_path / "run")
    for i in range(N_TRIALS):
        depth = rng.randint(1, 4)
        parts = [random_generating_process_name(rng) for _ in range(depth)]
        relative = "/".join(parts) + ".npy"
        target = root.path.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(rng.randbytes(8))
        found = File.from_under(root=root, relative_path=relative)
        assert found.path == target.resolve()


def test_from_under_requires_existing_root(tmp_path: Path) -> None:
    missing = Directory.from_absolute(
        path=tmp_path / "missing_root",
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )
    assert missing.spec == DirectorySpec.MISSING
    with pytest.raises(ValueError, match="root Directory must exist"):
        File.from_under(root=missing, relative_path="a.bin")


def test_from_under_rejects_escape(tmp_path: Path) -> None:
    root = _existing_directory(tmp_path / "run")
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"no")
    with pytest.raises(ValueError, match="stay under"):
        File.from_under(root=root, relative_path="../secret.bin")


def test_from_under_rejects_missing_file(tmp_path: Path) -> None:
    root = _existing_directory(tmp_path / "run")
    with pytest.raises(FileNotFoundError):
        File.from_under(root=root, relative_path="missing.bin")


def test_from_under_rejects_raw_path_root(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x")
    with pytest.raises(TypeError, match="root must be Directory"):
        File.from_under(root=tmp_path, relative_path="a.bin")


def test_parent_is_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    found = File.from_absolute(path=target)
    parent = found.parent()
    assert parent.path == tmp_path
    assert parent.spec == DirectorySpec.EXISTS


def test_from_parent_walk_from_file_and_directory(
    tmp_path: Path, rng: random.Random
) -> None:
    marker = tmp_path / "pyproject.toml"
    marker.write_text("[project]\nname = 'x'\n", encoding="utf-8")
    for _ in range(N_TRIALS):
        nested = tmp_path.joinpath(
            *[random_generating_process_name(rng) for _ in range(rng.randint(1, 4))]
        )
        nested_dir = _existing_directory(nested)
        start_file_path = nested / "module.py"
        start_file_path.write_text("x = 1\n", encoding="utf-8")
        start_file = File.from_absolute(path=start_file_path)
        from_file = File.from_parent_walk(start=start_file, basename="pyproject.toml")
        from_dir = File.from_parent_walk(start=nested_dir, basename="pyproject.toml")
        assert from_file.path == marker.resolve()
        assert from_dir.path == marker.resolve()


def test_from_parent_walk_rejects_path_start(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x")
    with pytest.raises(TypeError, match="File or Directory"):
        File.from_parent_walk(start=tmp_path, basename="a.bin")


def test_from_parent_walk_rejects_missing_directory_start(tmp_path: Path) -> None:
    missing = Directory.from_absolute(
        path=tmp_path / "ghost",
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )
    with pytest.raises(ValueError, match="walk start Directory must exist"):
        File.from_parent_walk(start=missing, basename="a.bin")


def test_does_not_use_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"x")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    found = File.from_absolute(path=target)
    assert found.path == target
    with pytest.raises(ValueError, match="absolute"):
        File.from_absolute(path="a.bin")
