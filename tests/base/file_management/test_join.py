from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.join import join_under

from tests.base.base_support import N_TRIALS, random_generating_process_name


def _write_nested_file(root: Path, rng: random.Random) -> tuple[str, Path]:
    depth = rng.randint(1, 4)
    parts = [random_generating_process_name(rng) for _ in range(depth)]
    relative = "/".join(parts) + ".npy"
    target = root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(rng.randbytes(8))
    return relative, target


def test_existing_files_resolve_under_the_artifact_dir(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        root = tmp_path / f"run_{i}"
        root.mkdir()
        relative, target = _write_nested_file(root, rng)
        found = join_under(root, relative)
        assert found == target.resolve()
        assert found.is_file()
        found_again = join_under(root, relative.replace("/", "\\"))
        assert found_again == target.resolve()


def test_missing_file_is_an_error(tmp_path: Path, rng: random.Random) -> None:
    root = tmp_path / "run"
    root.mkdir()
    for _ in range(N_TRIALS):
        relative = f"missing_{random_generating_process_name(rng)}.npy"
        with pytest.raises(FileNotFoundError):
            join_under(root, relative)


def test_paths_that_leave_the_directory_are_rejected(
    tmp_path: Path, rng: random.Random
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"no")
    escapes = ("../secret.bin", "..\\secret.bin", "../run/../secret.bin")
    for relative in escapes:
        with pytest.raises(ValueError, match="stay under"):
            join_under(root, relative)
    for _ in range(N_TRIALS):
        with pytest.raises(ValueError, match="stay under"):
            join_under(root, f"../{random_generating_process_name(rng)}.bin")


def test_relative_directory_argument_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "a.bin").write_bytes(b"ok")
    with pytest.raises(ValueError, match="absolute"):
        join_under("run", "a.bin")


def test_absolute_payload_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    target = root / "a.bin"
    target.write_bytes(b"ok")
    with pytest.raises(ValueError, match="relative"):
        join_under(root, str(target))
