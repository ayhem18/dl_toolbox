"""RemoteStore transport + mirror, pure-logic over the in-memory fake. No boto3,
no server, no hashing: everything is keyed by location.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import Directory, DirectoryParsePolicy
from dl_toolbox.base.file_management.file import File
from dl_toolbox.remote_store.backends import RemoteBackend, parse_backend
from dl_toolbox.remote_store.connection import DIRECTORY_MANIFEST_NAME
from dl_toolbox.remote_store.locations import S3Location, parse_s3_location
from dl_toolbox.remote_store.project_mirror import ProjectRemoteMirror
from tests.remote_store.support import InMemoryConnection


def _mirror(tmp_path: Path) -> tuple[ProjectRemoteMirror, InMemoryConnection, Directory]:
    storage = Directory.from_absolute(
        path=tmp_path / "storage", policy=DirectoryParsePolicy.CREATE
    )
    connection = InMemoryConnection()
    mirror = ProjectRemoteMirror(
        connection=connection, storage=storage, bucket="dl-toolbox"
    )
    return mirror, connection, storage


def _write(storage: Directory, relative: str, text: str) -> File:
    path = storage.path / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return File.from_absolute(path=path)


# --- backend + location models ------------------------------------------------


def test_parse_backend() -> None:
    assert parse_backend("s3") == RemoteBackend.S3
    assert parse_backend("huggingface") == RemoteBackend.HUGGINGFACE
    with pytest.raises(ValueError):
        parse_backend("gcs")


def test_s3_location_valid() -> None:
    loc = parse_s3_location({"bucket": "b", "path_within_bucket": "runs/a.json"})
    assert isinstance(loc, S3Location)
    assert loc.bucket == "b"
    assert loc.path_within_bucket == "runs/a.json"


@pytest.mark.parametrize(
    "payload",
    [
        {"bucket": "b"},
        {"bucket": "b", "path_within_bucket": "a", "extra": 1},
        {"bucket": "b", "path_within_bucket": "/abs/key"},
        {"bucket": "b", "path_within_bucket": "a/../b"},
        {"bucket": "", "path_within_bucket": "a"},
    ],
)
def test_s3_location_rejects(payload: dict) -> None:
    with pytest.raises(ValueError):
        parse_s3_location(payload)


# --- file ops -----------------------------------------------------------------


def test_push_exists_pull_file(tmp_path: Path) -> None:
    mirror, _connection, storage = _mirror(tmp_path)
    local = _write(storage, "run_artifacts/exp/ckpt.bin", "weights-v1")

    relative = mirror.push_file(local_file=local)
    assert relative == "run_artifacts/exp/ckpt.bin"
    assert mirror.exists(relative_path="run_artifacts/exp/ckpt.bin")

    local.path.unlink()
    pulled = mirror.pull_file(relative_path="run_artifacts/exp/ckpt.bin")
    assert pulled.path.read_text(encoding="utf-8") == "weights-v1"


def test_exists_false_when_absent(tmp_path: Path) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    assert not mirror.exists(relative_path="run_artifacts/exp/ckpt.bin")


def test_push_file_overwrites(tmp_path: Path) -> None:
    mirror, _connection, storage = _mirror(tmp_path)
    local = _write(storage, "a/b.bin", "v1")
    mirror.push_file(local_file=local)
    local.path.write_text("v2", encoding="utf-8")
    mirror.push_file(local_file=File.from_absolute(path=local.path))
    local.path.unlink()
    pulled = mirror.pull_file(relative_path="a/b.bin")
    assert pulled.path.read_text(encoding="utf-8") == "v2"


def test_push_file_rejects_path_outside_storage(tmp_path: Path) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    outside = tmp_path / "elsewhere.bin"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        mirror.push_file(local_file=File.from_absolute(path=outside))


@pytest.mark.parametrize("bad", ["/abs/key", "a/../b", "", "."])
def test_relative_path_rejected(tmp_path: Path, bad: str) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    with pytest.raises(ValueError):
        mirror.exists(relative_path=bad)


# --- directory ops (manifest) -------------------------------------------------


def test_push_then_pull_directory(tmp_path: Path) -> None:
    mirror, connection, storage = _mirror(tmp_path)
    _write(storage, "runs/exp/a.bin", "a")
    _write(storage, "runs/exp/sub/b.bin", "b")

    mirror.push_directory(relative_path="runs/exp")
    # manifest written last, under the prefix
    assert ("dl-toolbox", f"runs/exp/{DIRECTORY_MANIFEST_NAME}") in connection.objects

    # wipe local payload, pull it back from the remote truth
    (storage.path / "runs" / "exp" / "a.bin").unlink()
    (storage.path / "runs" / "exp" / "sub" / "b.bin").unlink()
    mirror.pull_directory(relative_path="runs/exp")
    assert (storage.path / "runs" / "exp" / "a.bin").read_text(encoding="utf-8") == "a"
    assert (
        storage.path / "runs" / "exp" / "sub" / "b.bin"
    ).read_text(encoding="utf-8") == "b"


def test_pull_directory_without_push_errors(tmp_path: Path) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    with pytest.raises(FileNotFoundError):
        mirror.pull_directory(relative_path="runs/exp")


def test_push_directory_rejects_reserved_manifest_name(tmp_path: Path) -> None:
    mirror, _connection, storage = _mirror(tmp_path)
    _write(storage, f"runs/exp/{DIRECTORY_MANIFEST_NAME}", "oops")
    with pytest.raises(ValueError, match="reserved"):
        mirror.push_directory(relative_path="runs/exp")
