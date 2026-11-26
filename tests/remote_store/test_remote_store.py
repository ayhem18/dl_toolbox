"""RemoteStore transport, pure-logic. An in-memory fake stands in for a live
S3/MinIO backend, so these run without the ``s3`` extra or a server.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import hash_file
from dl_toolbox.remote_store.backends import RemoteBackend, parse_backend
from dl_toolbox.remote_store.connection import RemoteStoreConnection
from dl_toolbox.remote_store.locations import S3Location, parse_s3_location
from dl_toolbox.remote_store.project_mirror import ProjectRemoteMirror


class FakeS3Connection(RemoteStoreConnection):
    """In-memory object store keyed by ``(bucket, key)``."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, object]] = {}

    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def object_exists(self, *, identity: str, location: Mapping) -> bool:
        loc = parse_s3_location(location)
        return (loc.bucket, loc.path_within_bucket) in self.objects

    def push_file(self, *, identity: str, location: Mapping, source: File) -> None:
        loc = parse_s3_location(location)
        self.objects[(loc.bucket, loc.path_within_bucket)] = {
            "identity": identity,
            "bytes": source.path.read_bytes(),
        }

    def pull_file(
        self, *, identity: str, location: Mapping, destination: Directory, name: str
    ) -> File:
        loc = parse_s3_location(location)
        record = self.objects[(loc.bucket, loc.path_within_bucket)]
        dest = destination.path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(record["bytes"])  # type: ignore[arg-type]
        return File.from_absolute(path=dest)

    def push_directory(
        self, *, identity: str, location: Mapping, source: Directory
    ) -> None:
        loc = parse_s3_location(location)
        root = source.path
        prefix = loc.path_within_bucket.rstrip("/") + "/"
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            self.objects[(loc.bucket, prefix + relative)] = {
                "identity": identity,
                "bytes": path.read_bytes(),
            }

    def pull_directory(
        self, *, identity: str, location: Mapping, destination: Directory
    ) -> None:
        loc = parse_s3_location(location)
        prefix = loc.path_within_bucket.rstrip("/") + "/"
        for (bucket, key), record in self.objects.items():
            if bucket != loc.bucket or not key.startswith(prefix):
                continue
            out = destination.path / Path(key[len(prefix):])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(record["bytes"])  # type: ignore[arg-type]


def _hash_manager() -> HashManager:
    return HashManager(
        prefix_len=12,
        human_readable_dir_name="human_readable",
        hash_dir_name="hashes",
        lock_timeout_seconds=2,
        lock_poll_seconds=0.05,
    )


def _mirror(tmp_path: Path) -> tuple[ProjectRemoteMirror, FakeS3Connection, Directory]:
    storage = Directory.from_absolute(
        path=tmp_path / "storage", policy=DirectoryParsePolicy.CREATE
    )
    connection = FakeS3Connection()
    mirror = ProjectRemoteMirror(
        connection=connection,
        hash_manager=_hash_manager(),
        storage=storage,
        bucket="dl-toolbox",
    )
    return mirror, connection, storage


def _write(storage: Directory, relative: str, text: str) -> File:
    path = storage.path / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return File.from_absolute(path=path)


# --- backend + location models -------------------------------------------------


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


# --- mirror round trip ----------------------------------------------------------


def test_push_exists_pull_round_trip(tmp_path: Path) -> None:
    mirror, _connection, storage = _mirror(tmp_path)
    local = _write(storage, "run_artifacts/exp/ckpt.bin", "weights-v1")
    expected = hash_file(local.path)

    identity = mirror.push_file(local_file=local)
    assert identity == expected
    assert mirror.object_exists(
        relative_path="run_artifacts/exp/ckpt.bin", expected_hash=expected
    )

    local.path.unlink()
    pulled = mirror.pull_file(
        relative_path="run_artifacts/exp/ckpt.bin", expected_hash=expected
    )
    assert pulled.path.read_text(encoding="utf-8") == "weights-v1"


def test_object_exists_false_when_absent(tmp_path: Path) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    absent = "0" * 64
    assert not mirror.object_exists(
        relative_path="run_artifacts/exp/ckpt.bin", expected_hash=absent
    )


def test_pull_rejects_hash_mismatch(tmp_path: Path) -> None:
    mirror, connection, storage = _mirror(tmp_path)
    local = _write(storage, "a/b.bin", "good")
    expected = mirror.push_file(local_file=local)
    # Remote is source of truth; corrupt what it holds, then a pull must reject it.
    connection.objects[("dl-toolbox", "a/b.bin")]["bytes"] = b"tampered"
    local.path.unlink()
    with pytest.raises(ValueError):
        mirror.pull_file(relative_path="a/b.bin", expected_hash=expected)


def test_push_overwrites(tmp_path: Path) -> None:
    mirror, _connection, storage = _mirror(tmp_path)
    local = _write(storage, "a/b.bin", "v1")
    mirror.push_file(local_file=local)
    local.path.write_text("v2", encoding="utf-8")
    new_identity = mirror.push_file(local_file=File.from_absolute(path=local.path))
    local.path.unlink()
    pulled = mirror.pull_file(relative_path="a/b.bin", expected_hash=new_identity)
    assert pulled.path.read_text(encoding="utf-8") == "v2"


def test_index_file_written_last(tmp_path: Path) -> None:
    """A directory's presence is decided by the single config file pushed last."""
    mirror, _connection, storage = _mirror(tmp_path)
    _write(storage, "runs/exp/payload/part-0.bin", "p0")
    _write(storage, "runs/exp/payload/part-1.bin", "p1")
    config = _write(storage, "runs/exp/collection.json", '{"hash": "x"}')
    config_hash = hash_file(config.path)

    # payload up, index entry (config) not yet: resolve-or-fetch sees it as absent.
    mirror.push_directory(relative_path="runs/exp/payload", identity=config_hash)
    assert not mirror.object_exists(
        relative_path="runs/exp/collection.json", expected_hash=config_hash
    )
    # config pushed last -> now present.
    mirror.push_file(local_file=config)
    assert mirror.object_exists(
        relative_path="runs/exp/collection.json", expected_hash=config_hash
    )


def test_relative_path_outside_storage_rejected(tmp_path: Path) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    outside = tmp_path / "elsewhere.bin"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        mirror.push_file(local_file=File.from_absolute(path=outside))


@pytest.mark.parametrize("bad", ["/abs/key", "a/../b", "", "."])
def test_relative_path_rejected(tmp_path: Path, bad: str) -> None:
    mirror, _connection, _storage = _mirror(tmp_path)
    with pytest.raises(ValueError):
        mirror.object_exists(relative_path=bad, expected_hash="0" * 64)
