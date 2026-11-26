"""remote-push / remote-pull / remote-exists CLI, end to end over a stateful
in-memory fake connection (S3Credentials.connect is monkeypatched), so these
run without boto3 or a server.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import Directory, DirectoryParsePolicy
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import JsonTextForm, write_object
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.cli import main
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.connection import RemoteStoreConnection
from dl_toolbox.remote_store.credentials import S3Credentials
from dl_toolbox.remote_store.locations import parse_s3_location

_CREDENTIALS_DIR = "project_level_configs/remote/credentials"


class _FakeConnection(RemoteStoreConnection):
    """In-memory store shared across connect() calls within one test."""

    def __init__(self, objects: dict[tuple[str, str], dict[str, object]]) -> None:
        self._objects = objects

    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def object_exists(self, *, identity: str, location: Mapping) -> bool:
        loc = parse_s3_location(location)
        return (loc.bucket, loc.path_within_bucket) in self._objects

    def push_file(self, *, identity: str, location: Mapping, source: File) -> None:
        loc = parse_s3_location(location)
        self._objects[(loc.bucket, loc.path_within_bucket)] = {
            "identity": identity,
            "bytes": source.path.read_bytes(),
        }

    def pull_file(
        self, *, identity: str, location: Mapping, destination: Directory, name: str
    ) -> File:
        loc = parse_s3_location(location)
        record = self._objects[(loc.bucket, loc.path_within_bucket)]
        dest = destination.path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(record["bytes"])  # type: ignore[arg-type]
        return File.from_absolute(path=dest)

    def push_directory(
        self, *, identity: str, location: Mapping, source: Directory
    ) -> None:
        loc = parse_s3_location(location)
        prefix = loc.path_within_bucket.rstrip("/") + "/"
        for path in sorted(p for p in source.path.rglob("*") if p.is_file()):
            relative = path.relative_to(source.path).as_posix()
            self._objects[(loc.bucket, prefix + relative)] = {
                "identity": identity,
                "bytes": path.read_bytes(),
            }

    def pull_directory(
        self, *, identity: str, location: Mapping, destination: Directory
    ) -> None:
        loc = parse_s3_location(location)
        prefix = loc.path_within_bucket.rstrip("/") + "/"
        for (bucket, key), record in self._objects.items():
            if bucket != loc.bucket or not key.startswith(prefix):
                continue
            out = destination.path / Path(key[len(prefix):])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(record["bytes"])  # type: ignore[arg-type]


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    name = "demo"
    root = Directory.from_absolute(
        path=tmp_path / name, policy=DirectoryParsePolicy.CREATE
    )
    ProjectLayout.scaffold(
        root=root,
        project_name=name,
        tracking_backend=TrackingBackend.TENSORBOARD,
        remote_backend="s3",
    )
    credentials_dir = Directory.from_under(
        root=root, relative_path=_CREDENTIALS_DIR, policy=DirectoryParsePolicy.MUST_EXIST
    )
    write_object(
        directory=credentials_dir,
        name="s3.json",
        payload={"access_key_id": "AKIA", "secret_access_key": "secret"},
        form=JsonTextForm.DOCUMENT,
    )
    objects: dict[tuple[str, str], dict[str, object]] = {}
    monkeypatch.setattr(
        S3Credentials, "connect", lambda self: _FakeConnection(objects)
    )
    monkeypatch.chdir(root.path)
    return root.path


def test_push_exists_pull_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    target = root / "storage" / "run_artifacts" / "exp" / "ckpt.bin"
    target.parent.mkdir(parents=True)
    target.write_text("weights-v1", encoding="utf-8")

    assert main(["remote-push", "storage/run_artifacts/exp/ckpt.bin"]) == 0
    identity = capsys.readouterr().out.strip()
    assert len(identity) == 64

    assert (
        main(["remote-exists", "storage/run_artifacts/exp/ckpt.bin", "--hash", identity])
        == 0
    )
    assert capsys.readouterr().out.strip() == "present"

    target.unlink()
    assert (
        main(["remote-pull", "storage/run_artifacts/exp/ckpt.bin", "--hash", identity])
        == 0
    )
    capsys.readouterr()
    assert target.read_text(encoding="utf-8") == "weights-v1"


def test_exists_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _project(tmp_path, monkeypatch)
    code = main(["remote-exists", "storage/nope.bin", "--hash", "0" * 64])
    assert code == 0
    assert capsys.readouterr().out.strip() == "absent"


def test_pull_hash_mismatch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    target = root / "storage" / "a.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("good", encoding="utf-8")
    main(["remote-push", "storage/a.bin"])
    capsys.readouterr()
    target.unlink()
    wrong = "0" * 64
    assert main(["remote-pull", "storage/a.bin", "--hash", wrong]) == 1
    assert "hashes to" in capsys.readouterr().err


def test_push_directory_requires_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    folder = root / "storage" / "bundle"
    folder.mkdir(parents=True)
    (folder / "part.bin").write_text("x", encoding="utf-8")
    assert main(["remote-push", "storage/bundle"]) == 1
    assert "--hash" in capsys.readouterr().err


def test_push_rejects_path_outside_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    outside = root / "notes.txt"
    outside.write_text("x", encoding="utf-8")
    assert main(["remote-push", "notes.txt"]) == 1
    assert "not under storage" in capsys.readouterr().err
