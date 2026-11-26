"""ProjectLayout <-> remote store wiring: scaffold writes remote/ config and a
blank per-backend credential file (git-ignored), and remote_mirror builds the
mirror from them. connect() is monkeypatched so these run without boto3.
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
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.documents import S3RemoteDocument
from dl_toolbox.project_layout.names import (
    DEFAULT_S3_ENDPOINT_URL,
    DEFAULT_S3_REGION,
    GITIGNORE_FILENAME,
    MARKER_FILENAME,
)
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.connection import RemoteStoreConnection
from dl_toolbox.remote_store.credentials import S3Credentials, blank_credential_document
from dl_toolbox.remote_store.project_mirror import ProjectRemoteMirror

_REMOTE_DIR = "project_level_configs/remote"
_CREDENTIALS_DIR = f"{_REMOTE_DIR}/credentials"


class _FakeConnection(RemoteStoreConnection):
    @property
    def backend(self) -> str:
        return RemoteBackend.S3

    def object_exists(self, *, identity: str, location: Mapping) -> bool:
        return False

    def push_file(self, *, identity, location, source) -> None:  # noqa: ANN001
        return None

    def pull_file(self, *, identity, location, destination, name):  # noqa: ANN001
        raise NotImplementedError

    def push_directory(self, *, identity, location, source) -> None:  # noqa: ANN001
        return None

    def pull_directory(self, *, identity, location, destination) -> None:  # noqa: ANN001
        return None


def _scaffold(tmp_path: Path, *, name: str, remote_backend: str) -> Directory:
    root = Directory.from_absolute(
        path=tmp_path / name, policy=DirectoryParsePolicy.CREATE
    )
    ProjectLayout.scaffold(
        root=root,
        project_name=name,
        tracking_backend=TrackingBackend.TENSORBOARD,
        remote_backend=remote_backend,
    )
    return root


def _layout(root: Directory, *, name: str) -> ProjectLayout:
    marker = File.from_under(root=root, relative_path=MARKER_FILENAME)
    return ProjectLayout.from_start(
        start=marker,
        project_name=name,
        tracking_backend=TrackingBackend.TENSORBOARD,
    )


def _write_credentials(root: Directory, *, payload: dict) -> None:
    credentials_dir = Directory.from_under(
        root=root,
        relative_path=_CREDENTIALS_DIR,
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    write_object(
        directory=credentials_dir,
        name=f"{RemoteBackend.S3}.json",
        payload=payload,
        form=JsonTextForm.DOCUMENT,
    )


# --- scaffold -----------------------------------------------------------------


def test_scaffold_writes_remote_config_and_blank_credentials(tmp_path: Path) -> None:
    root = _scaffold(tmp_path, name="proj", remote_backend="s3")

    remote = read_object(
        file=File.from_under(root=root, relative_path=f"{_REMOTE_DIR}/remote.json")
    )
    document = S3RemoteDocument.model_validate(remote)
    assert document.backend == RemoteBackend.S3
    assert document.bucket == "proj"
    assert document.endpoint_url == DEFAULT_S3_ENDPOINT_URL
    assert document.region == DEFAULT_S3_REGION

    for backend in RemoteBackend.BACKENDS:
        creds = read_object(
            file=File.from_under(
                root=root, relative_path=f"{_CREDENTIALS_DIR}/{backend}.json"
            )
        )
        assert creds == blank_credential_document(backend)
        assert all(value == "" for value in creds.values())


def test_scaffold_gitignores_credentials(tmp_path: Path) -> None:
    root = _scaffold(tmp_path, name="proj", remote_backend="s3")
    gitignore = (root.path / GITIGNORE_FILENAME).read_text(encoding="utf-8")
    assert f"{_CREDENTIALS_DIR}/" in gitignore


def test_blank_credentials_keys_per_backend() -> None:
    assert set(blank_credential_document("s3")) == {
        "access_key_id",
        "secret_access_key",
    }
    assert set(blank_credential_document("huggingface")) == {"token"}


# --- remote_mirror ------------------------------------------------------------


def test_remote_mirror_builds_from_config_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _scaffold(tmp_path, name="proj", remote_backend="s3")
    _write_credentials(
        root, payload={"access_key_id": "AKIA", "secret_access_key": "secret"}
    )
    captured: dict[str, S3Credentials] = {}

    def fake_connect(self: S3Credentials) -> RemoteStoreConnection:
        captured["credentials"] = self
        return _FakeConnection()

    monkeypatch.setattr(S3Credentials, "connect", fake_connect)

    mirror = _layout(root, name="proj").remote_mirror()
    assert isinstance(mirror, ProjectRemoteMirror)
    assert mirror.bucket == "proj"
    assert isinstance(mirror.connection, _FakeConnection)

    credentials = captured["credentials"]
    assert credentials.endpoint_url == DEFAULT_S3_ENDPOINT_URL
    assert credentials.region == DEFAULT_S3_REGION
    assert credentials.access_key_id.get_secret_value() == "AKIA"
    assert credentials.secret_access_key.get_secret_value() == "secret"


def test_remote_mirror_blank_credentials_errors(tmp_path: Path) -> None:
    root = _scaffold(tmp_path, name="proj", remote_backend="s3")
    with pytest.raises(ValueError, match="incomplete"):
        _layout(root, name="proj").remote_mirror()


def test_remote_mirror_rejects_unwired_backend(tmp_path: Path) -> None:
    root = _scaffold(tmp_path, name="proj", remote_backend="huggingface")
    with pytest.raises(ValueError, match="not wired"):
        _layout(root, name="proj").remote_mirror()
