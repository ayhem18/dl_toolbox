"""ProjectLayout <-> remote store wiring: scaffold writes remote/ config and a
blank per-backend credential file (git-ignored), and remote_mirror builds the
mirror from them. S3Credentials.connect is monkeypatched, so these run without
boto3.
"""

from __future__ import annotations

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
    default_bucket_name,
)
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.credentials import S3Credentials, blank_credential_document
from dl_toolbox.remote_store.project_mirror import ProjectRemoteMirror
from tests.remote_store.support import InMemoryConnection

_CREDENTIALS_DIR = "project_level_configs/remote/credentials"


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


def _fill_credentials(root: Directory) -> None:
    credentials_dir = Directory.from_under(
        root=root,
        relative_path=_CREDENTIALS_DIR,
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    write_object(
        directory=credentials_dir,
        name=f"{RemoteBackend.S3}.json",
        payload={"access_key_id": "AKIA", "secret_access_key": "secret"},
        form=JsonTextForm.DOCUMENT,
    )


# --- scaffold -----------------------------------------------------------------


def test_scaffold_writes_remote_config_and_blank_credentials(tmp_path: Path) -> None:
    root = _scaffold(tmp_path, name="proj", remote_backend="s3")

    remote = read_object(
        file=File.from_under(
            root=root, relative_path="project_level_configs/remote/remote.json"
        )
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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ccf-tabformer", "ccf-tabformer"),
        ("ccf_tabformer", "ccf-tabformer"),
        ("CCF_Tabformer", "ccf-tabformer"),
        ("_tmp", ""),
        ("my.project", ""),
        ("ab", ""),
    ],
)
def test_default_bucket_name(name: str, expected: str) -> None:
    assert default_bucket_name(name) == expected


def test_scaffold_bucket_sanitizes_underscores(tmp_path: Path) -> None:
    root = _scaffold(tmp_path, name="ccf_tabformer", remote_backend="s3")
    remote = read_object(
        file=File.from_under(
            root=root, relative_path="project_level_configs/remote/remote.json"
        )
    )
    assert remote["bucket"] == "ccf-tabformer"


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
    _fill_credentials(root)
    captured: dict[str, S3Credentials] = {}

    def fake_connect(self: S3Credentials) -> InMemoryConnection:
        captured["credentials"] = self
        return InMemoryConnection()

    monkeypatch.setattr(S3Credentials, "connect", fake_connect)

    mirror = _layout(root, name="proj").remote_mirror()
    assert isinstance(mirror, ProjectRemoteMirror)
    assert mirror.bucket == "proj"
    assert isinstance(mirror.connection, InMemoryConnection)

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
