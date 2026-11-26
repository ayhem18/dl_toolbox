"""remote-push / remote-pull / remote-exists CLI, end to end over one shared
in-memory connection (S3Credentials.connect monkeypatched). No boto3, no server,
no hashes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import Directory, DirectoryParsePolicy
from dl_toolbox.base.input_output.json_io import JsonTextForm, write_object
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.cli import main
from dl_toolbox.remote_store.backends import RemoteBackend
from dl_toolbox.remote_store.credentials import S3Credentials
from tests.remote_store.support import InMemoryConnection

_CREDENTIALS_DIR = "project_level_configs/remote/credentials"


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
        name=f"{RemoteBackend.S3}.json",
        payload={"access_key_id": "AKIA", "secret_access_key": "secret"},
        form=JsonTextForm.DOCUMENT,
    )
    connection = InMemoryConnection()
    monkeypatch.setattr(S3Credentials, "connect", lambda self: connection)
    monkeypatch.chdir(root.path)
    return root.path


def test_push_exists_pull_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    target = root / "storage" / "run_artifacts" / "exp" / "ckpt.bin"
    target.parent.mkdir(parents=True)
    target.write_text("weights-v1", encoding="utf-8")

    assert main(["remote-push", "storage/run_artifacts/exp/ckpt.bin"]) == 0
    assert capsys.readouterr().out.strip() == "run_artifacts/exp/ckpt.bin"

    assert main(["remote-exists", "storage/run_artifacts/exp/ckpt.bin"]) == 0
    assert capsys.readouterr().out.strip() == "present"

    target.unlink()
    assert main(["remote-pull", "storage/run_artifacts/exp/ckpt.bin"]) == 0
    capsys.readouterr()
    assert target.read_text(encoding="utf-8") == "weights-v1"


def test_exists_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _project(tmp_path, monkeypatch)
    assert main(["remote-exists", "storage/nope.bin"]) == 0
    assert capsys.readouterr().out.strip() == "absent"


def test_push_pull_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    bundle = root / "storage" / "run_artifacts" / "pretrain"
    (bundle / "sub").mkdir(parents=True)
    (bundle / "model.bin").write_text("m", encoding="utf-8")
    (bundle / "sub" / "opt.bin").write_text("o", encoding="utf-8")

    assert main(["remote-push", "storage/run_artifacts/pretrain"]) == 0
    capsys.readouterr()

    (bundle / "model.bin").unlink()
    (bundle / "sub" / "opt.bin").unlink()
    assert main(["remote-pull", "storage/run_artifacts/pretrain", "--dir"]) == 0
    capsys.readouterr()
    assert (bundle / "model.bin").read_text(encoding="utf-8") == "m"
    assert (bundle / "sub" / "opt.bin").read_text(encoding="utf-8") == "o"


def test_push_rejects_path_outside_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _project(tmp_path, monkeypatch)
    outside = root / "notes.txt"
    outside.write_text("x", encoding="utf-8")
    assert main(["remote-push", "notes.txt"]) == 1
    assert "not under storage" in capsys.readouterr().err
