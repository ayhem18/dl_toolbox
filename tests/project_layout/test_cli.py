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
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.cli import main
from dl_toolbox.project_layout.names import (
    MARKER_FILENAME,
    MLFLOW_RUN_FILENAME,
    TENSORBOARD_RUN_FILENAME,
    TRACKING_BACKEND_FILENAME,
)
from tests.base.base_support import N_TRIALS, random_generating_process_name


def _existing_project(path: Path) -> Directory:
    return Directory.from_absolute(
        path=path, policy=DirectoryParsePolicy.MUST_EXIST
    )


def test_cli_setup_project_uses_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    for i in range(N_TRIALS):
        cwd = tmp_path / f"cwd_{i}"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        name = random_generating_process_name(rng)
        code = main(
            [
                "setup-project",
                name,
                "--tracking-backend",
                TrackingBackend.TENSORBOARD,
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        root_path = cwd / name
        assert str(root_path.resolve()) in captured.out
        root = _existing_project(root_path)
        marker = read_object(
            file=File.from_under(root=root, relative_path=MARKER_FILENAME)
        )
        assert marker == {"name": name}
        layout = ProjectLayout.from_start(
            start=root,
            project_name=name,
            tracking_backend=TrackingBackend.TENSORBOARD,
        )
        assert layout.root.path == root_path.resolve()
        assert layout.tracking_backend == TrackingBackend.TENSORBOARD
        assert (
            layout.hash_manager.lock_timeout_seconds
            == layout.lock_timeout_seconds
        )
        assert layout.hash_manager.lock_poll_seconds == ProjectLayout.LOCK_POLL_SECONDS
        File.from_under(
            root=root, relative_path=f"scripts/{TENSORBOARD_RUN_FILENAME}"
        )
        with pytest.raises(FileNotFoundError):
            File.from_under(
                root=root, relative_path=f"scripts/{MLFLOW_RUN_FILENAME}"
            )
        again = main(
            [
                "setup-project",
                name,
                "--tracking-backend",
                TrackingBackend.TENSORBOARD,
            ]
        )
        assert again == 1
        err = capsys.readouterr().err
        assert "already exists" in err


def test_cli_setup_project_tracking_backend_mlflow(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    name = random_generating_process_name(rng)
    code = main(
        ["setup-project", name, "--tracking-backend", TrackingBackend.MLFLOW]
    )
    assert code == 0
    capsys.readouterr()
    root = _existing_project(tmp_path / name)
    backend_file = File.from_under(root=root, relative_path=TRACKING_BACKEND_FILENAME)
    assert backend_file.path.read_text(encoding="utf-8") == (
        TrackingBackend.MLFLOW + "\n"
    )
    logging_dir = Directory.from_under(
        root=root,
        relative_path="project_level_configs/logging",
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    File.from_under(root=logging_dir, relative_path="mlflow.json")
    File.from_under(root=logging_dir, relative_path="tensorboard.json")
    with pytest.raises(FileNotFoundError):
        File.from_under(root=logging_dir, relative_path="wandb.json")
    runs = Directory.from_under(
        root=root,
        relative_path="storage/runs",
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )
    assert runs.spec == DirectorySpec.MISSING
    for dirname in ("storage", "scripts", "experiments", "run_recipes"):
        Directory.from_under(
            root=root,
            relative_path=dirname,
            policy=DirectoryParsePolicy.MUST_EXIST,
        )
    layout = ProjectLayout.from_start(
        start=root,
        project_name=name,
        tracking_backend=TrackingBackend.MLFLOW,
    )
    assert layout.mlflow_db_path() == root.path / "storage" / "mlflow" / "tracking.db"
    assert layout.mlflow_artifact_root().spec == DirectorySpec.MISSING
    File.from_under(root=root, relative_path=f"scripts/{MLFLOW_RUN_FILENAME}")
    with pytest.raises(FileNotFoundError):
        File.from_under(
            root=root, relative_path=f"scripts/{TENSORBOARD_RUN_FILENAME}"
        )


def test_cli_requires_tracking_backend(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    name = random_generating_process_name(rng)
    with pytest.raises(SystemExit) as raised:
        main(["setup-project", name])
    assert raised.value.code == 2
    missing = Directory.from_absolute(
        path=tmp_path / name, policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    assert missing.spec == DirectorySpec.MISSING


def test_cli_rejects_package_and_wandb(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    name = random_generating_process_name(rng)
    with pytest.raises(SystemExit) as package_exit:
        main(
            [
                "setup-project",
                name,
                "--package",
                name,
                "--tracking-backend",
                TrackingBackend.MLFLOW,
            ]
        )
    assert package_exit.value.code == 2
    with pytest.raises(SystemExit) as wandb_exit:
        main(["setup-project", name, "--tracking-backend", "wandb"])
    assert wandb_exit.value.code == 2
    missing = Directory.from_absolute(
        path=tmp_path / name, policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    assert missing.spec == DirectorySpec.MISSING


def test_cli_rejects_nested_project_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "setup-project",
            "nested/name",
            "--tracking-backend",
            TrackingBackend.MLFLOW,
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "basename" in err
    missing = Directory.from_absolute(
        path=tmp_path / "nested", policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    assert missing.spec == DirectorySpec.MISSING


def test_cli_removes_directory_if_scaffold_fails(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    name = random_generating_process_name(rng)

    def fake_scaffold(*, root: Directory, project_name: str, tracking_backend: str) -> None:
        del project_name, tracking_backend
        (root.path / "partial.txt").write_text("x\n", encoding="utf-8")
        raise ValueError("scaffold exploded")

    monkeypatch.setattr(ProjectLayout, "scaffold", fake_scaffold)
    code = main(
        ["setup-project", name, "--tracking-backend", TrackingBackend.MLFLOW]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "scaffold exploded" in err
    missing = Directory.from_absolute(
        path=tmp_path / name, policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    assert missing.spec == DirectorySpec.MISSING
