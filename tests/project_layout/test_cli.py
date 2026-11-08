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
from dl_toolbox.base.input_output.json_io import read
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.cli import main
from dl_toolbox.project_layout.names import MARKER_FILENAME, TRACKING_BACKEND_FILENAME
from tests.base.base_support import N_TRIALS, random_generating_process_name


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
        root = cwd / name
        assert str(root.resolve()) in captured.out
        assert (root / MARKER_FILENAME).is_file()
        marker = read(
            file=File.from_under(
                root=Directory.from_absolute(
                    path=root, policy=DirectoryParsePolicy.MUST_EXIST
                ),
                relative_path=MARKER_FILENAME,
            )
        )
        assert marker == {"name": name}
        layout = ProjectLayout.from_start(
            start=Directory.from_absolute(
                path=root, policy=DirectoryParsePolicy.MUST_EXIST
            ),
            project_name=name,
            tracking_backend=TrackingBackend.TENSORBOARD,
        )
        assert layout.root.path == root.resolve()
        assert layout.tracking_backend == TrackingBackend.TENSORBOARD
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
    root = tmp_path / name
    assert (root / TRACKING_BACKEND_FILENAME).read_text(encoding="utf-8") == (
        TrackingBackend.MLFLOW + "\n"
    )
    logging_dir = root / "project_level_configs" / "logging"
    assert (logging_dir / "mlflow.json").is_file()
    assert (logging_dir / "tensorboard.json").is_file()
    assert not (logging_dir / "wandb.json").exists()
    assert not (root / "storage" / "runs").exists()
    assert (root / "storage").is_dir()
    assert (root / "scripts").is_dir()
    assert (root / "experiments").is_dir()
    assert (root / "run_recipes").is_dir()
    layout = ProjectLayout.from_start(
        start=Directory.from_absolute(
            path=root, policy=DirectoryParsePolicy.MUST_EXIST
        ),
        project_name=name,
        tracking_backend=TrackingBackend.MLFLOW,
    )
    assert layout.mlflow_db_path() == root / "storage" / "mlflow" / "tracking.db"
    assert layout.mlflow_artifact_root().spec == DirectorySpec.MISSING


def test_cli_requires_tracking_backend(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    name = random_generating_process_name(rng)
    with pytest.raises(SystemExit) as raised:
        main(["setup-project", name])
    assert raised.value.code == 2
    assert not (tmp_path / name).exists()


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
    assert not (tmp_path / name).exists()


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
    assert not (tmp_path / "nested").exists()


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
    assert not (tmp_path / name).exists()
