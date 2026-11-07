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
from dl_toolbox.base.input_output.json_io import JsonTextForm, read, write
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.names import (
    DEFAULT_CHECKPOINT_CONFIG,
    DEFAULT_IDENTITY_CONFIG,
    MARKER_FILENAME,
    TRACKING_BACKEND_FILENAME,
)
from tests.base.base_support import N_TRIALS, random_generating_process_name


def _existing_root(path: Path) -> Directory:
    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.CREATE)


def _scaffold(
    tmp_path: Path, *, name: str, backend: str
) -> Directory:
    root = _existing_root(tmp_path / name)
    ProjectLayout.scaffold(
        root=root, project_name=name, tracking_backend=backend
    )
    return root


def test_scaffold_then_from_start_tensorboard(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        name = random_generating_process_name(rng)
        root = _scaffold(
            tmp_path / f"tb_{i}",
            name=name,
            backend=TrackingBackend.TENSORBOARD,
        )
        nested = Directory.from_under(
            root=root,
            relative_path="scripts/nested",
            policy=DirectoryParsePolicy.CREATE,
        )
        train_path = nested.path / "train.py"
        train_path.write_text("x = 1\n", encoding="utf-8")
        start = File.from_absolute(path=train_path)
        layout = ProjectLayout.from_start(
            start=start,
            project_name=name,
            tracking_backend=TrackingBackend.TENSORBOARD,
        )
        assert layout.root.path == root.path
        assert layout.project_name == name
        assert layout.tracking_backend == TrackingBackend.TENSORBOARD
        assert layout.hash_manager.prefix_len == DEFAULT_IDENTITY_CONFIG["prefix_len"]
        assert layout.lock_timeout_seconds == DEFAULT_CHECKPOINT_CONFIG[
            "lock_timeout_seconds"
        ]
        assert layout.tensorboard_log_dir().path == root.path / "storage" / "tensorboard"
        assert layout.tensorboard_log_dir().spec == DirectorySpec.MISSING
        index = layout.identity_index(run_group=())
        assert index.path == root.path / "storage" / "runs"
        assert index.spec == DirectorySpec.MISSING
        with pytest.raises(ValueError, match="mlflow"):
            layout.mlflow_db_path()


def test_from_start_does_not_use_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path / "repo", name=name, backend=TrackingBackend.TENSORBOARD
    )
    start = File.from_absolute(path=root.path / MARKER_FILENAME)
    layout = ProjectLayout.from_start(
        start=start,
        project_name=name,
        tracking_backend=TrackingBackend.TENSORBOARD,
    )
    assert layout.root.path == root.path
    with pytest.raises(FileNotFoundError):
        ProjectLayout.from_start(
            start=Directory.from_absolute(
                path=other, policy=DirectoryParsePolicy.MUST_EXIST
            ),
            project_name=name,
            tracking_backend=TrackingBackend.TENSORBOARD,
        )


def test_from_start_mlflow_and_does_not_rewrite_tracking_file(
    tmp_path: Path, rng: random.Random
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    before = (root.path / TRACKING_BACKEND_FILENAME).read_text(encoding="utf-8")
    layout = ProjectLayout.from_start(
        start=root,
        project_name=name,
        tracking_backend=TrackingBackend.MLFLOW,
    )
    assert layout.tracking_backend == TrackingBackend.MLFLOW
    assert layout.mlflow_db_path() == (
        root.path / "storage" / "mlflow" / "tracking.db"
    )
    assert layout.mlflow_timeout_seconds() == 30.0
    assert layout.mlflow_artifact_root().path == (
        root.path / "storage" / "mlflow" / "artifacts"
    )
    assert (root.path / TRACKING_BACKEND_FILENAME).read_text(
        encoding="utf-8"
    ) == before
    with pytest.raises(ValueError, match="tensorboard"):
        layout.tensorboard_log_dir()


def test_scaffold_does_not_write_wandb(tmp_path: Path, rng: random.Random) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    logging_dir = root.path / "project_level_configs" / "logging"
    assert (logging_dir / "tensorboard.json").is_file()
    assert (logging_dir / "mlflow.json").is_file()
    assert not (logging_dir / "wandb.json").exists()
    assert not (root.path / "storage" / "runs").exists()
    marker = read(file=File.from_under(root=root, relative_path=MARKER_FILENAME))
    assert marker == {"name": name}


def test_marker_name_mismatch(tmp_path: Path, rng: random.Random) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    with pytest.raises(ValueError, match="does not match"):
        ProjectLayout.from_start(
            start=root,
            project_name=name + "_other",
            tracking_backend=TrackingBackend.TENSORBOARD,
        )


def test_marker_rejects_package_key(tmp_path: Path, rng: random.Random) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    write(
        directory=root,
        name=MARKER_FILENAME,
        payload={"name": name, "package": name},
        form=JsonTextForm.DOCUMENT,
    )
    with pytest.raises(ValueError, match=MARKER_FILENAME):
        ProjectLayout.from_start(
            start=root,
            project_name=name,
            tracking_backend=TrackingBackend.TENSORBOARD,
        )


def test_absolute_log_dir_stays_absolute(
    tmp_path: Path, rng: random.Random
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path / "proj", name=name, backend=TrackingBackend.TENSORBOARD
    )
    absolute = tmp_path / "outside_tb"
    write(
        directory=Directory.from_under(
            root=root,
            relative_path="project_level_configs/logging",
            policy=DirectoryParsePolicy.MUST_EXIST,
        ),
        name="tensorboard.json",
        payload={"log_dir": str(absolute)},
        form=JsonTextForm.DOCUMENT,
    )
    layout = ProjectLayout.from_start(
        start=root,
        project_name=name,
        tracking_backend=TrackingBackend.TENSORBOARD,
    )
    assert layout.tensorboard_log_dir().path == absolute


def test_run_group_joins_identity_and_artifacts(
    tmp_path: Path, rng: random.Random
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    layout = ProjectLayout.from_start(
        start=root,
        project_name=name,
        tracking_backend=TrackingBackend.TENSORBOARD,
    )
    group = ProjectLayout.parse_run_group(
        ["learning_rate_grid", "less_than10"]
    )
    assert group == ("learning_rate_grid", "less_than10")
    index = layout.identity_index(run_group=group)
    assert index.path == (
        root.path / "storage" / "runs" / "learning_rate_grid" / "less_than10"
    )
    assert index.spec == DirectorySpec.MISSING
    run_name = random_generating_process_name(rng)
    artifacts = layout.artifacts_directory(run_group=group, name=run_name)
    assert artifacts.path == (
        root.path
        / "storage"
        / "run_artifacts"
        / "learning_rate_grid"
        / "less_than10"
        / run_name
    )
    flat = layout.artifacts_directory(run_group=(), name=run_name)
    assert flat.path == root.path / "storage" / "run_artifacts" / run_name


def test_parse_run_group_rejects_dotdot() -> None:
    with pytest.raises(ValueError, match="basename"):
        ProjectLayout.parse_run_group(["learning_rate_grid", ".."])
    with pytest.raises(ValueError, match="basename"):
        ProjectLayout.parse_run_group(["a/b"])
    with pytest.raises(TypeError, match="sequence"):
        ProjectLayout.parse_run_group("learning_rate_grid")


def test_unknown_backend(tmp_path: Path, rng: random.Random) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    with pytest.raises(ValueError, match="tracking_backend"):
        ProjectLayout.from_start(
            start=root, project_name=name, tracking_backend="wandb"
        )


def test_scaffold_refuses_existing_marker(
    tmp_path: Path, rng: random.Random
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    with pytest.raises(FileExistsError, match="marker already exists"):
        ProjectLayout.scaffold(
            root=root,
            project_name=name,
            tracking_backend=TrackingBackend.TENSORBOARD,
        )
