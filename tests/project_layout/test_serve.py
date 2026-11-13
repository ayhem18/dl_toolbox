from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import JsonTextForm, write_object
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from dl_toolbox.project_layout.names import (
    MLFLOW_RUN_FILENAME,
    SQLITE_TIMEOUT_SECONDS,
    TENSORBOARD_RUN_FILENAME,
)
from dl_toolbox.project_layout.serve import (
    MLFLOW_UI_HOST,
    MLFLOW_UI_PORT,
    TENSORBOARD_UI_HOST,
    TENSORBOARD_UI_PORT,
    run_mlflow,
    run_tensorboard,
)
from tests.base.base_support import random_generating_process_name


def _existing_root(path: Path) -> Directory:
    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.CREATE)


def _scaffold(tmp_path: Path, *, name: str, backend: str) -> Directory:
    root = _existing_root(tmp_path / name)
    ProjectLayout.scaffold(
        root=root, project_name=name, tracking_backend=backend
    )
    return root


def _logging_dir(root: Directory) -> Directory:
    return Directory.from_under(
        root=root,
        relative_path="project_level_configs/logging",
        policy=DirectoryParsePolicy.MUST_EXIST,
    )


def _capture_call(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_call(cmd, *, cwd, env=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["env"] = env
        return 0

    monkeypatch.setattr(
        "dl_toolbox.project_layout.serve.subprocess.call", fake_call
    )
    return captured


def test_run_mlflow_uses_layout_store_paths(
    tmp_path: Path,
    rng: random.Random,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(tmp_path, name=name, backend=TrackingBackend.MLFLOW)
    captured = _capture_call(monkeypatch)
    script = File.from_under(
        root=root, relative_path=f"scripts/{MLFLOW_RUN_FILENAME}"
    )
    code = run_mlflow(
        start=script, argv=["--host", "0.0.0.0", "--port", "5001"]
    )
    assert code == 0
    db = (root.path / "storage" / "mlflow" / "tracking.db").resolve()
    artifacts = (root.path / "storage" / "mlflow" / "artifacts").resolve()
    cmd = captured["cmd"]
    assert cmd[cmd.index("--backend-store-uri") + 1] == (
        f"sqlite:///{db.as_posix()}"
    )
    assert cmd[cmd.index("--default-artifact-root") + 1] == str(artifacts)
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--port") + 1] == "5001"
    assert captured["cwd"] == str(root.path)
    assert captured["env"]["MLFLOW_DISABLE_AGENT_HINT"] == "1"
    assert artifacts.is_dir()
    assert db.parent.is_dir()
    assert "http://0.0.0.0:5001/" in capsys.readouterr().out


def test_run_mlflow_reads_logging_json(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(tmp_path, name=name, backend=TrackingBackend.MLFLOW)
    absolute_db = tmp_path / "outside_store" / "custom.db"
    absolute_art = tmp_path / "outside_artifacts"
    write_object(
        directory=_logging_dir(root),
        name="mlflow.json",
        payload={
            "backend_store": {
                "type": "database",
                "database_type": "sqlite",
                "path": str(absolute_db),
                "timeout_seconds": SQLITE_TIMEOUT_SECONDS,
            },
            "artifact_store": {
                "type": "filesystem",
                "root": str(absolute_art),
            },
        },
        form=JsonTextForm.DOCUMENT,
    )
    captured = _capture_call(monkeypatch)
    script = File.from_under(
        root=root, relative_path=f"scripts/{MLFLOW_RUN_FILENAME}"
    )
    code = run_mlflow(start=script, argv=[])
    assert code == 0
    cmd = captured["cmd"]
    assert cmd[cmd.index("--backend-store-uri") + 1] == (
        f"sqlite:///{absolute_db.resolve().as_posix()}"
    )
    assert cmd[cmd.index("--default-artifact-root") + 1] == str(
        absolute_art.resolve()
    )
    assert cmd[cmd.index("--host") + 1] == MLFLOW_UI_HOST
    assert cmd[cmd.index("--port") + 1] == str(MLFLOW_UI_PORT)
    assert absolute_art.is_dir()
    assert absolute_db.parent.is_dir()


def test_run_mlflow_rejects_non_positive_port(
    tmp_path: Path, rng: random.Random
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(tmp_path, name=name, backend=TrackingBackend.MLFLOW)
    script = File.from_under(
        root=root, relative_path=f"scripts/{MLFLOW_RUN_FILENAME}"
    )
    with pytest.raises(ValueError, match="port"):
        run_mlflow(start=script, argv=["--port", "0"])


def test_run_tensorboard_uses_layout_log_dir(
    tmp_path: Path,
    rng: random.Random,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(tmp_path, name=name, backend=TrackingBackend.TENSORBOARD)
    captured = _capture_call(monkeypatch)
    script = File.from_under(
        root=root, relative_path=f"scripts/{TENSORBOARD_RUN_FILENAME}"
    )
    code = run_tensorboard(
        start=script, argv=["--host", "127.0.0.1", "--port", "6007"]
    )
    assert code == 0
    log_dir = (root.path / "storage" / "tensorboard").resolve()
    cmd = captured["cmd"]
    assert cmd[cmd.index("--logdir") + 1] == str(log_dir)
    assert cmd[cmd.index("--reload_multifile") + 1] == "true"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "6007"
    assert captured["cwd"] == str(root.path)
    assert log_dir.is_dir()
    assert "http://127.0.0.1:6007/" in capsys.readouterr().out


def test_run_tensorboard_reads_logging_json(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = random_generating_process_name(rng)
    root = _scaffold(
        tmp_path, name=name, backend=TrackingBackend.TENSORBOARD
    )
    absolute = tmp_path / "outside_tb"
    write_object(
        directory=_logging_dir(root),
        name="tensorboard.json",
        payload={"log_dir": str(absolute)},
        form=JsonTextForm.DOCUMENT,
    )
    captured = _capture_call(monkeypatch)
    script = File.from_under(
        root=root, relative_path=f"scripts/{TENSORBOARD_RUN_FILENAME}"
    )
    code = run_tensorboard(start=script, argv=[])
    assert code == 0
    cmd = captured["cmd"]
    assert cmd[cmd.index("--logdir") + 1] == str(absolute.resolve())
    assert cmd[cmd.index("--host") + 1] == TENSORBOARD_UI_HOST
    assert cmd[cmd.index("--port") + 1] == str(TENSORBOARD_UI_PORT)
    assert absolute.is_dir()
