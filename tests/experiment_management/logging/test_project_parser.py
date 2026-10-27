from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.experiment_management.logging.project_parser import (
    MlflowProjectLoggerConfigParser,
    ProjectLoggerParseError,
    TensorBoardProjectLoggerConfigParser,
    WandbProjectLoggerConfigParser,
    parse_project_logger_config,
)


def test_tensorboard_project_parser_resolves_under_root(tmp_path: Path) -> None:
    parsed = TensorBoardProjectLoggerConfigParser(
        {"log_dir": "storage/tensorboard"},
        project_root=tmp_path,
    )
    assert parsed.log_dir == tmp_path / "storage" / "tensorboard"


def test_wandb_project_parser_rejects_extra_keys(tmp_path: Path) -> None:
    with pytest.raises(ProjectLoggerParseError, match="keys must be"):
        WandbProjectLoggerConfigParser(
            {"dir": "storage/wandb", "project": "x"},
            project_root=tmp_path,
        )


def test_mlflow_project_parser_exposes_typed_fields(tmp_path: Path) -> None:
    parsed = MlflowProjectLoggerConfigParser(
        {
            "backend_store": {
                "type": "database",
                "database_type": "sqlite",
                "path": "storage/mlflow/tracking.db",
                "timeout_seconds": 30,
            },
            "artifact_store": {
                "type": "filesystem",
                "root": "storage/mlflow/artifacts",
            },
        },
        project_root=tmp_path,
    )
    assert parsed.db_path == tmp_path / "storage" / "mlflow" / "tracking.db"
    assert parsed.artifact_root == tmp_path / "storage" / "mlflow" / "artifacts"
    assert parsed.timeout_seconds == 30.0


def test_mlflow_postgres_is_not_wired(tmp_path: Path) -> None:
    with pytest.raises(ProjectLoggerParseError, match="not wired"):
        MlflowProjectLoggerConfigParser(
            {
                "backend_store": {
                    "type": "database",
                    "database_type": "postgres",
                    "path": "storage/mlflow/tracking.db",
                    "timeout_seconds": 30,
                },
                "artifact_store": {
                    "type": "filesystem",
                    "root": "storage/mlflow/artifacts",
                },
            },
            project_root=tmp_path,
        )


def test_dispatch_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown tracking backend"):
        parse_project_logger_config("not-a-backend", {}, project_root=tmp_path)
