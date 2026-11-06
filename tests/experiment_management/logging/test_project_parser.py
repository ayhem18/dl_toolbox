# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from pathlib import Path

import pytest

from dl_toolbox.experiment_management.default_layout.parsing.logging import (
    MlflowProjectLoggerConfigParser,
    ProjectLoggerParseError,
    TensorBoardProjectLoggerConfigParser,
    WandbProjectLoggerConfigParser,
    parse_project_logger_config,
    tensorboard_project_from_pydantic,
)


def test_tensorboard_project_parser_resolves_under_root(tmp_path: Path) -> None:
    parsed = TensorBoardProjectLoggerConfigParser(
        {"log_dir": "storage/tensorboard"},
        project_root=tmp_path,
    )
    assert parsed.log_dir == tmp_path / "storage" / "tensorboard"


def test_tensorboard_pydantic_matches_parser(tmp_path: Path) -> None:
    payload = {"log_dir": "storage/tensorboard"}
    parsed = tensorboard_project_from_pydantic(payload, project_root=tmp_path)
    original = TensorBoardProjectLoggerConfigParser(payload, project_root=tmp_path)
    assert parsed.log_dir == original.log_dir
    with pytest.raises(ProjectLoggerParseError):
        tensorboard_project_from_pydantic(
            {"log_dir": "storage/tensorboard", "extra": 1},
            project_root=tmp_path,
        )


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
