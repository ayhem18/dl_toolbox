from pathlib import Path

import pytest

from dl_toolbox.loggers.mlflow.default_configurations import (
    default_filesystem_artifact_store,
    default_mlflow_setup,
    default_sqlite_backend_store,
)
from dl_toolbox.loggers.mlflow.mlflow_parser import MlflowParseError, MlflowParser


def _payload(tmp_path: Path) -> dict:
    return default_mlflow_setup(
        backend_store=default_sqlite_backend_store(
            path=tmp_path / "mlflow" / "mlflow.sqlite"
        ),
        artifact_store=default_filesystem_artifact_store(
            root=tmp_path / "mlflow" / "artifacts"
        ),
    )


def test_parser_exposes_persistence_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "mlflow" / "mlflow.sqlite"
    artifact_root = tmp_path / "mlflow" / "artifacts"
    parsed = MlflowParser(_payload(tmp_path))
    assert parsed.db_path == db_path
    assert parsed.artifact_root == artifact_root
    assert parsed.backend_store_uri.startswith("sqlite:///")
    assert parsed.backend_store_uri.endswith("timeout=30")
    assert parsed.artifact_store_uri == artifact_root.as_posix()


def test_relative_sqlite_path_is_an_error(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["backend_store"]["path"] = "mlflow.sqlite"
    with pytest.raises(MlflowParseError, match="absolute"):
        MlflowParser(payload)


def test_relative_artifact_root_is_an_error(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["artifact_store"]["root"] = "artifacts"
    with pytest.raises(MlflowParseError, match="absolute"):
        MlflowParser(payload)


def test_postgres_is_not_wired(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["backend_store"]["database_type"] = "postgres"
    with pytest.raises(MlflowParseError, match="not wired"):
        MlflowParser(payload)


def test_unknown_setup_key_is_an_error(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["extra"] = 1
    with pytest.raises(MlflowParseError, match="keys must be"):
        MlflowParser(payload)


def test_parser_does_not_create_directories(tmp_path: Path) -> None:
    parsed = MlflowParser(_payload(tmp_path))
    assert not parsed.db_path.parent.exists()
    assert not parsed.artifact_root.exists()
