"""Opinionated MLflow setup dicts. Paths stay caller-owned."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SQLITE_TIMEOUT_SECONDS = 30


def default_sqlite_backend_store(
    *,
    path: str | Path,
    timeout_seconds: float = DEFAULT_SQLITE_TIMEOUT_SECONDS,
) -> dict:
    return {
        "type": "database",
        "database_type": "sqlite",
        "path": str(path),
        "timeout_seconds": timeout_seconds,
    }


def default_filesystem_artifact_store(*, root: str | Path) -> dict:
    return {
        "type": "filesystem",
        "root": str(root),
    }


def default_mlflow_setup(*, backend_store: dict, artifact_store: dict) -> dict:
    return {
        "backend_store": backend_store,
        "artifact_store": artifact_store,
    }
