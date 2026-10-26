"""Validate an MLflow setup dict and expose the persistence fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_positive_number
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_BACKEND_TYPES = frozenset({"database", "filesystem"})
_DATABASE_TYPES = frozenset({"sqlite", "postgres"})
_IMPLEMENTED_DATABASE_TYPES = frozenset({"sqlite"})
_ARTIFACT_TYPES = frozenset({"filesystem"})
_TOP_KEYS = ("backend_store", "artifact_store")
_BACKEND_STORE_KEYS = ("type", "database_type", "path", "timeout_seconds")
_ARTIFACT_STORE_KEYS = ("type", "root")


class MlflowParseError(ValueError):
    """Invalid MLflow setup dictionary."""


@dataclass(frozen=True)
class _SqliteBackend:
    db_path: Path
    timeout_seconds: float

    def store_uri(self) -> str:
        posix = self.db_path.as_posix()
        timeout = (
            int(self.timeout_seconds)
            if self.timeout_seconds == int(self.timeout_seconds)
            else self.timeout_seconds
        )
        return f"sqlite:///{posix}?timeout={timeout}"


@dataclass(frozen=True)
class _FilesystemArtifactStore:
    root: Path

    def store_uri(self) -> str:
        return self.root.as_posix()


class MlflowParser:
    """Dictionary in; persistence fields out. No directories are created."""

    def __init__(self, payload: dict) -> None:
        try:
            obj = parse_mapping(payload, label="mlflow setup")
            require_exact_keys(obj, _TOP_KEYS, label="mlflow setup")
            backend = _parse_backend_store(obj["backend_store"])
            artifacts = _parse_artifact_store(obj["artifact_store"])
        except MlflowParseError:
            raise
        except (ValueError, TypeError) as exc:
            raise MlflowParseError(str(exc)) from exc
        self._backend_store_uri = backend.store_uri()
        self._artifact_store_uri = artifacts.store_uri()
        self._db_path = backend.db_path
        self._artifact_root = artifacts.root

    @property
    def backend_store_uri(self) -> str:
        return self._backend_store_uri

    @property
    def artifact_store_uri(self) -> str:
        return self._artifact_store_uri

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root


def _parse_backend_store(payload: Any) -> _SqliteBackend:
    obj = parse_mapping(payload, label="backend_store")
    require_exact_keys(obj, _BACKEND_STORE_KEYS, label="backend_store")
    store_type = parse_non_empty_str(obj["type"], label="backend_store.type")
    if store_type not in _BACKEND_TYPES:
        raise MlflowParseError(
            "backend_store.type must be 'database' or 'filesystem', "
            f"found {store_type!r}"
        )
    if store_type == "filesystem":
        raise MlflowParseError(
            "backend_store.type='filesystem' is not wired yet; use type='database'"
        )
    database_type = parse_non_empty_str(
        obj["database_type"], label="backend_store.database_type"
    )
    if database_type not in _DATABASE_TYPES:
        raise MlflowParseError(
            "backend_store.database_type must be 'sqlite' or 'postgres', "
            f"found {database_type!r}"
        )
    if database_type not in _IMPLEMENTED_DATABASE_TYPES:
        raise MlflowParseError(
            f"backend_store.database_type={database_type!r} is not wired yet; "
            "use 'sqlite' (no database server). Postgres needs a running server "
            "and will be added when several writers share one store."
        )
    db_path = parse_absolute_path(obj["path"], label="backend_store.path")
    timeout = parse_positive_number(
        obj["timeout_seconds"], label="backend_store.timeout_seconds"
    )
    return _SqliteBackend(db_path=db_path, timeout_seconds=float(timeout))


def _parse_artifact_store(payload: Any) -> _FilesystemArtifactStore:
    obj = parse_mapping(payload, label="artifact_store")
    require_exact_keys(obj, _ARTIFACT_STORE_KEYS, label="artifact_store")
    store_type = parse_non_empty_str(obj["type"], label="artifact_store.type")
    if store_type not in _ARTIFACT_TYPES:
        raise MlflowParseError(
            f"artifact_store.type must be 'filesystem', found {store_type!r}"
        )
    root = parse_absolute_path(obj["root"], label="artifact_store.root")
    return _FilesystemArtifactStore(root=root)
