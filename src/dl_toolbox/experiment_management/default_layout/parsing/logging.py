"""Project-level logging store JSON. Relative paths resolve under project_root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_positive_number
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys, require_key

_BACKEND_TYPES = frozenset({"database", "filesystem"})
_DATABASE_TYPES = frozenset({"sqlite", "postgres"})
_IMPLEMENTED_DATABASE_TYPES = frozenset({"sqlite"})
_ARTIFACT_TYPES = frozenset({"filesystem"})
_MLFLOW_TOP_KEYS = ("backend_store", "artifact_store")
_BACKEND_STORE_KEYS = ("type", "database_type", "path", "timeout_seconds")
_ARTIFACT_STORE_KEYS = ("type", "root")


class ProjectLoggerParseError(ValueError):
    """Invalid project logging JSON."""


def _mapping(payload: object, label: str) -> dict:
    try:
        return parse_mapping(payload, label=label)
    except (TypeError, ValueError) as exc:
        raise ProjectLoggerParseError(str(exc)) from exc


class TensorBoardProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "tensorboard project config")
        try:
            require_exact_keys(obj, ("log_dir",), label="tensorboard project config")
            relative = parse_non_empty_str(obj["log_dir"], label="log_dir")
            self._log_dir = resolve_under_project(project_root, relative, label="log_dir")
        except (ValueError, TypeError) as exc:
            raise ProjectLoggerParseError(str(exc)) from exc

    @property
    def log_dir(self) -> Path:
        return self._log_dir


class PlaceholderProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "placeholder project config")
        try:
            require_exact_keys(obj, ("log_dir",), label="placeholder project config")
            relative = parse_non_empty_str(obj["log_dir"], label="log_dir")
            self._log_dir = resolve_under_project(project_root, relative, label="log_dir")
        except (ValueError, TypeError) as exc:
            raise ProjectLoggerParseError(str(exc)) from exc

    @property
    def log_dir(self) -> Path:
        return self._log_dir


class WandbProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "wandb project config")
        try:
            require_exact_keys(obj, ("dir",), label="wandb project config")
            relative = parse_non_empty_str(obj["dir"], label="dir")
            self._dir = resolve_under_project(project_root, relative, label="dir")
        except (ValueError, TypeError) as exc:
            raise ProjectLoggerParseError(str(exc)) from exc

    @property
    def dir(self) -> Path:
        return self._dir


class MlflowProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "mlflow project config")
        try:
            require_exact_keys(obj, _MLFLOW_TOP_KEYS, label="mlflow project config")
            backend = _parse_backend_store(obj["backend_store"], project_root=project_root)
            artifacts = _parse_artifact_store(obj["artifact_store"], project_root=project_root)
        except ProjectLoggerParseError:
            raise
        except (ValueError, TypeError) as exc:
            raise ProjectLoggerParseError(str(exc)) from exc
        self._db_path = backend[0]
        self._timeout_seconds = backend[1]
        self._artifact_root = artifacts

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root


def _parse_backend_store(payload: Any, *, project_root: Path) -> tuple[Path, float]:
    obj = _mapping(payload, "backend_store")
    require_exact_keys(obj, _BACKEND_STORE_KEYS, label="backend_store")
    store_type = parse_non_empty_str(obj["type"], label="backend_store.type")
    if store_type not in _BACKEND_TYPES:
        raise ProjectLoggerParseError(
            "backend_store.type must be 'database' or 'filesystem', "
            f"found {store_type!r}"
        )
    if store_type == "filesystem":
        raise ProjectLoggerParseError(
            "backend_store.type='filesystem' is not wired yet; use type='database'"
        )
    database_type = parse_non_empty_str(
        obj["database_type"], label="backend_store.database_type"
    )
    if database_type not in _DATABASE_TYPES:
        raise ProjectLoggerParseError(
            "backend_store.database_type must be 'sqlite' or 'postgres', "
            f"found {database_type!r}"
        )
    if database_type not in _IMPLEMENTED_DATABASE_TYPES:
        raise ProjectLoggerParseError(
            f"backend_store.database_type={database_type!r} is not wired yet; "
            "use 'sqlite' (no database server). Postgres needs a running server "
            "and will be added when several writers share one store."
        )
    relative = parse_non_empty_str(obj["path"], label="backend_store.path")
    db_path = resolve_under_project(project_root, relative, label="backend_store.path")
    timeout = parse_positive_number(
        obj["timeout_seconds"], label="backend_store.timeout_seconds"
    )
    return db_path, float(timeout)


def _parse_artifact_store(payload: Any, *, project_root: Path) -> Path:
    obj = _mapping(payload, "artifact_store")
    require_exact_keys(obj, _ARTIFACT_STORE_KEYS, label="artifact_store")
    store_type = parse_non_empty_str(obj["type"], label="artifact_store.type")
    if store_type not in _ARTIFACT_TYPES:
        raise ProjectLoggerParseError(
            f"artifact_store.type must be 'filesystem', found {store_type!r}"
        )
    relative = parse_non_empty_str(obj["root"], label="artifact_store.root")
    return resolve_under_project(project_root, relative, label="artifact_store.root")


def parse_project_logger_config(backend: str, payload: dict, *, project_root: Path):
    if backend == "tensorboard":
        return TensorBoardProjectLoggerConfigParser(payload, project_root=project_root)
    if backend == "placeholder":
        return PlaceholderProjectLoggerConfigParser(payload, project_root=project_root)
    if backend == "wandb":
        return WandbProjectLoggerConfigParser(payload, project_root=project_root)
    if backend == "mlflow":
        return MlflowProjectLoggerConfigParser(payload, project_root=project_root)
    raise ValueError(f"unknown tracking backend {backend!r}")
