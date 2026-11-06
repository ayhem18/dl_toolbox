"""Project-level logging store JSON. Relative paths resolve under project_root.

Replacement patterns (File / Directory). Properties still return ``Path``
because logger managers are outside this folder.

1. ``log_dir`` / ``dir`` / ``artifact_root`` — five ``resolve_under_project`` calls

   Before::

       relative = parse_non_empty_str(obj["log_dir"], label="log_dir")
       self._log_dir = resolve_under_project(
           project_root, relative, label="log_dir"
       )

   After::

       project = Directory.from_absolute(
           path=project_root, policy=DirectoryParsePolicy.MUST_EXIST
       )
       self._log_dir = Directory.from_under(
           root=project,
           relative_path=relative,
           policy=DirectoryParsePolicy.ALLOW_MISSING,
       )

2. sqlite ``tracking.db`` is not a File (it does not exist yet)

   Before::

       db_path = resolve_under_project(
           project_root, "storage/mlflow/tracking.db", label="..."
       )

   After::

       db_dir = Directory.from_under(
           root=project,
           relative_path="storage/mlflow",
           policy=DirectoryParsePolicy.ALLOW_MISSING,
       )
       db_path = db_dir.path / "tracking.db"

3. ``project_root`` is parsed once per parser, not once per relative field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_positive_number
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

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


def _project(project_root: Path) -> Directory:
    return Directory.from_absolute(
        path=project_root, policy=DirectoryParsePolicy.MUST_EXIST
    )


def _allow_missing_under(*, project: Directory, relative_path: str) -> Directory:
    return Directory.from_under(
        root=project,
        relative_path=relative_path,
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )


def _dest_file_path_under(*, project: Directory, relative_path: str) -> Path:
    """Join a not-yet-written file under ``project``. Not a File parse."""
    parts = [
        part
        for part in relative_path.replace("\\", "/").split("/")
        if part not in ("", ".")
    ]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(
            f"relative_path must stay under the directory: {relative_path!r}"
        )
    n_parts = len(parts)
    basename = parts[n_parts - 1]
    if n_parts == 1:
        return project.path / basename
    parent = _allow_missing_under(
        project=project, relative_path="/".join(parts[: n_parts - 1])
    )
    return parent.path / basename


class TensorBoardProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "tensorboard project config")
        try:
            require_exact_keys(obj, ("log_dir",), label="tensorboard project config")
            relative = parse_non_empty_str(obj["log_dir"], label="log_dir")
            project = _project(project_root)
            self._log_dir = _allow_missing_under(
                project=project, relative_path=relative
            )
        except (ValueError, TypeError, FileNotFoundError) as exc:
            raise ProjectLoggerParseError(str(exc)) from exc

    @property
    def log_dir(self) -> Path:
        return self._log_dir.path


# --- pydantic (optional extra: validation) ---
# JSON shape of log_dir. Leftover: Directory.from_under (relative, no ..,
# ALLOW_MISSING). ValidationError is wrapped as ProjectLoggerParseError.


class TensorBoardProjectLoggerDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    log_dir: _NonEmptyStr


def tensorboard_project_from_pydantic(
    payload: object, *, project_root: Path
) -> TensorBoardProjectLoggerConfigParser:
    try:
        document = TensorBoardProjectLoggerDocument.model_validate(payload)
        project = _project(project_root)
        parsed = TensorBoardProjectLoggerConfigParser.__new__(
            TensorBoardProjectLoggerConfigParser
        )
        parsed._log_dir = _allow_missing_under(
            project=project, relative_path=document.log_dir
        )
        return parsed
    except ValidationError as exc:
        raise ProjectLoggerParseError(str(exc)) from exc
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise ProjectLoggerParseError(str(exc)) from exc


class WandbProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "wandb project config")
        try:
            require_exact_keys(obj, ("dir",), label="wandb project config")
            relative = parse_non_empty_str(obj["dir"], label="dir")
            project = _project(project_root)
            self._dir = _allow_missing_under(project=project, relative_path=relative)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            raise ProjectLoggerParseError(str(exc)) from exc

    @property
    def dir(self) -> Path:
        return self._dir.path


class MlflowProjectLoggerConfigParser:
    def __init__(self, payload: dict, *, project_root: Path) -> None:
        obj = _mapping(payload, "mlflow project config")
        try:
            require_exact_keys(obj, _MLFLOW_TOP_KEYS, label="mlflow project config")
            project = _project(project_root)
            backend = _parse_backend_store(
                obj["backend_store"], project=project
            )
            artifacts = _parse_artifact_store(
                obj["artifact_store"], project=project
            )
        except ProjectLoggerParseError:
            raise
        except (ValueError, TypeError, FileNotFoundError) as exc:
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
        return self._artifact_root.path


def _parse_backend_store(
    payload: Any, *, project: Directory
) -> tuple[Path, float]:
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
    db_path = _dest_file_path_under(project=project, relative_path=relative)
    timeout = parse_positive_number(
        obj["timeout_seconds"], label="backend_store.timeout_seconds"
    )
    return db_path, float(timeout)


def _parse_artifact_store(payload: Any, *, project: Directory) -> Directory:
    obj = _mapping(payload, "artifact_store")
    require_exact_keys(obj, _ARTIFACT_STORE_KEYS, label="artifact_store")
    store_type = parse_non_empty_str(obj["type"], label="artifact_store.type")
    if store_type not in _ARTIFACT_TYPES:
        raise ProjectLoggerParseError(
            f"artifact_store.type must be 'filesystem', found {store_type!r}"
        )
    relative = parse_non_empty_str(obj["root"], label="artifact_store.root")
    return _allow_missing_under(project=project, relative_path=relative)


def parse_project_logger_config(backend: str, payload: dict, *, project_root: Path):
    if backend == "tensorboard":
        return TensorBoardProjectLoggerConfigParser(payload, project_root=project_root)
    if backend == "wandb":
        return WandbProjectLoggerConfigParser(payload, project_root=project_root)
    if backend == "mlflow":
        return MlflowProjectLoggerConfigParser(payload, project_root=project_root)
    raise ValueError(f"unknown tracking backend {backend!r}")
