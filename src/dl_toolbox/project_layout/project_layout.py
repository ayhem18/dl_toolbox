"""Opinionated toolbox project tree."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import JsonTextForm, read, write
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.project_layout.documents import (
    CheckpointDocument,
    IdentityDocument,
    MarkerDocument,
    MlflowLoggingDocument,
    TensorBoardLoggingDocument,
)
from dl_toolbox.project_layout.names import (
    CHECKPOINT_CONFIG_FILENAME,
    DEFAULT_CHECKPOINT_CONFIG,
    DEFAULT_IDENTITY_CONFIG,
    EXPERIMENTS_DIRNAME,
    IDENTITY_CONFIG_FILENAME,
    LOGGING_CONFIGS_DIRNAME,
    MARKER_FILENAME,
    MLFLOW_ARTIFACTS_DIRNAME,
    MLFLOW_DIRNAME,
    MLFLOW_TRACKING_DB_FILENAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    RUN_ARTIFACTS_DIRNAME,
    RUN_RECIPES_DIRNAME,
    RUNS_DIRNAME,
    SCRIPTS_DIRNAME,
    SQLITE_TIMEOUT_SECONDS,
    STORAGE_DIRNAME,
    TENSORBOARD_DIRNAME,
    TRACKING_BACKEND_FILENAME,
)


class TrackingBackend:
    TENSORBOARD = "tensorboard"
    MLFLOW = "mlflow"


_BACKENDS = frozenset({TrackingBackend.TENSORBOARD, TrackingBackend.MLFLOW})


class _TensorBoardLogging:
    def __init__(self, *, log_dir: Directory) -> None:
        self.log_dir = log_dir


class _MlflowLogging:
    def __init__(
        self, *, db_path: Path, timeout_seconds: float, artifact_root: Directory
    ) -> None:
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        self.artifact_root = artifact_root


class ProjectLayout:
    """One toolbox project tree, already located and read."""

    def __init__(
        self,
        *,
        root: Directory,
        project_name: str,
        tracking_backend: str,
        hash_manager: HashManager,
        lock_timeout_seconds: float,
        logging: _TensorBoardLogging | _MlflowLogging,
    ) -> None:
        if root.spec != DirectorySpec.EXISTS:
            raise ValueError(f"project root Directory must exist: {root.path}")
        if tracking_backend == TrackingBackend.TENSORBOARD:
            if not isinstance(logging, _TensorBoardLogging):
                raise TypeError("tensorboard backend requires tensorboard logging")
        elif tracking_backend == TrackingBackend.MLFLOW:
            if not isinstance(logging, _MlflowLogging):
                raise TypeError("mlflow backend requires mlflow logging")
        else:
            raise ValueError(f"unknown tracking_backend {tracking_backend!r}")
        self._root = root
        self._project_name = project_name
        self._tracking_backend = tracking_backend
        self._hash_manager = hash_manager
        self._lock_timeout_seconds = lock_timeout_seconds
        self._logging = logging

    @property
    def root(self) -> Directory:
        return self._root

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def tracking_backend(self) -> str:
        return self._tracking_backend

    @property
    def hash_manager(self) -> HashManager:
        return self._hash_manager

    @property
    def lock_timeout_seconds(self) -> float:
        return self._lock_timeout_seconds

    def tensorboard_log_dir(self) -> Directory:
        logging = self._logging
        if not isinstance(logging, _TensorBoardLogging):
            raise ValueError(
                "tensorboard_log_dir is only legal when tracking_backend is "
                f"{TrackingBackend.TENSORBOARD!r}"
            )
        return logging.log_dir

    def mlflow_db_path(self) -> Path:
        return self._mlflow_logging().db_path

    def mlflow_timeout_seconds(self) -> float:
        return self._mlflow_logging().timeout_seconds

    def mlflow_artifact_root(self) -> Directory:
        return self._mlflow_logging().artifact_root

    @classmethod
    def from_start(
        cls,
        *,
        start: File | Directory,
        project_name: str,
        tracking_backend: str,
    ) -> ProjectLayout:
        backend = _parse_backend(tracking_backend)
        expected_name = parse_non_empty_str(project_name, label="project_name")
        marker = File.from_parent_walk(start=start, basename=MARKER_FILENAME)
        root = marker.parent()
        document = _marker_document(read(file=marker))
        if document.name != expected_name:
            raise ValueError(
                f"{MARKER_FILENAME} name {document.name!r} does not match "
                f"project_name {expected_name!r}"
            )
        identity = _identity_document(
            read(
                file=File.from_under(
                    root=root,
                    relative_path=(
                        f"{PROJECT_LEVEL_CONFIGS_DIRNAME}/{IDENTITY_CONFIG_FILENAME}"
                    ),
                )
            )
        )
        hash_manager = HashManager(
            prefix_len=identity.prefix_len,
            human_readable_dir_name=identity.human_readable_dirname,
            hash_dir_name=identity.hashes_dirname,
        )
        checkpoint = _checkpoint_document(
            read(
                file=File.from_under(
                    root=root,
                    relative_path=(
                        f"{PROJECT_LEVEL_CONFIGS_DIRNAME}/{CHECKPOINT_CONFIG_FILENAME}"
                    ),
                )
            )
        )
        logging_relative = (
            f"{PROJECT_LEVEL_CONFIGS_DIRNAME}/{LOGGING_CONFIGS_DIRNAME}/"
            f"{backend}.json"
        )
        logging_payload = read(
            file=File.from_under(root=root, relative_path=logging_relative)
        )
        if backend == TrackingBackend.TENSORBOARD:
            logging: _TensorBoardLogging | _MlflowLogging = _TensorBoardLogging(
                log_dir=_tensorboard_log_dir(root, logging_payload)
            )
        else:
            db_path, timeout_seconds, artifact_root = _mlflow_store(
                root, logging_payload
            )
            logging = _MlflowLogging(
                db_path=db_path,
                timeout_seconds=timeout_seconds,
                artifact_root=artifact_root,
            )
        return cls(
            root=root,
            project_name=expected_name,
            tracking_backend=backend,
            hash_manager=hash_manager,
            lock_timeout_seconds=checkpoint.lock_timeout_seconds,
            logging=logging,
        )

    @classmethod
    def scaffold(
        cls,
        *,
        root: Directory,
        project_name: str,
        tracking_backend: str,
    ) -> None:
        if root.spec != DirectorySpec.EXISTS:
            raise ValueError(f"scaffold root Directory must exist: {root.path}")
        backend = _parse_backend(tracking_backend)
        name = parse_non_empty_str(project_name, label="project_name")
        marker_path = root.path / MARKER_FILENAME
        if marker_path.is_file():
            raise FileExistsError(f"project marker already exists: {marker_path}")
        write(
            directory=root,
            name=MARKER_FILENAME,
            payload={"name": name},
            form=JsonTextForm.DOCUMENT,
        )
        (root.path / TRACKING_BACKEND_FILENAME).write_text(
            backend + "\n", encoding="utf-8"
        )
        configs = Directory.from_under(
            root=root,
            relative_path=PROJECT_LEVEL_CONFIGS_DIRNAME,
            policy=DirectoryParsePolicy.CREATE,
        )
        write(
            directory=configs,
            name=IDENTITY_CONFIG_FILENAME,
            payload=dict(DEFAULT_IDENTITY_CONFIG),
            form=JsonTextForm.DOCUMENT,
        )
        write(
            directory=configs,
            name=CHECKPOINT_CONFIG_FILENAME,
            payload=dict(DEFAULT_CHECKPOINT_CONFIG),
            form=JsonTextForm.DOCUMENT,
        )
        logging_dir = Directory.from_under(
            root=root,
            relative_path=f"{PROJECT_LEVEL_CONFIGS_DIRNAME}/{LOGGING_CONFIGS_DIRNAME}",
            policy=DirectoryParsePolicy.CREATE,
        )
        write(
            directory=logging_dir,
            name=f"{TrackingBackend.TENSORBOARD}.json",
            payload={"log_dir": f"{STORAGE_DIRNAME}/{TENSORBOARD_DIRNAME}"},
            form=JsonTextForm.DOCUMENT,
        )
        write(
            directory=logging_dir,
            name=f"{TrackingBackend.MLFLOW}.json",
            payload=_mlflow_scaffold_payload(),
            form=JsonTextForm.DOCUMENT,
        )
        reserved = (
            STORAGE_DIRNAME,
            SCRIPTS_DIRNAME,
            EXPERIMENTS_DIRNAME,
            RUN_RECIPES_DIRNAME,
        )
        n_reserved = len(reserved)
        for i in range(n_reserved):
            Directory.from_under(
                root=root,
                relative_path=reserved[i],
                policy=DirectoryParsePolicy.CREATE,
            )

    @classmethod
    def parse_run_group(cls, segments: object) -> tuple[str, ...]:
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
            raise TypeError(
                f"run_group must be a sequence of basenames, got {type(segments)!r}"
            )
        n_segments = len(segments)
        parsed: list[str] = []
        for i in range(n_segments):
            parsed.append(_assert_group_segment(segments[i], index=i))
        return tuple(parsed)

    def identity_index(self, *, run_group: tuple[str, ...]) -> Directory:
        _require_run_group(run_group)
        relative = _join_relative((STORAGE_DIRNAME, RUNS_DIRNAME) + run_group)
        return Directory.from_under(
            root=self._root,
            relative_path=relative,
            policy=DirectoryParsePolicy.ALLOW_MISSING,
        )

    def artifacts_directory(
        self, *, run_group: tuple[str, ...], name: str
    ) -> Directory:
        _require_run_group(run_group)
        checked = self._hash_manager.validate_human_readable_name(name)
        relative = _join_relative(
            (STORAGE_DIRNAME, RUN_ARTIFACTS_DIRNAME) + run_group + (checked,)
        )
        return Directory.from_under(
            root=self._root,
            relative_path=relative,
            policy=DirectoryParsePolicy.ALLOW_MISSING,
        )

    def _mlflow_logging(self) -> _MlflowLogging:
        logging = self._logging
        if not isinstance(logging, _MlflowLogging):
            raise ValueError(
                "mlflow store accessors are only legal when tracking_backend is "
                f"{TrackingBackend.MLFLOW!r}"
            )
        return logging


def _parse_backend(raw: object) -> str:
    backend = parse_non_empty_str(raw, label="tracking_backend")
    if backend not in _BACKENDS:
        raise ValueError(
            f"tracking_backend must be {TrackingBackend.TENSORBOARD!r} or "
            f"{TrackingBackend.MLFLOW!r}, got {backend!r}"
        )
    return backend


def _marker_document(payload: object) -> MarkerDocument:
    try:
        return MarkerDocument.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{MARKER_FILENAME}: {exc}") from exc


def _identity_document(payload: object) -> IdentityDocument:
    try:
        return IdentityDocument.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{IDENTITY_CONFIG_FILENAME}: {exc}") from exc


def _checkpoint_document(payload: object) -> CheckpointDocument:
    try:
        return CheckpointDocument.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{CHECKPOINT_CONFIG_FILENAME}: {exc}") from exc


def _tensorboard_log_dir(root: Directory, payload: object) -> Directory:
    try:
        document = TensorBoardLoggingDocument.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"tensorboard.json: {exc}") from exc
    return _resolve_directory(root, document.log_dir, label="log_dir")


def _mlflow_store(
    root: Directory, payload: object
) -> tuple[Path, float, Directory]:
    try:
        document = MlflowLoggingDocument.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"mlflow.json: {exc}") from exc
    backend = document.backend_store
    if backend.type != "database":
        raise ValueError(
            "backend_store.type='filesystem' is not wired; use type='database'"
        )
    if backend.database_type != "sqlite":
        raise ValueError(
            f"backend_store.database_type={backend.database_type!r} is not "
            "wired; use 'sqlite'"
        )
    db_path = _resolve_file_path(root, backend.path, label="backend_store.path")
    artifacts = _resolve_directory(
        root, document.artifact_store.root, label="artifact_store.root"
    )
    return db_path, backend.timeout_seconds, artifacts


def _resolve_directory(root: Directory, raw: str, *, label: str) -> Directory:
    del label
    path = Path(raw)
    if path.is_absolute():
        return Directory.from_absolute(
            path=path, policy=DirectoryParsePolicy.ALLOW_MISSING
        )
    return Directory.from_under(
        root=root,
        relative_path=raw,
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )


def _resolve_file_path(root: Directory, raw: str, *, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return parse_absolute_path(raw, label=label)
    return resolve_under_project(root.path, raw, label=label)


def _mlflow_scaffold_payload() -> dict:
    return {
        "backend_store": {
            "type": "database",
            "database_type": "sqlite",
            "path": (
                f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/{MLFLOW_TRACKING_DB_FILENAME}"
            ),
            "timeout_seconds": SQLITE_TIMEOUT_SECONDS,
        },
        "artifact_store": {
            "type": "filesystem",
            "root": (
                f"{STORAGE_DIRNAME}/{MLFLOW_DIRNAME}/{MLFLOW_ARTIFACTS_DIRNAME}"
            ),
        },
    }


def _assert_group_segment(raw: object, *, index: int) -> str:
    text = parse_non_empty_str(raw, label=f"run_group[{index}]")
    text = text.replace("\\", "/")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"run_group[{index}] must be a basename, got {text!r}")
    return text


def _require_run_group(run_group: object) -> None:
    if not isinstance(run_group, tuple):
        raise TypeError(
            f"run_group must be a tuple of basenames, got {type(run_group)!r}"
        )
    n_segments = len(run_group)
    for i in range(n_segments):
        if run_group[i] != _assert_group_segment(run_group[i], index=i):
            raise ValueError(
                f"run_group[{i}] must be parse_run_group output, got {run_group[i]!r}"
            )


def _join_relative(parts: tuple[str, ...]) -> str:
    n_parts = len(parts)
    if n_parts < 1:
        raise ValueError("relative path has no parts")
    return "/".join(parts)
