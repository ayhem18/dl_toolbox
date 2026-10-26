"""Default layout: project name plus a walk to dl_toolbox_project.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys, require_key
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.loggers.tensorboard.tensorboard_manager import TensorBoardManager
from dl_toolbox.loggers.wandb.wandb_manager import WandbManager
from dl_toolbox.project_layout.discover import find_project_root, load_project_document, read_tracking_backend
from dl_toolbox.project_layout.layout import (
    BACKEND_MLFLOW,
    BACKEND_PLACEHOLDER,
    BACKEND_TENSORBOARD,
    BACKEND_WANDB,
    EXPERIMENTS_DIRNAME,
    LOGGING_CONFIGS_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    RUNS_DIRNAME,
    SCRIPTS_DIRNAME,
    STORAGE_DIRNAME,
)
from dl_toolbox.run_manager import RunManager


class DefaultRunManager(RunManager):
    def __init__(self, project_name: str, *, start: Path) -> None:
        name = assert_generating_process_name(project_name)
        if not start.is_absolute():
            raise ValueError(f"start must be an absolute path, got {start}")
        root = find_project_root(start=start)
        document = load_project_document(root)
        stored_name = document["name"]
        if stored_name != name:
            raise ValueError(
                f"project_name {name!r} does not match dl_toolbox_project.json name "
                f"{stored_name!r} under {root}"
            )
        self._project_name = name
        self._root = root
        self._package = document["package"] if "package" in document else None

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def package_name(self) -> str | None:
        return self._package

    def project_root(self) -> Path:
        return self._root

    def storage_dir(self) -> Path:
        return self._root / STORAGE_DIRNAME

    def project_runs_dir(self) -> Path:
        return self.storage_dir() / RUNS_DIRNAME

    def experiments_dir(self) -> Path:
        return self._root / EXPERIMENTS_DIRNAME

    def scripts_dir(self) -> Path:
        return self._root / SCRIPTS_DIRNAME

    def project_level_configs_dir(self) -> Path:
        return self._root / PROJECT_LEVEL_CONFIGS_DIRNAME

    def tracking_backend(self) -> str:
        return read_tracking_backend(self._root)

    def make_logging_manager(self) -> object:
        backend = self.tracking_backend()
        payload = self._store_payload(backend)
        if backend == BACKEND_TENSORBOARD:
            return TensorBoardManager(payload)
        if backend == BACKEND_PLACEHOLDER:
            return PlaceholderManager(payload)
        if backend == BACKEND_WANDB:
            return WandbManager(payload)
        if backend == BACKEND_MLFLOW:
            return MlflowManager(payload)
        raise ValueError(f"unknown tracking backend {backend!r}")

    def _store_payload(self, backend: str) -> dict[str, Any]:
        if backend == BACKEND_TENSORBOARD:
            return self._materialize_tensorboard()
        if backend == BACKEND_PLACEHOLDER:
            return self._materialize_placeholder()
        if backend == BACKEND_WANDB:
            return self._materialize_wandb()
        if backend == BACKEND_MLFLOW:
            return self._materialize_mlflow()
        raise ValueError(f"unknown tracking backend {backend!r}")

    def _read_backend_document(self, backend: str) -> dict[str, Any]:
        path = (
            self.project_level_configs_dir() / LOGGING_CONFIGS_DIRNAME / f"{backend}.json"
        )
        return parse_mapping(read_json(path), label=f"{backend} project config")

    def _materialize_tensorboard(self) -> dict:
        document = self._read_backend_document(BACKEND_TENSORBOARD)
        require_exact_keys(document, ("log_dir",), label="tensorboard project config")
        log_dir = resolve_under_project(
            self._root,
            parse_non_empty_str(document["log_dir"], label="log_dir"),
            label="log_dir",
        )
        return {"log_dir": str(log_dir)}

    def _materialize_placeholder(self) -> dict:
        document = self._read_backend_document(BACKEND_PLACEHOLDER)
        require_exact_keys(document, ("log_dir",), label="placeholder project config")
        log_dir = resolve_under_project(
            self._root,
            parse_non_empty_str(document["log_dir"], label="log_dir"),
            label="log_dir",
        )
        return {"log_dir": str(log_dir)}

    def _materialize_wandb(self) -> dict:
        document = self._read_backend_document(BACKEND_WANDB)
        require_exact_keys(document, ("dir",), label="wandb project config")
        directory = resolve_under_project(
            self._root,
            parse_non_empty_str(document["dir"], label="dir"),
            label="dir",
        )
        return {"dir": str(directory)}

    def _materialize_mlflow(self) -> dict:
        document = self._read_backend_document(BACKEND_MLFLOW)
        require_exact_keys(
            document,
            ("backend_store", "artifact_store"),
            label="mlflow project config",
        )
        backend_store = parse_mapping(
            require_key(document, "backend_store", label="mlflow project config"),
            label="backend_store",
        )
        artifact_store = parse_mapping(
            require_key(document, "artifact_store", label="mlflow project config"),
            label="artifact_store",
        )
        resolved_backend = dict(backend_store)
        resolved_backend["path"] = str(
            resolve_under_project(
                self._root,
                parse_non_empty_str(backend_store["path"], label="backend_store.path"),
                label="backend_store.path",
            )
        )
        resolved_artifacts = dict(artifact_store)
        resolved_artifacts["root"] = str(
            resolve_under_project(
                self._root,
                parse_non_empty_str(artifact_store["root"], label="artifact_store.root"),
                label="artifact_store.root",
            )
        )
        return {
            "backend_store": resolved_backend,
            "artifact_store": resolved_artifacts,
        }
