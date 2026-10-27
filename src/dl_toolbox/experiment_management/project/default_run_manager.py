"""Default layout: project name plus a walk to dl_toolbox_project.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager
from dl_toolbox.dl_engine.loggers.wandb.wandb_manager import WandbManager
from dl_toolbox.experiment_management.default_layout.layout.discover import (
    find_project_root,
    load_project_document,
    read_tracking_backend,
)
from dl_toolbox.experiment_management.default_layout.layout.names import (
    BACKEND_MLFLOW,
    BACKEND_PLACEHOLDER,
    BACKEND_TENSORBOARD,
    BACKEND_WANDB,
    EXPERIMENTS_DIRNAME,
    IDENTITY_CONFIG_FILENAME,
    LOGGING_CONFIGS_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    RUN_ARTIFACTS_DIRNAME,
    RUNS_DIRNAME,
    SCRIPTS_DIRNAME,
    STORAGE_DIRNAME,
)
from dl_toolbox.experiment_management.default_layout.parsing.identity import (
    IdentityConfigParser,
)
from dl_toolbox.experiment_management.default_layout.parsing.logging import (
    parse_project_logger_config,
)
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager
from dl_toolbox.experiment_management.project.abstract_run_manager import AbstractRunManager


class DefaultRunManager(AbstractRunManager):
    def __init__(
        self,
        project_name: str,
        *,
        start: Path,
        experiment_config_manager: ExperimentConfigManager,
    ) -> None:
        super().__init__(experiment_config_manager=experiment_config_manager)
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

    def run_payload_dir(self, name: str) -> Path:
        return self.storage_dir() / RUN_ARTIFACTS_DIRNAME / name

    def experiments_dir(self) -> Path:
        return self._root / EXPERIMENTS_DIRNAME

    def scripts_dir(self) -> Path:
        return self._root / SCRIPTS_DIRNAME

    def project_level_configs_dir(self) -> Path:
        return self._root / PROJECT_LEVEL_CONFIGS_DIRNAME

    def tracking_backend(self) -> str:
        return read_tracking_backend(self._root)

    def make_hash_manager(self) -> HashManager:
        path = self.project_level_configs_dir() / IDENTITY_CONFIG_FILENAME
        parsed = IdentityConfigParser(read_json(path))
        return HashManager(
            prefix_len=parsed.prefix_len,
            human_readable_dirname=parsed.human_readable_dirname,
            hashes_dirname=parsed.hashes_dirname,
        )

    def make_logging_manager(self) -> object:
        backend = self.tracking_backend()
        payload = self._read_backend_document(backend)
        parsed = parse_project_logger_config(
            backend, payload, project_root=self._root
        )
        if backend == BACKEND_TENSORBOARD:
            return TensorBoardManager(log_dir=parsed.log_dir)
        if backend == BACKEND_PLACEHOLDER:
            return PlaceholderManager(log_dir=parsed.log_dir)
        if backend == BACKEND_WANDB:
            return WandbManager(dir=parsed.dir)
        if backend == BACKEND_MLFLOW:
            return MlflowManager(
                db_path=parsed.db_path,
                artifact_root=parsed.artifact_root,
                timeout_seconds=parsed.timeout_seconds,
            )
        raise ValueError(f"unknown tracking backend {backend!r}")

    def _read_backend_document(self, backend: str) -> dict[str, Any]:
        path = (
            self.project_level_configs_dir() / LOGGING_CONFIGS_DIRNAME / f"{backend}.json"
        )
        return parse_mapping(read_json(path), label=f"{backend} project config")
