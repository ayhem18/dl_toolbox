"""Opinionated project-tree names. Paths are relative to the project root."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

PROJECT_MARKER_FILENAME = "dl_toolbox_project.json"
TRACKING_BACKEND_FILENAME = ".tracking-backend"
PROJECT_LEVEL_CONFIGS_DIRNAME = "project_level_configs"
LOGGING_CONFIGS_DIRNAME = "logging"
STORAGE_DIRNAME = "storage"
RUNS_DIRNAME = "runs"
TENSORBOARD_DIRNAME = "tensorboard"
MLFLOW_DIRNAME = "mlflow"
MLFLOW_ARTIFACTS_DIRNAME = "artifacts"
WANDB_DIRNAME = "wandb"
PLACEHOLDER_DIRNAME = "placeholder"
SCRIPTS_DIRNAME = "scripts"
EXPERIMENTS_DIRNAME = "experiments"

BACKEND_TENSORBOARD = "tensorboard"
BACKEND_MLFLOW = "mlflow"
BACKEND_WANDB = "wandb"
BACKEND_PLACEHOLDER = "placeholder"
DEFAULT_TRACKING_BACKEND = BACKEND_TENSORBOARD
TRACKING_BACKENDS = frozenset(
    {
        BACKEND_TENSORBOARD,
        BACKEND_MLFLOW,
        BACKEND_WANDB,
        BACKEND_PLACEHOLDER,
    }
)

RESERVED_CHILD_NAMES = frozenset(
    {
        PROJECT_LEVEL_CONFIGS_DIRNAME,
        STORAGE_DIRNAME,
        SCRIPTS_DIRNAME,
        EXPERIMENTS_DIRNAME,
        PROJECT_MARKER_FILENAME,
        TRACKING_BACKEND_FILENAME,
    }
)


class ProjectLayout(ABC):
    """Filesystem and tracking-store locations for one toolbox project."""

    @abstractmethod
    def project_root(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def storage_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def project_runs_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def experiments_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def scripts_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def project_level_configs_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def tracking_backend(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_tensorboard_global_config(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_mlflow_global_config(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_wandb_global_config(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_placeholder_global_config(self) -> dict:
        raise NotImplementedError
