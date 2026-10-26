"""Opinionated project-tree names. Paths are relative to the project root."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dl_toolbox.parsing.parse_str import parse_non_empty_str

PROJECT_MARKER_FILENAME = "dl_toolbox_project.json"
TRACKING_BACKEND_FILENAME = ".tracking-backend"
PROJECT_LEVEL_CONFIGS_DIRNAME = "project_level_configs"
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


def resolve_under_project(project_root: Path, relative_path: str, *, label: str) -> Path:
    """Join a posix-relative path under an absolute project root.

    The target does not have to exist. ``..`` and absolute paths are rejected.
    This function does not use cwd.
    """
    if not project_root.is_absolute():
        raise ValueError(f"project_root must be an absolute path, got {project_root}")
    root = project_root.resolve()
    text = parse_non_empty_str(relative_path, label=label).replace("\\", "/")
    if Path(text).is_absolute() or text.startswith("/"):
        raise ValueError(f"{label} must be relative to the project root, got {relative_path!r}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"{label} is empty after normalization: {relative_path!r}")
    n_parts = len(parts)
    for i in range(n_parts):
        if parts[i] == "..":
            raise ValueError(
                f"{label} must stay under the project root, got {relative_path!r}"
            )
    absolute = root.joinpath(*parts).resolve()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes the project root: {relative_path!r}"
        ) from exc
    return absolute
