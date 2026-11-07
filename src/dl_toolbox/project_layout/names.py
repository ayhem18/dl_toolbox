"""Opinionated project-tree names. Relative to the project root."""

from __future__ import annotations

MARKER_FILENAME = "dl_toolbox_project.json"
TRACKING_BACKEND_FILENAME = ".tracking-backend"
PROJECT_LEVEL_CONFIGS_DIRNAME = "project_level_configs"
IDENTITY_CONFIG_FILENAME = "identity.json"
CHECKPOINT_CONFIG_FILENAME = "checkpoint.json"
LOGGING_CONFIGS_DIRNAME = "logging"
STORAGE_DIRNAME = "storage"
RUNS_DIRNAME = "runs"
RUN_ARTIFACTS_DIRNAME = "run_artifacts"
SCRIPTS_DIRNAME = "scripts"
EXPERIMENTS_DIRNAME = "experiments"
RUN_RECIPES_DIRNAME = "run_recipes"
TENSORBOARD_DIRNAME = "tensorboard"
MLFLOW_DIRNAME = "mlflow"
MLFLOW_ARTIFACTS_DIRNAME = "artifacts"
MLFLOW_TRACKING_DB_FILENAME = "tracking.db"

DEFAULT_IDENTITY_CONFIG = {
    "prefix_len": 12,
    "human_readable_dirname": "human_readable",
    "hashes_dirname": "hashes",
}
DEFAULT_CHECKPOINT_CONFIG = {
    "lock_timeout_seconds": 600.0,
}
SQLITE_TIMEOUT_SECONDS = 30.0
