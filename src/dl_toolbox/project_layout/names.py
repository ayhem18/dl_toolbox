"""Opinionated project-tree names. Relative to the project root."""

from __future__ import annotations

from dl_toolbox.remote_store.backends import RemoteBackend

MARKER_FILENAME = "dl_toolbox_project.json"
TRACKING_BACKEND_FILENAME = ".tracking-backend"
GITIGNORE_FILENAME = ".gitignore"
PROJECT_LEVEL_CONFIGS_DIRNAME = "project_level_configs"
IDENTITY_CONFIG_FILENAME = "identity.json"
CHECKPOINT_CONFIG_FILENAME = "checkpoint.json"
LOGGING_CONFIGS_DIRNAME = "logging"
REMOTE_CONFIGS_DIRNAME = "remote"
REMOTE_CONFIG_FILENAME = "remote.json"
REMOTE_CREDENTIALS_DIRNAME = "credentials"
STORAGE_DIRNAME = "storage"
RUNS_DIRNAME = "runs"
RUN_ARTIFACTS_DIRNAME = "run_artifacts"
SCRIPTS_DIRNAME = "scripts"
MLFLOW_RUN_FILENAME = "mlflow_run.py"
TENSORBOARD_RUN_FILENAME = "tensorboard_run.py"
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

DEFAULT_S3_ENDPOINT_URL = "http://localhost:9000"
DEFAULT_S3_REGION = "us-east-1"


def default_remote_config(*, backend: str, project_name: str) -> dict:
    """Opinionated, non-secret ``remote.json`` for ``backend``, committed to git.

    Only the S3/MinIO shape is wired today; the Hugging Face shape is a
    placeholder the user edits, and ``ProjectLayout.remote_mirror`` refuses it
    until that backend lands.
    """
    if backend == RemoteBackend.S3:
        return {
            "backend": RemoteBackend.S3,
            "bucket": project_name,
            "endpoint_url": DEFAULT_S3_ENDPOINT_URL,
            "region": DEFAULT_S3_REGION,
            "secure": False,
            "addressing_style": "path",
        }
    return {
        "backend": RemoteBackend.HUGGINGFACE,
        "repo_id": project_name,
        "repo_type": "model",
    }
