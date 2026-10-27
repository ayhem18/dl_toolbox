"""Default engine: same core, managers built from the default layout."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager
from dl_toolbox.dl_engine.loggers.wandb.wandb_manager import WandbManager
from dl_toolbox.dl_engine.training.checkpointer.manager import CheckpointManager
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
    CHECKPOINT_CONFIG_FILENAME,
    IDENTITY_CONFIG_FILENAME,
    LOGGING_CONFIGS_DIRNAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
    RUN_ARTIFACTS_DIRNAME,
    RUNS_DIRNAME,
    STORAGE_DIRNAME,
)
from dl_toolbox.experiment_management.default_layout.parsing.checkpoint import (
    CheckpointProjectConfigParser,
)
from dl_toolbox.experiment_management.default_layout.parsing.identity import IdentityConfigParser
from dl_toolbox.experiment_management.default_layout.parsing.logging import (
    parse_project_logger_config,
)
from dl_toolbox.experiment_management.engine.core.engine import ExpEngine
from dl_toolbox.experiment_management.engine.core.paths import EnginePaths
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager
from dl_toolbox.experiment_management.toolbox.manager import DlToolboxConfigManager


class DefaultExpEngine(ExpEngine):
    """Walks the default tree, parses project JSON, constructs Fold-1 managers."""

    def __init__(
        self,
        project_name: str,
        *,
        start: Path,
        experiment: ExperimentConfigManager,
    ) -> None:
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
        backend = read_tracking_backend(root)
        super().__init__(
            identity_manager=_make_identity_manager(root),
            logging_manager=_make_logging_manager(root, backend=backend),
            checkpoint_manager=_make_checkpoint_manager(root),
            experiment_manager=experiment,
            paths=EnginePaths(
                project_root=root,
                run_identities_root=root / STORAGE_DIRNAME / RUNS_DIRNAME,
                run_artifacts_root=root / STORAGE_DIRNAME / RUN_ARTIFACTS_DIRNAME,
            ),
            compose_toolbox=DlToolboxConfigManager,
            tracking_backend=backend,
        )
        self._project_name = name
        self._package = document["package"] if "package" in document else None

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def package_name(self) -> str | None:
        return self._package

    @property
    def project_root(self) -> Path:
        return self._paths.project_root

    @property
    def tracking_backend(self) -> str:
        return self._tracking_backend


def _project_level_configs_dir(project_root: Path) -> Path:
    return project_root / PROJECT_LEVEL_CONFIGS_DIRNAME


def _make_identity_manager(project_root: Path) -> HashManager:
    path = _project_level_configs_dir(project_root) / IDENTITY_CONFIG_FILENAME
    parsed = IdentityConfigParser(read_json(path))
    return HashManager(
        prefix_len=parsed.prefix_len,
        human_readable_dirname=parsed.human_readable_dirname,
        hashes_dirname=parsed.hashes_dirname,
    )


def _make_logging_manager(project_root: Path, *, backend: str) -> LoggingManager:
    path = (
        _project_level_configs_dir(project_root)
        / LOGGING_CONFIGS_DIRNAME
        / f"{backend}.json"
    )
    payload = parse_mapping(read_json(path), label=f"{backend} project config")
    parsed = parse_project_logger_config(
        backend, payload, project_root=project_root
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


def _make_checkpoint_manager(project_root: Path) -> CheckpointManager:
    path = _project_level_configs_dir(project_root) / CHECKPOINT_CONFIG_FILENAME
    parsed = CheckpointProjectConfigParser(read_json(path))
    return CheckpointManager(lock_timeout_seconds=parsed.lock_timeout_seconds)
