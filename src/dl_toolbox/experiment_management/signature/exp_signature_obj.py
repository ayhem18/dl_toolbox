"""Shared handle for fit and inspect. Assembled by AbstractRunManager."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.dl_engine.loggers.base_logger import Logger
from dl_toolbox.dl_engine.training.checkpoint import CheckpointManager
from dl_toolbox.dl_engine.training.config import FitConfig, SeedConfig
from dl_toolbox.experiment_management.experiment.build import ExperimentBuild
from dl_toolbox.experiment_management.recipe.run_identity import RunIdentity


class ExpSignatureObj:
    def __init__(
        self,
        *,
        identity: RunIdentity,
        fit_config: FitConfig,
        checkpoint_manager: CheckpointManager,
        logger: Logger,
        seed: SeedConfig,
        experiment: ExperimentBuild,
    ) -> None:
        self.identity = identity
        self.fit_config = fit_config
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.seed = seed
        self.experiment = experiment

    @property
    def run_dir(self) -> Path:
        return self.identity.run_dir

    @property
    def human_readable_name(self) -> str:
        return self.identity.human_readable_name

    @property
    def hash(self) -> str:
        return self.identity.hash
