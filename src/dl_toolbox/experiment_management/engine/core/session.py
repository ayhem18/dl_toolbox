"""Shared handle for fit and inspect. Assembled by ExpEngine."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.dl_engine.loggers.base_logger import Logger
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager
from dl_toolbox.dl_engine.training.checkpointer.checkpointer import Checkpointer
from dl_toolbox.dl_engine.training.config import FitConfig, SeedConfig
from dl_toolbox.experiment_management.experiment.build import ExperimentBuild
from dl_toolbox.experiment_management.recipe.run_identity import RunIdentity


class RunSession:
    def __init__(
        self,
        *,
        identity: RunIdentity,
        fit_config: FitConfig,
        checkpointer: Checkpointer,
        logger: Logger,
        logging_manager: LoggingManager,
        seed: SeedConfig,
        experiment: ExperimentBuild,
    ) -> None:
        self.identity = identity
        self.fit_config = fit_config
        self.checkpointer = checkpointer
        self.logger = logger
        self.logging_manager = logging_manager
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
