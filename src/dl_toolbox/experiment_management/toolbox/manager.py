"""Compose dl_toolbox_config parsers. Constructs Logger and CheckpointManager."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.loggers.base_logger import Logger
from dl_toolbox.dl_engine.loggers.factory import get_logger
from dl_toolbox.dl_engine.seed import seed_everything
from dl_toolbox.dl_engine.training.checkpoint import CheckpointManager
from dl_toolbox.dl_engine.training.config import (
    EarlyStoppingConfig,
    MonitorConfig,
    SeedConfig,
    TrainerConfig,
)
from dl_toolbox.experiment_management.logging.run_parser import parse_run_logger_config
from dl_toolbox.experiment_management.toolbox.checkpoint.parser import CheckpointConfigParser
from dl_toolbox.experiment_management.toolbox.early_stopping.parser import (
    EarlyStoppingConfigParser,
)
from dl_toolbox.experiment_management.toolbox.seed.parser import SeedConfigParser
from dl_toolbox.experiment_management.toolbox.trainer.parser import TrainerConfigParser

_KEYS = (
    "logger_config",
    "checkpoint_config",
    "trainer_config",
    "seed_config",
    "early_stopping_config",
)


class DlToolboxConfigManager:
    def __init__(
        self,
        payload: dict,
        *,
        backend: str,
        logging_manager: object,
    ) -> None:
        obj = parse_mapping(payload, label="dl_toolbox_config")
        require_exact_keys(obj, _KEYS, label="dl_toolbox_config")
        self._backend = backend
        self._logging_manager = logging_manager
        self._logger_parser = parse_run_logger_config(backend, obj["logger_config"])
        self._checkpoint_parser = CheckpointConfigParser(obj["checkpoint_config"])
        self._trainer_parser = TrainerConfigParser(obj["trainer_config"])
        self._seed_parser = SeedConfigParser(obj["seed_config"])
        self._early_stopping_parser = EarlyStoppingConfigParser(
            obj["early_stopping_config"]
        )

    def apply_seed(self) -> None:
        seed_everything(self._seed_parser.seed_config.seed)

    def seed_config(self) -> SeedConfig:
        return self._seed_parser.seed_config

    def trainer_config(self) -> TrainerConfig:
        return self._trainer_parser.trainer_config

    def early_stopping_config(self) -> EarlyStoppingConfig:
        return self._early_stopping_parser.early_stopping_config

    def build_logger(self) -> Logger:
        parsed = self._logger_parser
        kwargs: dict = {"run_name": parsed.run_name}
        if self._backend == "mlflow":
            kwargs["experiment"] = parsed.experiment
            kwargs["tags"] = parsed.tags
        elif self._backend == "wandb":
            kwargs["project"] = parsed.project
            kwargs["tags"] = parsed.tags
        return get_logger(
            self._backend,
            manager=self._logging_manager,
            **kwargs,
        )

    def materialize_checkpoint(
        self,
        *,
        run_dir: Path,
        monitor: MonitorConfig,
    ) -> CheckpointManager:
        return CheckpointManager(
            save_top_k=self._checkpoint_parser.save_top_k,
            save_last=self._checkpoint_parser.save_last,
            run_dir=run_dir,
            monitor=monitor,
        )
