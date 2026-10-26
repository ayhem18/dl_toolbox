"""TensorBoard training logger. Persistence lives on the manager."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.loggers.base_logger import Logger
from dl_toolbox.loggers.tensorboard.tensorboard_manager import TensorBoardManager


class TensorBoardLogger(Logger):
    def __init__(self, *, manager: TensorBoardManager) -> None:
        self._manager = manager

    def setup(self) -> None:
        self._manager.setup()

    def log_params(self, params: Mapping[str, str]) -> None:
        self._manager.log_params(params)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        self._manager.log_tags(tags)

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        self._manager.log_metrics(values, step=step)

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        self._manager.log_artifact(path, dest=dest)

    def close(self) -> None:
        self._manager.close()
