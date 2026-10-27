"""W&B local-dir store. Import wandb only here. Logging lives on the logger."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.dl_engine.loggers.base_logger import import_extra
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager

_EXTRA = "wandb"


class WandbManager(LoggingManager):
    def __init__(self, *, dir: Path) -> None:
        self._dir = parse_absolute_path(str(dir), label="dir")

    @property
    def dir(self) -> Path:
        return self._dir

    def prepare(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def import_wandb():
        return import_extra("wandb", extra=_EXTRA)
