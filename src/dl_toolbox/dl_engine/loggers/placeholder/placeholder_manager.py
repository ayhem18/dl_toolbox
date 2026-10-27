"""Local directory store for the placeholder logger. No log_*."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager


class PlaceholderManager(LoggingManager):
    def __init__(self, *, log_dir: Path) -> None:
        self._log_dir = parse_absolute_path(str(log_dir), label="log_dir")

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def prepare(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_name: str) -> Path:
        name = parse_non_empty_str(run_name, label="run_name")
        return self._log_dir / name
