"""Local directory persistence for the placeholder logger."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.loggers.base_logger import LoggerError, copy_local_artifact, require_logger_open
from dl_toolbox.loggers.placeholder.placeholder_parser import PlaceholderParser


class PlaceholderManager:
    def __init__(self, payload: dict) -> None:
        parsed = PlaceholderParser(payload)
        self._log_dir = parsed.log_dir
        self._open = False

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def setup(self) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._open = True

    def log_params(self, params: Mapping[str, str]) -> None:
        del params
        require_logger_open(self._open)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        del tags
        require_logger_open(self._open)

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        del values, step
        require_logger_open(self._open)

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        require_logger_open(self._open)
        copy_local_artifact(path, self._log_dir, dest=dest)

    def close(self) -> None:
        self._open = False
