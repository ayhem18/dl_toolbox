"""Local directory persistence for the placeholder logger."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import LoggerError, copy_local_artifact, require_logger_open


class PlaceholderManager:
    def __init__(self, *, log_dir: Path) -> None:
        self._log_dir = parse_absolute_path(str(log_dir), label="log_dir")
        self._run_dir: Path | None = None
        self._open = False

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def setup(self, *, run_name: str) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        name = parse_non_empty_str(run_name, label="run_name")
        run_dir = self._log_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
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
        if self._run_dir is None:
            raise LoggerError("logger.setup() must run before logging")
        copy_local_artifact(path, self._run_dir, dest=dest)

    def close(self) -> None:
        self._open = False
