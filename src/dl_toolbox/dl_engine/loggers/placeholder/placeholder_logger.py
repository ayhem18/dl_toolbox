"""Filesystem dump logger. Writes text/json under the run directory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import (
    STREAM_FIT,
    Logger,
    LoggerError,
    parse_log_stream,
    require_logger_open,
    table_as_json,
)
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager


class PlaceholderLogger(Logger):
    def __init__(self, *, manager: PlaceholderManager, run_name: str) -> None:
        self._manager = manager
        self.run_name = parse_non_empty_str(run_name, label="run_name")
        self._run_dir: Path | None = None
        self._open = False

    def setup(self, *, stream: str = STREAM_FIT) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        parse_log_stream(stream)
        self._manager.prepare()
        run_dir = self._manager.run_dir(self.run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._open = True

    def log_params(self, params: Mapping[str, str]) -> None:
        del params
        require_logger_open(self._open)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        del tags
        require_logger_open(self._open)

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        del values, step
        require_logger_open(self._open)

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        del image
        path = self._payload_path(tag, step, ".image.txt")
        path.write_text(f"tag={tag} step={int(step)}", encoding="utf-8")

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        path = self._payload_path(tag, step, ".table.json")
        path.write_text(table_as_json(rows) + "\n", encoding="utf-8")

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        path = self._payload_path(tag, step, ".txt")
        path.write_text(str(text), encoding="utf-8")

    def close(self) -> None:
        self._open = False

    def _payload_path(self, tag: str, step: int, suffix: str) -> Path:
        require_logger_open(self._open)
        if self._run_dir is None:
            raise LoggerError("logger.setup() must run before logging")
        name = parse_non_empty_str(tag, label="tag")
        parts = [part for part in name.split("/") if part]
        if not parts:
            raise LoggerError("tag must contain a non-empty path segment")
        filename = f"{parts[-1]}_step{int(step)}{suffix}"
        directory = self._run_dir.joinpath(*parts[:-1]) if len(parts) > 1 else self._run_dir
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename
