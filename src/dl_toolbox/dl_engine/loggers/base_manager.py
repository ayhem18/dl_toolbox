"""Project-level logging store. Inspect publish lives here, not on Logger.setup."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import Logger, LoggerError


class LoggingManager(ABC):
    @abstractmethod
    def publish_directory(
        self,
        local_dir: Path,
        *,
        dest: str,
        logger: Logger,
    ) -> None:
        """Copy a committed local folder into the store. Must not open a live run."""
        raise NotImplementedError


def logger_run_name(logger: Logger) -> str:
    name = getattr(logger, "run_name", None)
    if not isinstance(name, str) or not name.strip():
        raise LoggerError("logger is missing run_name")
    return parse_non_empty_str(name, label="run_name")


def publish_run_tree(
    local_dir: Path,
    *,
    dest: str,
    parent: Path,
    run_name: str,
) -> Path:
    source = Path(local_dir)
    if not source.is_dir():
        raise LoggerError(f"publish source is not a folder: {source}")
    dest_name = parse_non_empty_str(dest, label="dest")
    target = parent / run_name / dest_name
    if target.exists():
        raise FileExistsError(f"inspection vendor folder already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target
