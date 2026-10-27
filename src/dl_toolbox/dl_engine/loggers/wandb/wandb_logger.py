"""W&B training logger. Open-run state lives here."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import (
    STREAM_FIT,
    STREAM_INSPECT,
    Logger,
    LoggerError,
    parse_log_stream,
    require_logger_open,
    table_as_json,
)
from dl_toolbox.dl_engine.loggers.wandb.wandb_manager import WandbManager


class WandbLogger(Logger):
    def __init__(
        self,
        *,
        manager: WandbManager,
        project: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> None:
        self._manager = manager
        self.project = parse_non_empty_str(project, label="project")
        self.run_name = parse_non_empty_str(run_name, label="run_name")
        self._tags = dict(tags)
        self._open = False

    def setup(self, *, stream: str = STREAM_FIT) -> None:
        parsed = parse_log_stream(stream)
        if parsed == STREAM_INSPECT:
            self._manager.prepare()
            return
        if self._open:
            raise LoggerError("logger is already open")
        self._manager.prepare()
        wandb = self._manager.import_wandb()
        wandb.init(
            project=self.project,
            name=self.run_name,
            dir=str(self._manager.dir),
        )
        self._open = True
        if self._tags:
            prefixed = {f"tag.{key}": value for key, value in dict(self._tags).items()}
            wandb.config.update(prefixed)

    def log_params(self, params: Mapping[str, str]) -> None:
        require_logger_open(self._open)
        wandb = self._manager.import_wandb()
        wandb.config.update(dict(params))

    def log_tags(self, tags: Mapping[str, str]) -> None:
        require_logger_open(self._open)
        wandb = self._manager.import_wandb()
        prefixed = {f"tag.{key}": value for key, value in dict(tags).items()}
        wandb.config.update(prefixed)

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        require_logger_open(self._open)
        wandb = self._manager.import_wandb()
        if not values:
            return
        keys = list(values)
        n = len(keys)
        metrics: dict[str, float] = {}
        for i in range(n):
            key = keys[i]
            if not isinstance(key, str) or not key:
                raise LoggerError(f"metric name must be a non-empty str, got {key!r}")
            metrics[key] = float(values[key])
        wandb.log(metrics, step=int(step))

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        require_logger_open(self._open)
        wandb = self._manager.import_wandb()
        name = parse_non_empty_str(tag, label="tag")
        wandb.log({name: wandb.Image(image)}, step=int(step))

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.log_text(tag, table_as_json(rows), step=step)

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        require_logger_open(self._open)
        wandb = self._manager.import_wandb()
        name = parse_non_empty_str(tag, label="tag")
        wandb.log({name: str(text)}, step=int(step))

    def close(self) -> None:
        if not self._open:
            return
        wandb = self._manager.import_wandb()
        wandb.finish()
        self._open = False
