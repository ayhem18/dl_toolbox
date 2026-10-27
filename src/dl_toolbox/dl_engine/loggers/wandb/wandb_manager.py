"""W&B local-dir persistence. Import wandb only here. Cloud project is identity."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.base_logger import (
    LoggerError,
    copy_local_artifact,
    import_extra,
    require_logger_open,
)
from dl_toolbox.base.parsing.parse_path import parse_absolute_path

_EXTRA = "wandb"


class WandbManager:
    def __init__(self, *, dir: Path) -> None:
        self._dir = parse_absolute_path(str(dir), label="dir")
        self._open = False

    @property
    def dir(self) -> Path:
        return self._dir

    @staticmethod
    def _import_wandb():
        return import_extra("wandb", extra=_EXTRA)

    def open_run(self, *, project: str, run_name: str, tags: Mapping[str, str]) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        cloud_project = parse_non_empty_str(project, label="project")
        name = parse_non_empty_str(run_name, label="run_name")
        self._dir.mkdir(parents=True, exist_ok=True)
        wandb = self._import_wandb()
        wandb.init(project=cloud_project, name=name, dir=str(self._dir))
        self._open = True
        if tags:
            prefixed = {f"tag.{key}": value for key, value in dict(tags).items()}
            wandb.config.update(prefixed)

    def log_params(self, params: Mapping[str, str]) -> None:
        require_logger_open(self._open)
        wandb = self._import_wandb()
        wandb.config.update(dict(params))

    def log_tags(self, tags: Mapping[str, str]) -> None:
        require_logger_open(self._open)
        wandb = self._import_wandb()
        prefixed = {f"tag.{key}": value for key, value in dict(tags).items()}
        wandb.config.update(prefixed)

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        require_logger_open(self._open)
        wandb = self._import_wandb()
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

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        require_logger_open(self._open)
        wandb = self._import_wandb()
        run = wandb.run
        if run is None or run.dir is None:
            raise LoggerError("wandb run is not open")
        copy_local_artifact(path, Path(run.dir), dest=dest)

    def close_run(self) -> None:
        if not self._open:
            return
        wandb = self._import_wandb()
        wandb.finish()
        self._open = False
