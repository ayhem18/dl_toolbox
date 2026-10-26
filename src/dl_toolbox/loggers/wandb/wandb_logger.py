"""W&B training logger. Persistence lives on the manager."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.loggers.base_logger import Logger
from dl_toolbox.loggers.wandb.wandb_manager import WandbManager
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str


class WandbLogger(Logger):
    def __init__(
        self,
        *,
        manager: WandbManager,
        run_name: str,
        tags: Mapping[str, str],
    ) -> None:
        self._manager = manager
        self.run_name = parse_non_empty_str(run_name, label="run_name")
        self._tags = dict(tags)

    def setup(self) -> None:
        self._manager.open_run(run_name=self.run_name, tags=self._tags)

    def log_params(self, params: Mapping[str, str]) -> None:
        self._manager.log_params(params)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        self._manager.log_tags(tags)

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        self._manager.log_metrics(values, step=step)

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        self._manager.log_artifact(path, dest=dest)

    def close(self) -> None:
        self._manager.close_run()
