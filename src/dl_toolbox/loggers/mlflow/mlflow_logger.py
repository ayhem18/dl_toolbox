"""MLflow training logger. Persistence lives on the manager."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.loggers.base_logger import Logger
from dl_toolbox.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.parsing.parse_str import parse_non_empty_str


class MlflowLogger(Logger):
    """One MLflow run. The caller owns setup and close."""

    def __init__(
        self,
        *,
        manager: MlflowManager,
        experiment: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> None:
        self._manager = manager
        self.experiment = parse_non_empty_str(experiment, label="experiment")
        self.run_name = parse_non_empty_str(run_name, label="run_name")
        self._tags = dict(tags)

    @property
    def run_id(self) -> str | None:
        return self._manager.run_id

    @property
    def tracking_uri(self) -> str | None:
        return self._manager.backend_store_uri

    def setup(self) -> None:
        self._manager.open_run(
            experiment=self.experiment,
            run_name=self.run_name,
            tags=self._tags,
        )

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
