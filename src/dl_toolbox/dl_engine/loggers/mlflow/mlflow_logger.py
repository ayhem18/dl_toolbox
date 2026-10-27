"""MLflow training logger. Persistence helpers live on the manager."""

from __future__ import annotations

import time
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
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import (
    MlflowManager,
    bounded_mapping,
    param_bounds,
    tag_bounds,
)


class MlflowLogger(Logger):
    """One MLflow run. ``experiment`` is a Logger field, not the toolbox project name."""

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
        self._run_id: str | None = None
        self._open = False
        self._owns_run = False

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def tracking_uri(self) -> str | None:
        return self._manager.backend_store_uri

    def setup(self, *, stream: str = STREAM_FIT) -> None:
        if self._open:
            raise LoggerError("logger is already open")
        parsed = parse_log_stream(stream)
        if parsed == STREAM_INSPECT:
            self._attach_inspect()
            return
        self._open_fit()

    def log_params(self, params: Mapping[str, str]) -> None:
        client, run_id = self._client_run()
        from mlflow.entities import Param

        key_max, value_max, chunk = param_bounds()
        mapping = bounded_mapping(
            params, "params", key_max=key_max, value_max=value_max
        )
        keys = list(mapping)
        n = len(keys)
        if n == 0:
            return
        items: list = [None] * n
        for i in range(n):
            key = keys[i]
            items[i] = Param(key, mapping[key])
        n_chunks = (n + chunk - 1) // chunk
        for chunk_i in range(n_chunks):
            start = chunk_i * chunk
            end = min(start + chunk, n)
            client.log_batch(run_id, params=items[start:end], synchronous=True)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        client, run_id = self._client_run()
        from mlflow.entities import RunTag

        key_max, value_max = tag_bounds()
        mapping = bounded_mapping(tags, "tags", key_max=key_max, value_max=value_max)
        keys = list(mapping)
        n = len(keys)
        if n == 0:
            return
        items: list = [None] * n
        for i in range(n):
            key = keys[i]
            items[i] = RunTag(key, mapping[key])
        client.log_batch(run_id, tags=items, synchronous=True)

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        client, run_id = self._client_run()
        from mlflow.entities import Metric

        if not values:
            return
        keys = list(values)
        n = len(keys)
        timestamp = int(time.time() * 1000)
        step_i = int(step)
        items: list = [None] * n
        for i in range(n):
            key = keys[i]
            if not isinstance(key, str) or not key:
                raise LoggerError(f"metric name must be a non-empty str, got {key!r}")
            items[i] = Metric(key, float(values[key]), timestamp, step_i)
        client.log_batch(run_id, metrics=items, synchronous=True)

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        client, run_id = self._client_run()
        if not hasattr(client, "log_image"):
            raise LoggerError("this mlflow version has no log_image")
        name = parse_non_empty_str(tag, label="tag")
        client.log_image(
            run_id,
            _to_numpy_image(image),
            key=name,
            step=int(step),
            synchronous=True,
        )

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.log_text(tag, table_as_json(rows), step=step)

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        client, run_id = self._client_run()
        name = parse_non_empty_str(tag, label="tag")
        filename = _artifact_filename(name, step, ".txt")
        client.log_text(run_id, str(text), filename)

    def close(self) -> None:
        if not self._open:
            return
        try:
            if self._owns_run and self._run_id is not None:
                self._manager.client().set_terminated(self._run_id)
        finally:
            self._owns_run = False
            self._open = False

    def _open_fit(self) -> None:
        experiment_id = self._manager.open_experiment(experiment=self.experiment)
        client = self._manager.client()
        key_max, value_max = tag_bounds()
        mapping = bounded_mapping(
            self._tags, "tags", key_max=key_max, value_max=value_max
        )
        run = client.create_run(
            experiment_id,
            tags=mapping if mapping else None,
            run_name=self.run_name,
        )
        run_id = str(run.info.run_id)
        self._manager.record_run_id(
            experiment=self.experiment,
            run_name=self.run_name,
            run_id=run_id,
        )
        self._run_id = run_id
        self._owns_run = True
        self._open = True

    def _attach_inspect(self) -> None:
        client = self._manager.client()
        run_id = self._manager.load_run_id(
            experiment=self.experiment,
            run_name=self.run_name,
        )
        client.get_run(run_id)
        self._run_id = run_id
        self._owns_run = False
        self._open = True

    def _client_run(self):
        require_logger_open(self._open)
        run_id = self._run_id
        if run_id is None:
            raise LoggerError("logger.setup() must run before logging")
        return self._manager.client(), run_id


def _artifact_filename(tag: str, step: int, suffix: str) -> str:
    raw_parts = tag.split("/")
    n = len(raw_parts)
    kept: list[str | None] = [None] * n
    k = 0
    for i in range(n):
        part = raw_parts[i]
        if part in ("", ".", ".."):
            continue
        kept[k] = part
        k += 1
    if k == 0:
        raise LoggerError("tag must contain a non-empty path segment")
    filename = f"{kept[k - 1]}_step{int(step)}{suffix}"
    if k == 1:
        return filename
    prefix_parts = [""] * (k - 1)
    for i in range(k - 1):
        prefix_parts[i] = str(kept[i])
    return "/".join(prefix_parts) + "/" + filename


def _to_numpy_image(image: object):
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    return array
