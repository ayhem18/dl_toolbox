"""MlflowLogger. Entity is the tracking instance and the runs in it."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dl_toolbox.base.packaging import require
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    Logger,
    LoggerError,
    table_as_json,
    parse_log_stream,
    parse_log_existence,
)

# TYPE_CHECKING is True for the language server and False at runtime.
# Importing this module therefore does not load mlflow and does not fail
# when the mlflow extra is missing. Construction still calls
# _import_mlflow_client() so a missing extra errors at init, not setup.
if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient


def _import_mlflow_client() -> type[MlflowClient]:
    require(package="mlflow")
    from mlflow.tracking import MlflowClient

    return MlflowClient


class MlflowLogger(Logger):
    def __init__(
        self,
        *,
        backend_store_uri: str,
        artifact_store_uri: str,
        experiment_name: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> None:
        # sets up the self._open flag
        super().__init__()

        self._backend_store_uri = parse_non_empty_str(
            backend_store_uri, label="backend_store_uri"
        )

        self._artifact_store_uri = parse_non_empty_str(
            artifact_store_uri, label="artifact_store_uri"
        )

        self._experiment_name = parse_non_empty_str(
            experiment_name, label="experiment_name"
        )

        self._run_name = parse_non_empty_str(run_name, label="run_name")

        self._tags = dict(tags)
        self._client_cls = _import_mlflow_client()
        self._client: MlflowClient | None = None
        self._run_id: str | None = None

    @property
    def run_id(self) -> str:
        self._require_logger_open()
        return self._run_id


    def setup(self, *, stream: str, existence: str) -> None:
        if self._open:
            raise LoggerError("logger is already open")

        parsed_stream = parse_log_stream(stream=stream)
        parsed_existence = parse_log_existence(existence=existence)
        self._client = self._build_client()
        if parsed_existence == LogExistence.CREATE:
            self._setup_create()
            return
        self._setup_require()

    def log_params(self, params: Mapping[str, str]) -> None:
        self._require_logger_open()
        from mlflow.entities import Param

        mapping = bounded_mapping(params, "params", key_max=250, value_max=6000)
        keys = list(mapping)
        n = len(keys)
        if n == 0:
            return
        items: list = [None] * n
        for i in range(n):
            key = keys[i]
            items[i] = Param(key, mapping[key])
        chunk = 100
        n_chunks = (n + chunk - 1) // chunk
        for chunk_i in range(n_chunks):
            start = chunk_i * chunk
            end = min(start + chunk, n)
            self._client.log_batch(
                self._run_id, params=items[start:end], synchronous=True
            )

    def log_tags(self, tags: Mapping[str, str]) -> None:
        self._require_logger_open()
        from mlflow.entities import RunTag

        mapping = bounded_mapping(tags, "tags", key_max=250, value_max=5000)
        keys = list(mapping)
        n = len(keys)
        if n == 0:
            return
        items: list = [None] * n
        for i in range(n):
            key = keys[i]
            items[i] = RunTag(key, mapping[key])
        self._client.log_batch(self._run_id, tags=items, synchronous=True)

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        self._require_logger_open()
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
        self._client.log_batch(self._run_id, metrics=items, synchronous=True)

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        self._require_logger_open()
        if not hasattr(self._client, "log_image"):
            raise LoggerError("this mlflow version has no log_image")
        name = parse_non_empty_str(tag, label="tag")
        self._client.log_image(
            self._run_id,
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
        self._require_logger_open()
        name = parse_non_empty_str(tag, label="tag")
        filename = _artifact_filename(name, step, ".txt")
        self._client.log_text(self._run_id, str(text), filename)

    def log_artifact(self, *, local_path: Path, artifact_path: str) -> None:
        self._require_logger_open()
        path = parse_absolute_path(str(local_path), label="local_path")
        if not path.is_file():
            raise LoggerError(f"local_path must be an existing file, got {path}")
        dest = parse_non_empty_str(artifact_path, label="artifact_path")
        self._client.log_artifact(self._run_id, str(path), dest)

    def log_model(self, *, model: object, artifact_path: str) -> None:
        self._require_logger_open()
        name = parse_non_empty_str(artifact_path, label="artifact_path")
        mlflow = require(package="mlflow")
        mlflow.set_tracking_uri(self._backend_store_uri)
        mlflow.pytorch.log_model(
            model,
            name=name,
            run_id=self._run_id,
            pip_requirements=["torch"],
            serialization_format="pickle",
        )

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._client = None
        self._run_id = None

    def _setup_create(self) -> None:
        experiment_id = self._resolve_experiment_create()
        existing = self._lookup_run(experiment_id=experiment_id)
        if existing is not None:
            raise LoggerError(
                f"MLflow run {self._run_name!r} already exists in experiment "
                f"{self._experiment_name!r}"
            )
        mapping = bounded_mapping(self._tags, "tags", key_max=250, value_max=5000)
        run = self._client.create_run(
            experiment_id,
            tags=mapping if mapping else None,
            run_name=self._run_name,
        )
        self._run_id = str(run.info.run_id)
        self._open = True


    def _setup_require(self) -> None:
        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            raise LoggerError(
                f"MLflow experiment {self._experiment_name!r} does not exist"
            )
        run = self._lookup_run(
            experiment_id=str(experiment.experiment_id)
        )

        if run is None:
            raise LoggerError(
                f"MLflow run {self._run_name!r} does not exist in experiment "
                f"{self._experiment_name!r}"
            )

        if run.info.lifecycle_stage == "deleted":
            raise LoggerError(
                f"MLflow run {self._run_name!r} is deleted"
            )

        if run.info.status in ("FINISHED", "FAILED", "KILLED"):
            raise LoggerError(
                f"MLflow run {self._run_name!r} is terminal "
                f"(status={run.info.status})"
            )

        self._run_id = str(run.info.run_id)
        self._open = True

    def _resolve_experiment_create(self) -> str:
        existing = self._client.get_experiment_by_name(self._experiment_name)
        if existing is None:
            experiment_id = self._client.create_experiment(
                self._experiment_name,
                artifact_location=self._artifact_store_uri,
            )
            return str(experiment_id)
        return str(existing.experiment_id)

    def _lookup_run(self, *, experiment_id: str):
        from mlflow.entities import ViewType

        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                "attributes.run_name = "
                + _sql_quote(self._run_name)
                + " AND attributes.status = 'RUNNING'"
            ),
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=2,
        )
        n = len(runs)
        if n == 0:
            return None
        if n > 1:
            raise LoggerError(
                f"multiple MLflow runs named {self._run_name!r} in experiment "
                f"{self._experiment_name!r}"
            )
        return runs[0]

    def _build_client(self) -> MlflowClient:
        mlflow = require(package="mlflow")
        self._prepare_local_stores()
        mlflow.set_tracking_uri(self._backend_store_uri)
        return self._client_cls(tracking_uri=self._backend_store_uri)

    def _prepare_local_stores(self) -> None:
        artifact = Path(self._artifact_store_uri)
        if artifact.drive or artifact.is_absolute():
            artifact.mkdir(parents=True, exist_ok=True)
        uri = self._backend_store_uri
        prefix = "sqlite:///"
        if not uri.startswith(prefix):
            return
        rest = uri[len(prefix) :].split("?", 1)[0]
        db_path = Path(rest)
        if db_path.drive or db_path.is_absolute():
            db_path.parent.mkdir(parents=True, exist_ok=True)


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def string_mapping(payload: Mapping[str, str], label: str) -> dict[str, str]:
    keys = list(payload)
    n = len(keys)
    mapped: dict[str, str] = {}
    for i in range(n):
        key = keys[i]
        if not isinstance(key, str) or not key:
            raise LoggerError(f"{label} key must be a non-empty str, got {key!r}")
        mapped[key] = str(payload[key])
    return mapped


def bounded_mapping(
    payload: Mapping[str, str],
    label: str,
    *,
    key_max: int,
    value_max: int,
) -> dict[str, str]:
    mapped = string_mapping(payload, label)
    keys = list(mapped)
    n = len(keys)
    for i in range(n):
        key = keys[i]
        value = mapped[key]
        if len(key) > key_max:
            raise LoggerError(f"{label} key exceeds {key_max} characters: {key!r}")
        if len(value) > value_max:
            raise LoggerError(
                f"{label} value for {key!r} exceeds {value_max} characters"
            )
    return mapped


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
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[-1] not in (1, 3, 4)
    ):
        array = np.transpose(array, (1, 2, 0))
    return array
