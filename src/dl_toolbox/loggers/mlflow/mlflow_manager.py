"""MLflow store layout and persistence. Import mlflow only here."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.loggers.base_logger import LoggerError, import_extra, require_logger_open
from dl_toolbox.loggers.mlflow.mlflow_parser import MlflowParser
from dl_toolbox.parsing.parse_str import parse_non_empty_str

_EXTRA = "mlflow"
# MLflow tracking API limits (key/value length) and log_params batch size.
_PARAM_KEY_MAX = 250
_PARAM_VALUE_MAX = 6000
_TAG_KEY_MAX = 250
_TAG_VALUE_MAX = 5000
_PARAM_CHUNK = 100


class MlflowSetupError(ValueError):
    """The MLflow store could not be opened."""


class MlflowManager:
    """Persistence for one MLflow tracking store. The logger owns the run identity."""

    def __init__(self, payload: dict) -> None:
        parsed = MlflowParser(payload)
        self._backend_store_uri = parsed.backend_store_uri
        self._artifact_store_uri = parsed.artifact_store_uri
        self._db_path = parsed.db_path
        self._artifact_root = parsed.artifact_root
        self._run_id: str | None = None
        self._open = False

    @property
    def backend_store_uri(self) -> str:
        return self._backend_store_uri

    @property
    def artifact_store_uri(self) -> str:
        return self._artifact_store_uri

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def setup(self) -> None:
        """Create store directories, set the tracking URI, and ping the backend."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow = _import_mlflow()
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self._backend_store_uri)
        MlflowClient().search_experiments(max_results=1)

    def open_experiment(self, *, experiment: str) -> str:
        experiment_name = parse_non_empty_str(experiment, label="experiment")
        self.setup()
        mlflow = _import_mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            client.create_experiment(
                experiment_name, artifact_location=self._artifact_store_uri
            )
        mlflow.set_experiment(experiment_name)
        experiment_obj = client.get_experiment_by_name(experiment_name)
        if experiment_obj is None:
            raise MlflowSetupError(f"failed to open experiment {experiment_name!r}")
        return str(experiment_obj.experiment_id)

    def open_run(
        self,
        *,
        experiment: str,
        run_name: str,
        tags: Mapping[str, str],
    ) -> str:
        if self._open:
            raise LoggerError("logger is already open")
        name = parse_non_empty_str(run_name, label="run_name")
        self.open_experiment(experiment=experiment)
        mlflow = _import_mlflow()
        run = mlflow.start_run(run_name=name)
        self._run_id = str(run.info.run_id)
        self._open = True
        mapping = _bounded_mapping(
            tags,
            "tags",
            key_max=_TAG_KEY_MAX,
            value_max=_TAG_VALUE_MAX,
        )
        if mapping:
            mlflow.set_tags(mapping)
        return self._run_id

    def log_params(self, params: Mapping[str, str]) -> None:
        require_logger_open(self._open)
        mlflow = _import_mlflow()
        mapping = _bounded_mapping(
            params,
            "params",
            key_max=_PARAM_KEY_MAX,
            value_max=_PARAM_VALUE_MAX,
        )
        keys = list(mapping)
        n = len(keys)
        if n == 0:
            return
        n_chunks = (n + _PARAM_CHUNK - 1) // _PARAM_CHUNK
        for chunk_i in range(n_chunks):
            start = chunk_i * _PARAM_CHUNK
            end = min(start + _PARAM_CHUNK, n)
            chunk: dict[str, str] = {}
            for i in range(start, end):
                key = keys[i]
                chunk[key] = mapping[key]
            mlflow.log_params(chunk)

    def log_tags(self, tags: Mapping[str, str]) -> None:
        require_logger_open(self._open)
        mlflow = _import_mlflow()
        mapping = _bounded_mapping(
            tags,
            "tags",
            key_max=_TAG_KEY_MAX,
            value_max=_TAG_VALUE_MAX,
        )
        if mapping:
            mlflow.set_tags(mapping)

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        require_logger_open(self._open)
        mlflow = _import_mlflow()
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
        mlflow.log_metrics(metrics, step=int(step))

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        require_logger_open(self._open)
        mlflow = _import_mlflow()
        source = Path(path)
        if not source.is_file():
            raise LoggerError(f"artifact is not a file: {source}")
        if dest is None:
            mlflow.log_artifact(str(source))
            return
        dest_name = parse_non_empty_str(dest, label="dest")
        mlflow.log_artifact(str(source), artifact_path=dest_name)

    def close_run(self) -> None:
        if not self._open:
            return
        mlflow = _import_mlflow()
        mlflow.end_run()
        self._open = False


def _import_mlflow():
    return import_extra("mlflow", extra=_EXTRA)


def _string_mapping(payload: Mapping[str, str], label: str) -> dict[str, str]:
    keys = list(payload)
    n = len(keys)
    mapped: dict[str, str] = {}
    for i in range(n):
        key = keys[i]
        if not isinstance(key, str) or not key:
            raise LoggerError(f"{label} key must be a non-empty str, got {key!r}")
        mapped[key] = str(payload[key])
    return mapped


def _bounded_mapping(
    payload: Mapping[str, str],
    label: str,
    *,
    key_max: int,
    value_max: int,
) -> dict[str, str]:
    mapped = _string_mapping(payload, label)
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
