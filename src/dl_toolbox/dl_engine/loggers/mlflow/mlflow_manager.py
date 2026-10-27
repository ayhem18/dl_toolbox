"""MLflow store layout. Import mlflow only here. Logging lives on the logger."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.loggers.base_logger import LoggerError, import_extra
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager

_EXTRA = "mlflow"
_PARAM_KEY_MAX = 250
_PARAM_VALUE_MAX = 6000
_TAG_KEY_MAX = 250
_TAG_VALUE_MAX = 5000
_PARAM_CHUNK = 100
_RUN_ID_DIRNAME = "run_ids"
_RUN_ID_KEYS = ("run_id",)


class MlflowSetupError(ValueError):
    """The MLflow store could not be opened."""


class MlflowManager(LoggingManager):
    """One MLflow tracking store. The logger owns the run identity."""

    def __init__(
        self,
        *,
        db_path: Path,
        artifact_root: Path,
        timeout_seconds: float,
    ) -> None:
        self._db_path = parse_absolute_path(str(db_path), label="db_path")
        self._artifact_root = parse_absolute_path(str(artifact_root), label="artifact_root")
        timeout = float(timeout_seconds)
        if timeout <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
        self._timeout_seconds = timeout
        posix = self._db_path.as_posix()
        timeout_token: int | float = (
            int(timeout) if timeout == int(timeout) else timeout
        )
        self._backend_store_uri = f"sqlite:///{posix}?timeout={timeout_token}"
        self._artifact_store_uri = self._artifact_root.as_posix()

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

    def prepare(self) -> None:
        self.client().search_experiments(max_results=1)

    def client(self):
        """Tracking client pinned to this store. Does not open a run."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow = import_mlflow()
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self._backend_store_uri)
        return MlflowClient(tracking_uri=self._backend_store_uri)

    def open_experiment(self, *, experiment: str) -> str:
        experiment_name = parse_non_empty_str(experiment, label="experiment")
        client = self.client()
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            client.create_experiment(
                experiment_name, artifact_location=self._artifact_store_uri
            )
        experiment_obj = client.get_experiment_by_name(experiment_name)
        if experiment_obj is None:
            raise MlflowSetupError(f"failed to open experiment {experiment_name!r}")
        return str(experiment_obj.experiment_id)

    def run_id_path(self, *, experiment: str, run_name: str) -> Path:
        experiment_name = parse_non_empty_str(experiment, label="experiment")
        name = parse_non_empty_str(run_name, label="run_name")
        return self._db_path.parent / _RUN_ID_DIRNAME / experiment_name / f"{name}.json"

    def record_run_id(self, *, experiment: str, run_name: str, run_id: str) -> Path:
        pointer = self.run_id_path(experiment=experiment, run_name=run_name)
        return write_json(
            pointer,
            {"run_id": parse_non_empty_str(run_id, label="run_id")},
        )

    def load_run_id(self, *, experiment: str, run_name: str) -> str:
        pointer = self.run_id_path(experiment=experiment, run_name=run_name)
        if not pointer.is_file():
            raise LoggerError(
                f"no MLflow run_id recorded for experiment={experiment!r} "
                f"run_name={run_name!r}; fit logger.setup has not run"
            )
        payload = read_json(pointer)
        require_exact_keys(payload, _RUN_ID_KEYS, label="mlflow run_id pointer")
        return parse_non_empty_str(payload["run_id"], label="run_id")


def import_mlflow():
    return import_extra("mlflow", extra=_EXTRA)


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


def param_bounds() -> tuple[int, int, int]:
    return _PARAM_KEY_MAX, _PARAM_VALUE_MAX, _PARAM_CHUNK


def tag_bounds() -> tuple[int, int]:
    return _TAG_KEY_MAX, _TAG_VALUE_MAX
