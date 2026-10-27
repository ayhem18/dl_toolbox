"""Run-level logger_config. Fields go into the existing Logger constructor."""

from __future__ import annotations

from collections.abc import Mapping

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys


def _string_tags(raw: object, *, label: str) -> dict[str, str]:
    obj = parse_mapping(raw, label=label)
    keys = list(obj)
    n = len(keys)
    tags: dict[str, str] = {}
    for i in range(n):
        key = keys[i]
        name = parse_non_empty_str(key, label=f"{label} keys")
        tags[name] = parse_non_empty_str(obj[key], label=f"{label}.{name}")
    return tags


class MlflowRunLoggerConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="mlflow logger_config")
        require_exact_keys(obj, ("experiment", "run_name", "tags"), label="mlflow logger_config")
        self._experiment = parse_non_empty_str(obj["experiment"], label="experiment")
        self._run_name = parse_non_empty_str(obj["run_name"], label="run_name")
        self._tags = _string_tags(obj["tags"], label="tags")

    @property
    def experiment(self) -> str:
        return self._experiment

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def tags(self) -> dict[str, str]:
        return dict(self._tags)


class WandbRunLoggerConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="wandb logger_config")
        require_exact_keys(obj, ("project", "run_name", "tags"), label="wandb logger_config")
        self._project = parse_non_empty_str(obj["project"], label="project")
        self._run_name = parse_non_empty_str(obj["run_name"], label="run_name")
        self._tags = _string_tags(obj["tags"], label="tags")

    @property
    def project(self) -> str:
        return self._project

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def tags(self) -> dict[str, str]:
        return dict(self._tags)


class TensorBoardRunLoggerConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="tensorboard logger_config")
        require_exact_keys(obj, ("run_name",), label="tensorboard logger_config")
        self._run_name = parse_non_empty_str(obj["run_name"], label="run_name")

    @property
    def run_name(self) -> str:
        return self._run_name


class PlaceholderRunLoggerConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="placeholder logger_config")
        require_exact_keys(obj, ("run_name",), label="placeholder logger_config")
        self._run_name = parse_non_empty_str(obj["run_name"], label="run_name")

    @property
    def run_name(self) -> str:
        return self._run_name


def parse_run_logger_config(backend: str, payload: Mapping[str, object]):
    obj = parse_mapping(payload, label="logger_config")
    if backend == "mlflow":
        return MlflowRunLoggerConfigParser(obj)
    if backend == "wandb":
        return WandbRunLoggerConfigParser(obj)
    if backend == "tensorboard":
        return TensorBoardRunLoggerConfigParser(obj)
    if backend == "placeholder":
        return PlaceholderRunLoggerConfigParser(obj)
    raise ValueError(f"unknown tracking backend {backend!r}")
