"""Backend-agnostic run logger. Training code never imports a vendor SDK."""

from __future__ import annotations

import importlib
import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path


class LoggerError(ValueError):
    """Logger is used before setup or after close."""


def flatten_mapping(payload: dict, *, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts to dotted string params for any backend."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    keys = list(payload)
    n = len(keys)
    params: dict[str, str] = {}
    for i in range(n):
        key = keys[i]
        if not isinstance(key, str) or not key:
            raise ValueError(f"mapping key must be a non-empty str, got {key!r}")
        name = f"{prefix}{key}" if prefix else key
        value = payload[key]
        if isinstance(value, dict):
            nested = flatten_mapping(value, prefix=f"{name}.")
            nested_keys = list(nested)
            n_nested = len(nested_keys)
            for j in range(n_nested):
                nested_key = nested_keys[j]
                params[nested_key] = nested[nested_key]
            continue
        if isinstance(value, (list, tuple)):
            params[name] = json.dumps(list(value), ensure_ascii=True)
            continue
        if isinstance(value, bool):
            params[name] = "true" if value else "false"
            continue
        if value is None:
            params[name] = "null"
            continue
        params[name] = str(value)
    return params


def import_extra(package: str, extra: str):
    """Import an optional backend and explain which extra installs it."""
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise ImportError(
            f"{package} is required for this logger and is not installed. "
            "dl_toolbox keeps logging backends as separate extras so you "
            "only install the one you use:\n"
            f"  uv add 'dl-toolbox[{extra}]'\n"
            f"  pip install 'dl-toolbox[{extra}]'"
        ) from exc


def require_logger_open(is_open: bool) -> None:
    if not is_open:
        raise LoggerError("logger.setup() must run before logging")


def copy_local_artifact(
    path: Path | str,
    log_dir: Path,
    *,
    dest: str | None,
) -> Path:
    source = Path(path)
    if not source.is_file():
        raise LoggerError(f"artifact is not a file: {source}")
    if dest is None:
        target_dir = log_dir
    else:
        if not isinstance(dest, str) or not dest.strip():
            raise LoggerError("dest must be a non-empty str when set")
        target_dir = log_dir / dest.strip()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


class Logger(ABC):
    @abstractmethod
    def setup(self) -> None:
        """Validate the backend and open a run. Fail before training work."""

    @abstractmethod
    def log_params(self, params: Mapping[str, str]) -> None:
        """Record flattened string parameters."""

    @abstractmethod
    def log_tags(self, tags: Mapping[str, str]) -> None:
        """Record run tags. Backends without tags may store them as text."""

    @abstractmethod
    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        """Record every scalar for one step in a single backend call."""

    def log_scalar(self, tag: str, value: float, *, step: int) -> None:
        self.log_scalars({tag: float(value)}, step=step)

    @abstractmethod
    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        """Upload or copy a local file into the run."""

    @abstractmethod
    def close(self) -> None:
        """Release the run. Idempotent."""

    @property
    def run_id(self) -> str | None:
        return None

    @property
    def tracking_uri(self) -> str | None:
        return None
