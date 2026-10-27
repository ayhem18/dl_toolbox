"""Backend-agnostic run logger. Training code never imports a vendor SDK."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


STREAM_FIT = "fit"
STREAM_INSPECT = "inspect"
_LOG_STREAMS = frozenset({STREAM_FIT, STREAM_INSPECT})


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


def parse_log_stream(stream: object) -> str:
    if not isinstance(stream, str) or stream not in _LOG_STREAMS:
        raise LoggerError(
            f"stream must be {STREAM_FIT!r} or {STREAM_INSPECT!r}, got {stream!r}"
        )
    return stream


def table_as_json(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [dict(row) for row in rows]
    return json.dumps(payload, indent=2, ensure_ascii=True, default=str)


class Logger(ABC):
    @abstractmethod
    def setup(self, *, stream: str = STREAM_FIT) -> None:
        """Open this process's writer or run. Inspect uses stream='inspect'."""

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
    def log_image(self, tag: str, image: object, *, step: int) -> None:
        """Record one image at step. HW, HWC, or CHW arrays."""

    @abstractmethod
    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        """Record a table at step. TensorBoard stores JSON text."""

    @abstractmethod
    def log_text(self, tag: str, text: str, *, step: int) -> None:
        """Record a text blob at step."""

    @abstractmethod
    def close(self) -> None:
        """Release the run. Idempotent."""

    @property
    def run_id(self) -> str | None:
        return None

    @property
    def tracking_uri(self) -> str | None:
        return None
