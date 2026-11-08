"""Downstream Logger: setup / log_* / close. Store URIs are not on these methods."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from dl_toolbox.base.parsing.parse_utils import parse_one_of


class LogStream:
    FIT = "fit"
    INSPECT = "inspect"


class LogExistence:
    CREATE = "create"
    REQUIRE = "require"


STREAM_FIT = LogStream.FIT
STREAM_INSPECT = LogStream.INSPECT
EXISTENCE_CREATE = LogExistence.CREATE
EXISTENCE_REQUIRE = LogExistence.REQUIRE


class LoggerError(ValueError):
    """Logger is used before setup, after close, or with an illegal argument."""


def require_logger_open(is_open: bool) -> None:
    if not is_open:
        raise LoggerError("logger.setup() must run before logging")


def parse_log_stream(stream: object) -> str:
    return parse_one_of(
        stream,
        allowed=(LogStream.FIT, LogStream.INSPECT),
        label="stream",
        error_type=LoggerError,
        error_message="{label} must be {allowed}, got {value!r}",
    )


def parse_log_existence(existence: object) -> str:
    return parse_one_of(
        existence,
        allowed=(LogExistence.CREATE, LogExistence.REQUIRE),
        label="existence",
        error_type=LoggerError,
        error_message="{label} must be {allowed}, got {value!r}",
    )


def table_as_json(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [dict(row) for row in rows]
    return json.dumps(payload, indent=2, ensure_ascii=True, default=str)


class Logger(ABC):
    @abstractmethod
    def setup(self, *, stream: str, existence: str) -> None:
        """Bind this process so log_* is legal. Both arguments required."""

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
        """Record one image at step."""

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
        """Unbind this process. Idempotent. Does not finish the vendor run."""
