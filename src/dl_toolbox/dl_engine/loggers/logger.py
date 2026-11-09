"""Logger: setup, log_params, log_tags, log_scalars, log_image, log_text, log_table, close."""



from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of


class LogStream:
    FIT = "fit"
    INSPECT = "inspect"


class LogExistence:
    CREATE = "create"
    REQUIRE = "require"


class LoggerError(ValueError):
    """Logger is used before setup, after close, or with an illegal argument."""


# TODO move this to the .json_io contract
def table_as_json(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [dict(row) for row in rows]
    return json.dumps(payload, indent=2, ensure_ascii=True, default=str)


class Logger(ABC):
    def __init__(self) -> None:
        self._open = False

    @abstractmethod
    def setup(self, *, stream: str, existence: str) -> None:
        """
        1. verify the backend is ready 
        2. connect the process to the run in the backend: the logic depends on the `stream` and `existence` parameters.
        """

    @abstractmethod
    def log_params(self, params: Mapping[str, str]) -> None:
        """record run-level string maps, no `step`."""

    @abstractmethod
    def log_tags(self, tags: Mapping[str, str]) -> None:
        """record run tags. run-level string maps, no `step`."""

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
        """Record a table at step."""

    @abstractmethod
    def log_text(self, tag: str, text: str, *, step: int) -> None:
        """Record a text blob at step."""

    @abstractmethod
    def close(self) -> None:
        """Unbind this process. Idempotent."""


    ############################################ Parsing Helpers ############################################
    def _parse_log_stream(self, *, stream: object) -> str:
        return parse_one_of(
            stream,
            allowed=(LogStream.FIT, LogStream.INSPECT),
            label="stream",
            error_type=LoggerError,
            error_message=one_of_message,
        )


    def _parse_log_existence(self, *, existence: object) -> str:
        return parse_one_of(
            existence,
            allowed=(LogExistence.CREATE, LogExistence.REQUIRE),
            label="existence",
            error_type=LoggerError,
            error_message=one_of_message,
        )


    ############################################ State Validation helpers ############################################
    def _require_logger_open(self) -> None:
        if not self._open:
            raise LoggerError("logger.setup() must run before logging")
