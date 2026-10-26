"""Records Logger calls. Used in unit tests, not a production backend."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.loggers.base_logger import Logger


class NullLogger(Logger):
    def __init__(self) -> None:
        self.setup_calls = 0
        self.closed = False
        self.params: list[dict[str, str]] = []
        self.tags: list[dict[str, str]] = []
        self.scalars: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def setup(self) -> None:
        self.setup_calls += 1
        self.closed = False

    def log_params(self, params: Mapping[str, str]) -> None:
        self.params.append(dict(params))

    def log_tags(self, tags: Mapping[str, str]) -> None:
        self.tags.append(dict(tags))

    def log_scalars(self, values: Mapping[str, float], *, step: int) -> None:
        recorded: dict[str, float] = {}
        keys = list(values)
        n = len(keys)
        for i in range(n):
            key = keys[i]
            recorded[key] = float(values[key])
        self.scalars.append((recorded, int(step)))

    def log_artifact(self, path: Path | str, *, dest: str | None = None) -> None:
        self.artifacts.append((str(path), dest))

    def close(self) -> None:
        self.closed = True
