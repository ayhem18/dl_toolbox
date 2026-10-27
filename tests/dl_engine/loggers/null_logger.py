"""Records Logger calls. Used in unit tests, not a production backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dl_toolbox.dl_engine.loggers.base_logger import STREAM_FIT, Logger, parse_log_stream


class NullLogger(Logger):
    def __init__(self) -> None:
        self.setup_calls = 0
        self.setup_stream: str | None = None
        self.closed = False
        self.params: list[dict[str, str]] = []
        self.tags: list[dict[str, str]] = []
        self.scalars: list[tuple[dict[str, float], int]] = []
        self.images: list[tuple[str, object, int]] = []
        self.tables: list[tuple[str, list[dict[str, Any]], int]] = []
        self.texts: list[tuple[str, str, int]] = []

    def setup(self, *, stream: str = STREAM_FIT) -> None:
        self.setup_stream = parse_log_stream(stream)
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

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        self.images.append((tag, image, int(step)))

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.tables.append((tag, [dict(row) for row in rows], int(step)))

    def log_text(self, tag: str, text: str, *, step: int) -> None:
        self.texts.append((tag, str(text), int(step)))

    def close(self) -> None:
        self.closed = True
