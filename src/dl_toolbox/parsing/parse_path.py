"""Absolute paths from JSON strings. Resolution is caller-owned."""

from pathlib import Path

from dl_toolbox.parsing.parse_str import parse_non_empty_str


def parse_absolute_path(raw: object, *, label: str) -> Path:
    path = Path(parse_non_empty_str(raw, label=label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path, got {path}")
    return path
