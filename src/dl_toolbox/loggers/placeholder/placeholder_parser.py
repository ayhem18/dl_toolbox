"""Validate a placeholder setup dict and expose the log directory."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_TOP_KEYS = ("log_dir",)


class PlaceholderParseError(ValueError):
    """Invalid placeholder setup dictionary."""


class PlaceholderParser:
    def __init__(self, payload: dict) -> None:
        try:
            obj = parse_mapping(payload, label="placeholder setup")
            require_exact_keys(obj, _TOP_KEYS, label="placeholder setup")
            self._log_dir = parse_absolute_path(obj["log_dir"], label="log_dir")
        except PlaceholderParseError:
            raise
        except (ValueError, TypeError) as exc:
            raise PlaceholderParseError(str(exc)) from exc

    @property
    def log_dir(self) -> Path:
        return self._log_dir
