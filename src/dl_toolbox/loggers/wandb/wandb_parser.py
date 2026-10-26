"""Validate a W&B setup dict and expose project and local directory."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_path import parse_absolute_path
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_TOP_KEYS = ("project", "dir")


class WandbParseError(ValueError):
    """Invalid W&B setup dictionary."""


class WandbParser:
    def __init__(self, payload: dict) -> None:
        try:
            obj = parse_mapping(payload, label="wandb setup")
            require_exact_keys(obj, _TOP_KEYS, label="wandb setup")
            self._project = parse_non_empty_str(obj["project"], label="project")
            self._dir = parse_absolute_path(obj["dir"], label="dir")
        except WandbParseError:
            raise
        except (ValueError, TypeError) as exc:
            raise WandbParseError(str(exc)) from exc

    @property
    def project(self) -> str:
        return self._project

    @property
    def dir(self) -> Path:
        return self._dir
