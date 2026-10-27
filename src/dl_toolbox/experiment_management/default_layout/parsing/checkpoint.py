"""Parse project_level_configs/checkpoint.json. Does not construct CheckpointManager."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_positive_number
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_KEYS = ("lock_timeout_seconds",)


class CheckpointProjectConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="checkpoint.json")
        require_exact_keys(obj, _KEYS, label="checkpoint.json")
        self._lock_timeout_seconds = parse_positive_number(
            obj["lock_timeout_seconds"], label="lock_timeout_seconds"
        )

    @property
    def lock_timeout_seconds(self) -> float:
        return self._lock_timeout_seconds
