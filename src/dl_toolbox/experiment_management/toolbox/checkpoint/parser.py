"""Pass-through checkpoint_config. Policy only: no dirpath, no monitor."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_bool, parse_int
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_KEYS = ("save_top_k", "save_last")


class CheckpointConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="checkpoint_config")
        require_exact_keys(obj, _KEYS, label="checkpoint_config")
        save_top_k = parse_int(obj["save_top_k"], label="save_top_k")
        if save_top_k < -1:
            raise ValueError(f"save_top_k must be >= -1, got {save_top_k}")
        self._save_top_k = save_top_k
        self._save_last = parse_bool(obj["save_last"], label="save_last")

    @property
    def save_top_k(self) -> int:
        return self._save_top_k

    @property
    def save_last(self) -> bool:
        return self._save_last
