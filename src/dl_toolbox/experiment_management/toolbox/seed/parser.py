"""Pass-through seed_config. Integer goes to seed_everything."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_non_negative_int
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.training.config import SeedConfig

_KEYS = ("seed",)


class SeedConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="seed_config")
        require_exact_keys(obj, _KEYS, label="seed_config")
        self._config = SeedConfig(
            seed=parse_non_negative_int(obj["seed"], label="seed")
        )

    @property
    def seed_config(self) -> SeedConfig:
        return self._config
