"""Pass-through seed_config. Integer goes to seed_everything."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

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


# --- pydantic document ---
# Covers this parser completely: exact keys, int (not bool), seed >= 0.


class SeedConfigDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seed: Annotated[int, Field(ge=0)]


def seed_config_from_pydantic(payload: object) -> SeedConfig:
    document = SeedConfigDocument.model_validate(payload)
    return SeedConfig(seed=document.seed)
