"""Parse project_level_configs/identity.json. Layout rules live on HashManager."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import SHA256_HEX_LENGTH
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_KEYS = ("prefix_len", "human_readable_dirname", "hashes_dirname")


class IdentityConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="identity.json")
        require_exact_keys(obj, _KEYS, label="identity.json")
        manager = HashManager(
            prefix_len=obj["prefix_len"],
            human_readable_dirname=obj["human_readable_dirname"],
            hashes_dirname=obj["hashes_dirname"],
        )
        self._prefix_len = manager.prefix_len
        self._human_readable_dirname = manager.human_readable_dirname
        self._hashes_dirname = manager.hashes_dirname

    @property
    def prefix_len(self) -> int:
        return self._prefix_len

    @property
    def human_readable_dirname(self) -> str:
        return self._human_readable_dirname

    @property
    def hashes_dirname(self) -> str:
        return self._hashes_dirname


# --- pydantic document ---
# Exact keys, prefix_len range. Basename / not-a-hash-stem / distinct folders
# are HashManager.__init__.


class IdentityDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prefix_len: Annotated[int, Field(ge=8, le=SHA256_HEX_LENGTH)]
    human_readable_dirname: _NonEmptyStr
    hashes_dirname: _NonEmptyStr


def identity_from_pydantic(payload: object) -> IdentityConfigParser:
    document = IdentityDocument.model_validate(payload)
    manager = HashManager(
        prefix_len=document.prefix_len,
        human_readable_dirname=document.human_readable_dirname,
        hashes_dirname=document.hashes_dirname,
    )
    parsed = IdentityConfigParser.__new__(IdentityConfigParser)
    parsed._prefix_len = manager.prefix_len
    parsed._human_readable_dirname = manager.human_readable_dirname
    parsed._hashes_dirname = manager.hashes_dirname
    return parsed
