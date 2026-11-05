"""Parse project_level_configs/identity.json. Does not construct HashManager."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from dl_toolbox.base.hashing.hash_manager import assert_index_dirname, assert_prefix_len
from dl_toolbox.base.hashing.identity import SHA256_HEX_LENGTH
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_KEYS = ("prefix_len", "human_readable_dirname", "hashes_dirname")


class IdentityConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="identity.json")
        require_exact_keys(obj, _KEYS, label="identity.json")
        self._prefix_len = assert_prefix_len(obj["prefix_len"])
        human = assert_index_dirname(
            obj["human_readable_dirname"],
            prefix_len=self._prefix_len,
            label="human_readable_dirname",
        )
        hashes = assert_index_dirname(
            obj["hashes_dirname"],
            prefix_len=self._prefix_len,
            label="hashes_dirname",
        )
        if human == hashes:
            raise ValueError("human_readable_dirname and hashes_dirname must be distinct")
        self._human_readable_dirname = human
        self._hashes_dirname = hashes

    @property
    def prefix_len(self) -> int:
        return self._prefix_len

    @property
    def human_readable_dirname(self) -> str:
        return self._human_readable_dirname

    @property
    def hashes_dirname(self) -> str:
        return self._hashes_dirname


# --- pydantic (optional extra: validation) ---
# Exact keys, prefix_len range. Leftover: basename / not-a-hash-stem rules
# (assert_index_dirname) and the two dirnames must be distinct.


class IdentityDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prefix_len: Annotated[int, Field(ge=8, le=SHA256_HEX_LENGTH)]
    human_readable_dirname: _NonEmptyStr
    hashes_dirname: _NonEmptyStr


def identity_from_pydantic(payload: object) -> IdentityConfigParser:
    document = IdentityDocument.model_validate(payload)
    human = assert_index_dirname(
        document.human_readable_dirname,
        prefix_len=document.prefix_len,
        label="human_readable_dirname",
    )
    hashes = assert_index_dirname(
        document.hashes_dirname,
        prefix_len=document.prefix_len,
        label="hashes_dirname",
    )
    if human == hashes:
        raise ValueError("human_readable_dirname and hashes_dirname must be distinct")
    parsed = IdentityConfigParser.__new__(IdentityConfigParser)
    parsed._prefix_len = document.prefix_len
    parsed._human_readable_dirname = human
    parsed._hashes_dirname = hashes
    return parsed
