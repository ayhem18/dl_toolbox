"""Parse project_level_configs/identity.json. Does not construct HashManager."""

from __future__ import annotations

from dl_toolbox.base.hashing.hash_manager import assert_index_dirname, assert_prefix_len
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

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
