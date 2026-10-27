"""Project-level exclusive (name, hash) JSON pair. Does not read identity.json."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dl_toolbox.base.hashing.identity import (
    SHA256_HEX_LENGTH,
    artifact_stem,
    hash_payload,
    is_full_sha256,
    is_hashed_artifact_stem,
)
from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.base.parsing.parse_numerical import parse_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

BOTH_FREE = "both_free"
BOTH_MATCH = "both_match"
NAME_TAKEN = "name_taken"
HASH_TAKEN = "hash_taken"


@dataclass(frozen=True)
class NameHashLookup:
    state: str
    name_path: Path
    hash_path: Path
    document: dict[str, Any] | None = None


def assert_index_dirname(raw: object, *, prefix_len: int, label: str) -> str:
    text = parse_non_empty_str(raw, label=label)
    text = text.replace("\\", "/")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be a basename, got {text!r}")
    if is_hashed_artifact_stem(text, prefix_len=prefix_len):
        raise ValueError(f"{label} must not be a hash stem, got {text!r}")
    return text


def assert_prefix_len(raw: object) -> int:
    value = parse_int(raw, label="prefix_len")
    if value < 8 or value > SHA256_HEX_LENGTH:
        raise ValueError(f"prefix_len must be in [8, {SHA256_HEX_LENGTH}], got {value}")
    return value


class HashManager:
    def __init__(
        self,
        *,
        prefix_len: int,
        human_readable_dirname: str,
        hashes_dirname: str,
    ) -> None:
        self._prefix_len = assert_prefix_len(prefix_len)
        human = assert_index_dirname(
            human_readable_dirname,
            prefix_len=self._prefix_len,
            label="human_readable_dirname",
        )
        hashes = assert_index_dirname(
            hashes_dirname,
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

    def hash_payload(self, payload: object) -> str:
        return hash_payload(payload)

    def stem(self, configuration_hash: str) -> str:
        return artifact_stem(configuration_hash, prefix_len=self._prefix_len)

    def is_hash_stem(self, name: str) -> bool:
        return is_hashed_artifact_stem(name, prefix_len=self._prefix_len)

    def assert_name(self, name: object) -> str:
        text = str(name).strip()
        if text.lower().endswith(".json"):
            text = text[:-5].strip()
        text = text.replace("\\", "/")
        if not text:
            raise ValueError("name is empty")
        if "/" in text or text in {".", ".."}:
            raise ValueError(f"name must be a basename: {text!r}")
        if self.is_hash_stem(text):
            raise ValueError(f"name must not be a hash stem: {text!r}")
        return text

    def name_hash_paths(
        self,
        directory: Path,
        *,
        name: str,
        configuration_hash: str,
    ) -> tuple[Path, Path]:
        if not directory.is_absolute():
            raise ValueError(f"directory must be an absolute path, got {directory}")
        checked = self.assert_name(name)
        if not is_full_sha256(configuration_hash):
            raise ValueError(
                f"configuration_hash must be a 64-char lowercase SHA-256, got "
                f"{configuration_hash!r}"
            )
        name_path = directory / self._human_readable_dirname / f"{checked}.json"
        hash_path = directory / self._hashes_dirname / f"{self.stem(configuration_hash)}.json"
        return name_path, hash_path

    def lookup_name_hash_json(
        self,
        directory: Path,
        *,
        name: str,
        configuration_hash: str,
    ) -> NameHashLookup:
        name_path, hash_path = self.name_hash_paths(
            directory, name=name, configuration_hash=configuration_hash
        )
        checked = self.assert_name(name)
        name_exists = name_path.is_file()
        hash_exists = hash_path.is_file()
        if name_exists and hash_exists:
            name_document = read_json(name_path)
            hash_document = read_json(hash_path)
            if name_document != hash_document:
                raise ValueError(
                    f"name/hash JSON disagree for {checked!r} at {name_path} and {hash_path}"
                )
            if (
                name_document.get("name") != checked
                or name_document.get("hash") != configuration_hash
            ):
                stored = name_document.get("hash")
                if stored != configuration_hash and hash_path.stem == self.stem(
                    configuration_hash
                ):
                    raise ValueError(
                        f"hash prefix {hash_path.stem!r} already used by {stored!r}"
                    )
                raise ValueError(
                    f"stored pair for {checked!r} does not match configuration_hash"
                )
            return NameHashLookup(
                state=BOTH_MATCH,
                name_path=name_path,
                hash_path=hash_path,
                document=name_document,
            )
        if name_exists:
            return NameHashLookup(
                state=NAME_TAKEN, name_path=name_path, hash_path=hash_path
            )
        if hash_exists:
            hash_document = read_json(hash_path)
            stored = hash_document.get("hash")
            if stored != configuration_hash:
                raise ValueError(
                    f"hash prefix {hash_path.stem!r} already used by {stored!r}"
                )
            return NameHashLookup(
                state=HASH_TAKEN, name_path=name_path, hash_path=hash_path
            )
        return NameHashLookup(
            state=BOTH_FREE, name_path=name_path, hash_path=hash_path
        )

    def verify_name_hash_json_unique(
        self,
        directory: Path,
        *,
        name: str,
        configuration_hash: str,
    ) -> None:
        lookup = self.lookup_name_hash_json(
            directory, name=name, configuration_hash=configuration_hash
        )
        if lookup.state == BOTH_FREE:
            return
        if lookup.state == BOTH_MATCH:
            raise FileExistsError(
                f"already claimed name {name!r} with this configuration hash"
            )
        if lookup.state == NAME_TAKEN:
            occupant = read_json(lookup.name_path)
            raise FileExistsError(
                f"name {name!r} already exists with hash {occupant.get('hash')!r}"
            )
        occupant = read_json(lookup.hash_path)
        raise FileExistsError(
            f"configuration hash already saved as name {occupant.get('name')!r}"
        )

    def write_name_hash_json_unique(
        self,
        directory: Path,
        *,
        name: str,
        configuration_hash: str,
        extra: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        checked = self.assert_name(name)
        extra_payload = dict(extra) if extra is not None else {}
        if "name" in extra_payload or "hash" in extra_payload:
            raise ValueError("extra must not repeat name or hash")
        self.verify_name_hash_json_unique(
            directory, name=checked, configuration_hash=configuration_hash
        )
        document: dict[str, Any] = {
            "name": checked,
            "hash": configuration_hash,
            **extra_payload,
        }
        name_path, hash_path = self.name_hash_paths(
            directory, name=checked, configuration_hash=configuration_hash
        )
        write_json(hash_path, document)
        write_json(name_path, document)
        return name_path, hash_path
