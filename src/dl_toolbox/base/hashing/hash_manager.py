"""
Context: Among the dl_toolbox objectives is to facilitate running experiments just by specifying configuration or .json files. 

Consequently, I want to know whether a certain specific combination of parameters (in a .json file) has already been run. 
Hashing is a good mechanism for that. Now, Imagine comparing runs with a tracking tool such as tensorboard or MLflow. 
I can't tell which run is which. So, we need human readable names.  And that's the identity of a .json file. 

The HashManager is a class that encapsulates the logic associated with a pair of (human_readable_name, hash) of a given .json file. 

The HashManager needs the following information to operate

1. the name of the subdirectory where the hashes are stored: hash_dir_name
2. the name of the subdirectory where the human readable names are stored: human_readable_dir_name
3. the prefix length of the hash: prefix_len

HashManager makes decisions based on a directory. The directory should have the following structure

- directory_in_question
-- hash_dir_name
-- human_readable_dir_name


Given a directory (with the above semantic structure), here is what HashManager can do:

1. determine the state of the pair (human_readable_name, hash): there are 4 possible states hash: taken vs free & name: taken vs free
... (to be continued)


Exclusive (name, hash) JSON index. Layout on the class; Directory per call. 
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.hashing.identity import (
    SHA256_HEX_LENGTH,
    artifact_stem,
    hash_payload,
    is_full_sha256,
    is_hashed_artifact_stem,
)

# although all the files are in a .json_io.py file, 
# I don't want to go lookup the file everytime I want to understand what a function does
# when I am reading the code below, the only way for me to understand what "write" means is to go lookup 
# the import and then the source file. 
from dl_toolbox.base.input_output.json_io import JsonTextForm, dump, read, write 
from dl_toolbox.base.parsing.parse_numerical import parse_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

_OCCUPANCY_NAMES = frozenset(
    {"both_free", "both_match", "name_taken", "hash_taken"}
)


class Occupancy:
    """Four-way occupancy of one (name, hash) query. Named constants, not strings."""

    def __init__(self, *, name: str) -> None:
        if name not in _OCCUPANCY_NAMES:
            raise ValueError(
                "occupancy name must be 'both_free', 'both_match', "
                f"'name_taken', or 'hash_taken', got {name!r}"
            )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Occupancy) and self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Occupancy.{self._name.upper()}"


# Occupancy.BOTH_FREE = Occupancy(name="both_free")
# Occupancy.BOTH_MATCH = Occupancy(name="both_match")
# Occupancy.NAME_TAKEN = Occupancy(name="name_taken")
# Occupancy.HASH_TAKEN = Occupancy(name="hash_taken")


class NameHashLookup:
    """Occupancy snapshot for one (name, hash) query. Disk is unchanged."""

    def __init__(
        self,
        *,
        state: Occupancy,
        name_path: Path,
        hash_path: Path,
        document: dict[str, Any] | None) -> None:
        if not isinstance(state, Occupancy):
            raise TypeError(f"state must be Occupancy, got {type(state)!r}")
        if not isinstance(name_path, Path) or not name_path.is_absolute():
            raise ValueError(f"name_path must be an absolute Path, got {name_path!r}")
        if not isinstance(hash_path, Path) or not hash_path.is_absolute():
            raise ValueError(f"hash_path must be an absolute Path, got {hash_path!r}")
        if state == Occupancy.BOTH_FREE:
            if document is not None:
                raise ValueError("BOTH_FREE lookup has no document")
        elif document is None:
            raise ValueError(f"{state!r} lookup requires a document")
        self._state = state
        self._name_path = name_path
        self._hash_path = hash_path
        self._document = document

    @property
    def state(self) -> Occupancy:
        return self._state

    @property
    def name_path(self) -> Path:
        return self._name_path

    @property
    def hash_path(self) -> Path:
        return self._hash_path

    @property
    def document(self) -> dict[str, Any] | None:
        return self._document


class HashManager:
    """Operations on an exclusive (name, hash) index. Does not lock. Does not mkdir the root."""

    BOTH_FREE = Occupancy(name="both_free")
    BOTH_MATCH = Occupancy(name="both_match")
    NAME_TAKEN = Occupancy(name="name_taken")
    HASH_TAKEN = Occupancy(name="hash_taken")

    def __init__(
        self,
        *,
        prefix_len: int,
        human_readable_dirname: str,
        hashes_dirname: str) -> None:
        self._prefix_len = _require_prefix_len(prefix_len)
        human = _require_index_dirname(
            human_readable_dirname,
            prefix_len=self._prefix_len,
            label="human_readable_dirname",
        )
        hashes = _require_index_dirname(
            hashes_dirname,
            prefix_len=self._prefix_len,
            label="hashes_dirname",
        )
        if human == hashes:
            raise ValueError(
                "human_readable_dirname and hashes_dirname must be distinct"
            )
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

    def assert_name(self, name: object) -> str:
        text = str(name).strip()
        if text.lower().endswith(".json"):
            text = text[:-5].strip()
        text = text.replace("\\", "/")
        if not text:
            raise ValueError("name is empty")
        if "/" in text or text in {".", ".."}:
            raise ValueError(f"name must be a basename: {text!r}")
        if self._is_hash_stem(text):
            raise ValueError(f"name must not be a hash stem: {text!r}")
        return text

    def lookup(
        self,
        directory: Directory,
        *,
        name: str,
        configuration_hash: str) -> NameHashLookup:

        name_path, hash_path, checked = self._query_paths(
            directory, name=name, configuration_hash=configuration_hash
        )
    
        name_exists = name_path.is_file()
        hash_exists = hash_path.is_file()
    
        if name_exists and hash_exists:
            name_document = self._read_pair_document(name_path)
            hash_document = self._read_pair_document(hash_path)
            if name_document != hash_document:
                return self._lookup_disagreeing(
                    checked=checked,
                    configuration_hash=configuration_hash,
                    name_path=name_path,
                    hash_path=hash_path,
                    name_document=name_document,
                    hash_document=hash_document,
                )
            stored_name = name_document["name"]
            stored_hash = name_document["hash"]
            if stored_name == checked and stored_hash == configuration_hash:
                return NameHashLookup(
                    state=HashManager.BOTH_MATCH,
                    name_path=name_path,
                    hash_path=hash_path,
                    document=name_document,
                )
            if stored_hash != configuration_hash:
                raise ValueError(
                    f"hash prefix {hash_path.stem!r} already used by {stored_hash!r}"
                )
            raise ValueError(
                f"stored pair for {checked!r} does not match configuration_hash"
            )
    
        if name_exists:
            return NameHashLookup(
                state=HashManager.NAME_TAKEN,
                name_path=name_path,
                hash_path=hash_path,
                document=self._read_pair_document(name_path),
            )
    
        if hash_exists:
            hash_document = self._read_pair_document(hash_path)
            stored_hash = hash_document["hash"]
            if stored_hash != configuration_hash:
                raise ValueError(
                    f"hash prefix {hash_path.stem!r} already used by {stored_hash!r}"
                )
            return NameHashLookup(
                state=HashManager.HASH_TAKEN,
                name_path=name_path,
                hash_path=hash_path,
                document=hash_document,
            )
    
        return NameHashLookup(
            state=HashManager.BOTH_FREE,
            name_path=name_path,
            hash_path=hash_path,
            document=None,
        )

    def claim(
        self,
        directory: Directory,
        *,
        name: str,
        configuration_hash: str,
        extra: Mapping[str, Any]) -> tuple[Path, Path]:

        checked = self.assert_name(name)

        extra_payload = _require_extra(extra)

        lookup = self.lookup(
            directory, name=checked, configuration_hash=configuration_hash
        )

        if lookup.state != HashManager.BOTH_FREE:
            self._raise_not_free(checked=checked, lookup=lookup)

        document: dict[str, Any] = {
            "name": checked,
            "hash": configuration_hash,
            **extra_payload,
        }

        hash_path = self._write_index_file(
            directory,
            folder=self._hashes_dirname,
            filename=f"{self._stem(configuration_hash)}.json",
            document=document,
        )

        name_path = self._write_index_file(
            directory,
            folder=self._human_readable_dirname,
            filename=f"{checked}.json",
            document=document,
        )

        return name_path, hash_path

    def release_name(
        self, directory: Directory, *, name: str
    ) -> dict[str, Any] | None:
        _require_index(directory)
        checked = self.assert_name(name)
        name_path = resolve_under_project(
            directory.path,
            f"{self._human_readable_dirname}/{checked}.json",
            label="name_path",
        )
        if not name_path.is_file():
            return None
        document = self._read_pair_document(name_path)
        if document["name"] != checked:
            raise ValueError(
                f"stored name {document['name']!r} does not match {checked!r}"
            )
        stored_hash = document["hash"]
        if not isinstance(stored_hash, str) or not is_full_sha256(stored_hash):
            raise ValueError(
                f"stored hash for {checked!r} must be a 64-char lowercase SHA-256, "
                f"got {stored_hash!r}"
            )
        hash_path = resolve_under_project(
            directory.path,
            f"{self._hashes_dirname}/{self._stem(stored_hash)}.json",
            label="hash_path",
        )
        if hash_path.is_file():
            hash_document = read(file=File.from_absolute(path=hash_path))
            if hash_document != document:
                raise ValueError(
                    f"name/hash JSON disagree for {checked!r} at {name_path} "
                    f"and {hash_path}"
                )
            hash_path.unlink()
        name_path.unlink()
        return document

    def _stem(self, configuration_hash: str) -> str:
        return artifact_stem(configuration_hash, prefix_len=self._prefix_len)

    def _is_hash_stem(self, name: str) -> bool:
        return is_hashed_artifact_stem(name, prefix_len=self._prefix_len)

    def _query_paths(
        self,
        directory: Directory,
        *,
        name: str,
        configuration_hash: str,
    ) -> tuple[Path, Path, str]:

        _require_index(directory)
        checked = self.assert_name(name)
        if not is_full_sha256(configuration_hash):
            raise ValueError(
                f"configuration_hash must be a 64-char lowercase SHA-256, got "
                f"{configuration_hash!r}"
            )
        name_path = resolve_under_project(
            directory.path,
            f"{self._human_readable_dirname}/{checked}.json",
            label="name_path",
        )
        hash_path = resolve_under_project(
            directory.path,
            f"{self._hashes_dirname}/{self._stem(configuration_hash)}.json",
            label="hash_path",
        )
        return name_path, hash_path, checked

    def _read_pair_document(self, path: Path) -> dict[str, Any]:
        document = read(file=File.from_absolute(path=path))
        if "name" not in document:
            raise ValueError(f"identity document at {path} is missing 'name'")
        if "hash" not in document:
            raise ValueError(f"identity document at {path} is missing 'hash'")
        return document

    def _lookup_disagreeing(
        self,
        *,
        checked: str,
        configuration_hash: str,
        name_path: Path,
        hash_path: Path,
        name_document: dict[str, Any],
        hash_document: dict[str, Any]) -> NameHashLookup:
        stored_hash = hash_document["hash"]
        if stored_hash != configuration_hash:
            raise ValueError(
                f"hash prefix {hash_path.stem!r} already used by {stored_hash!r}"
            )
        name_is_this_pair = (
            name_document["name"] == checked
            and name_document["hash"] == configuration_hash
        )
        hash_is_this_pair = (
            hash_document["name"] == checked
            and hash_document["hash"] == configuration_hash
        )
        if name_is_this_pair or hash_is_this_pair:
            raise ValueError(
                f"name/hash JSON disagree for {checked!r} at {name_path} and {hash_path}"
            )
        return NameHashLookup(
            state=HashManager.HASH_TAKEN,
            name_path=name_path,
            hash_path=hash_path,
            document=hash_document,
        )

    def _write_index_file(
        self,
        directory: Directory,
        *,
        folder: str,
        filename: str,
        document: dict[str, Any]) -> Path:

        child = Directory.from_under(
            root=directory,
            relative_path=folder,
            policy=DirectoryParsePolicy.CREATE,
        )

        written = write(
            directory=child,
            name=filename,
            payload=document,
            form=JsonTextForm.DOCUMENT,
        )
        
        return written.path

    def _raise_not_free(self, *, checked: str, lookup: NameHashLookup) -> None:
        if lookup.state == HashManager.BOTH_MATCH:
            raise FileExistsError(
                f"already claimed name {checked!r} with this configuration hash"
            )
        
        occupant = lookup.document
        
        if occupant is None:
            raise ValueError(f"occupied lookup {lookup.state!r} has no document")
        
        if lookup.state == HashManager.NAME_TAKEN:
            raise FileExistsError(
                f"name {checked!r} already exists with hash {occupant['hash']!r}"
            )
        raise FileExistsError(
            f"configuration hash already saved as name {occupant['name']!r}"
        )


def _require_prefix_len(raw: object) -> int:
    value = parse_int(raw, label="prefix_len")
    if value < 8 or value > SHA256_HEX_LENGTH:
        raise ValueError(
            f"prefix_len must be in [8, {SHA256_HEX_LENGTH}], got {value}"
        )
    return value


def _require_index_dirname(raw: object, *, prefix_len: int, label: str) -> str:
    text = parse_non_empty_str(raw, label=label)
    text = text.replace("\\", "/")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be a basename, got {text!r}")
    if is_hashed_artifact_stem(text, prefix_len=prefix_len):
        raise ValueError(f"{label} must not be a hash stem, got {text!r}")
    return text


def _require_index(directory: Directory) -> None:
    if not isinstance(directory, Directory):
        raise TypeError(f"directory must be Directory, got {type(directory)!r}")
    if directory.spec != DirectorySpec.EXISTS:
        raise ValueError(f"directory must exist: {directory.path}")


def _require_extra(extra: object) -> dict[str, Any]:
    if not isinstance(extra, Mapping):
        raise TypeError(f"extra must be a JSON object, got {type(extra)!r}")
    payload = dict(extra)
    dump(payload, form=JsonTextForm.CANONICAL)
    if "name" in payload or "hash" in payload:
        raise ValueError("extra must not repeat name or hash")
    return payload
