"""
Context: Among the dl_toolbox objectives is to facilitate running experiments just by specifying configuration or .json files.

Consequently, I want to know whether a certain specific combination of parameters (in a .json file) has already been run.
Hashing is a good mechanism for that. Now, Imagine comparing runs with a tracking tool such as tensorboard or MLflow.
I can't tell which run is which. So, we need human readable names.  And that's the identity of a .json file.

The HashManager is a class that encapsulates the logic associated with a pair of (human_readable_name, hash) of a given .json file.


1. HashManager Assumptions / Context

The HashManager needs the following information to operate

1. the name of the subdirectory where the hashes are stored: hash_dir_name
2. the name of the subdirectory where the human readable names are stored: human_readable_dir_name
3. the prefix length of the hash: prefix_len
4. how long to wait for the exclusive lock: lock_timeout_seconds
5. how often to retry that lock: lock_poll_seconds

HashManager makes decisions based on a directory. After create_identity_index, that directory has the following structure

- identity index
|-- .dir_index
|-- hash_dir_name
|-- human_readable_dir_name
|-- identity.lock     (present only while a process holds the exclusive lock)

create_identity_index is the only HashManager call that creates that structure.
lookup / create_pair / drop_pair require .dir_index and the two subfolders.

2. HashManager main functionalities

- create_identity_index: create the directory that will be used as an identity index: the root, .dir_index, hash_dir_name, and human_readable_dir_name.

- lookup: given (human_readable_name, hash), determine the state of the pair: four states — hash taken vs free, and name taken vs free. One disk pass: load each existing JSON once as a dict (identical_to_name, identical_to_hash). The four states are derived from those dicts in memory.

- create_pair: given (human_readable_name, hash), verify both are free. Write a copy of the json file in the hash subdirectory first and then a copy in the human_readable subdirectory.

- drop_pair: given (human_readable_name, hash): the call is by the human readable name; the hash is the one stored in that json. If the name is not stored, nothing is deleted. If both copies exist, they must be the same document. Delete the copy in the hash subdirectory first, then the copy in the human_readable subdirectory.

3. HashManager secondary (but impactful) functionalities:

- hash_payload
- validate_human_readable_name
- pair_state
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
from dl_toolbox.base.file_management.exclusive_lock import ExclusiveDirLock
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.hashing import identity
from dl_toolbox.base.input_output import json_io
from dl_toolbox.base.input_output.json_io import JsonTextForm
from dl_toolbox.base.parsing.parse_numerical import parse_int, parse_number
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of


class PairLookup:
    def __init__(
        self,
        *,
        state: str,
        name_path: Path,
        hash_path: Path,
        document: dict[str, Any] | None,
    ) -> None:
        state = parse_one_of(
            state,
            allowed=HashManager.PAIR_STATES,
            label="pair state",
            error_type=ValueError,
            error_message=one_of_message,
        )
        if not isinstance(name_path, Path) or not name_path.is_absolute():
            raise ValueError(f"name_path must be an absolute Path, got {name_path!r}")
        if not isinstance(hash_path, Path) or not hash_path.is_absolute():
            raise ValueError(f"hash_path must be an absolute Path, got {hash_path!r}")
        if state == HashManager.BOTH_FREE:
            if document is not None:
                raise ValueError("BOTH_FREE lookup has no document")
        elif document is None:
            raise ValueError(f"{state!r} lookup requires a document")
        self._state = state
        self._name_path = name_path
        self._hash_path = hash_path
        self._document = document

    @property
    def state(self) -> str:
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
    BOTH_FREE = "both_free"
    BOTH_MATCH = "both_match"
    NAME_TAKEN = "name_taken"
    HASH_TAKEN = "hash_taken"
    PAIR_STATES = (BOTH_FREE, BOTH_MATCH, NAME_TAKEN, HASH_TAKEN)
    DIR_INDEX_NAME = ".dir_index"
    IDENTITY_LOCK_NAME = "identity.lock"

    def __init__(
        self,
        *,
        prefix_len: int,
        human_readable_dir_name: str,
        hash_dir_name: str,
        lock_timeout_seconds: float,
        lock_poll_seconds: float,
    ) -> None:
        self._prefix_len = _require_prefix_len(prefix_len)
        human = _require_index_dirname(
            human_readable_dir_name,
            prefix_len=self._prefix_len,
            label="human_readable_dir_name",
        )
        hashes = _require_index_dirname(
            hash_dir_name,
            prefix_len=self._prefix_len,
            label="hash_dir_name",
        )
        if human == hashes:
            raise ValueError(
                "human_readable_dir_name and hash_dir_name must be distinct"
            )
        self._human_readable_dir_name = human
        self._hash_dir_name = hashes
        self._lock_timeout_seconds = _require_positive_seconds(
            lock_timeout_seconds, label="lock_timeout_seconds"
        )
        self._lock_poll_seconds = _require_positive_seconds(
            lock_poll_seconds, label="lock_poll_seconds"
        )

    @property
    def prefix_len(self) -> int:
        return self._prefix_len

    @property
    def human_readable_dir_name(self) -> str:
        return self._human_readable_dir_name

    @property
    def hash_dir_name(self) -> str:
        return self._hash_dir_name

    @property
    def lock_timeout_seconds(self) -> float:
        return self._lock_timeout_seconds

    @property
    def lock_poll_seconds(self) -> float:
        return self._lock_poll_seconds

    def hash_payload(self, payload: object) -> str:
        return identity.hash_payload(payload)

    def validate_human_readable_name(self, human_readable_name: object) -> str:
        text = str(human_readable_name).strip()
        if text.lower().endswith(".json"):
            text = text[:-5].strip()
        text = text.replace("\\", "/")
        if not text:
            raise ValueError("human_readable_name is empty")
        if "/" in text or text in {".", ".."}:
            raise ValueError(f"human_readable_name must be a basename: {text!r}")
        if self._is_prefix(text):
            raise ValueError(f"human_readable_name must not be a prefix: {text!r}")
        return text

    def create_identity_index(self, directory: Directory) -> Directory:
        if not isinstance(directory, Directory):
            raise TypeError(f"directory must be Directory, got {type(directory)!r}")
        if directory.spec == DirectorySpec.EXISTS:
            if self._is_identity_index(directory):
                return directory
            raise ValueError(
                f"directory is not an identity index: {directory.path}"
            )
        created = Directory.from_absolute(
            path=directory.path, policy=DirectoryParsePolicy.CREATE
        )
        Directory.from_under(
            root=created,
            relative_path=self._hash_dir_name,
            policy=DirectoryParsePolicy.CREATE,
        )
        Directory.from_under(
            root=created,
            relative_path=self._human_readable_dir_name,
            policy=DirectoryParsePolicy.CREATE,
        )
        marker = created.path / HashManager.DIR_INDEX_NAME
        marker.write_text("", encoding="utf-8")
        return created

    def lookup(
        self,
        directory: Directory,
        *,
        human_readable_name: str,
        hash: str,
    ) -> PairLookup:
        """given (human_readable_name, hash), determine the state of the pair: four states — hash taken vs free, and name taken vs free."""
        self._require_identity_index(directory)
        with self._exclusive_lock(directory):
            checked, name_path, hash_path, identical_to_name, identical_to_hash = (
                self._read_matches(
                    directory, human_readable_name=human_readable_name, hash=hash
                )
            )
            state = self.pair_state(
                identical_to_name=identical_to_name,
                identical_to_hash=identical_to_hash,
                human_readable_name=checked,
                hash=hash,
            )
            document = _document_for_state(
                state=state,
                identical_to_name=identical_to_name,
                identical_to_hash=identical_to_hash,
            )
            return PairLookup(
                state=state,
                name_path=name_path,
                hash_path=hash_path,
                document=document,
            )

    def create_pair(
        self,
        directory: Directory,
        *,
        human_readable_name: str,
        hash: str,
        extra: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        """given (human_readable_name, hash), verify both are free. Write a copy of the json file in the hash subdirectory first and then a copy in the human_readable subdirectory."""
        checked = self.validate_human_readable_name(human_readable_name)
        extra_payload = _require_extra(extra)
        self._require_identity_index(directory)
        with self._exclusive_lock(directory):
            checked, name_path, hash_path, identical_to_name, identical_to_hash = (
                self._read_matches(
                    directory, human_readable_name=checked, hash=hash
                )
            )
            state = self.pair_state(
                identical_to_name=identical_to_name,
                identical_to_hash=identical_to_hash,
                human_readable_name=checked,
                hash=hash,
            )
            if state != HashManager.BOTH_FREE:
                _raise_not_free(
                    checked=checked,
                    state=state,
                    document=_document_for_state(
                        state=state,
                        identical_to_name=identical_to_name,
                        identical_to_hash=identical_to_hash,
                    ),
                )
            document: dict[str, Any] = {
                "name": checked,
                "hash": hash,
                **extra_payload,
            }
            written_hash = self._write_index_file(
                directory,
                folder=self._hash_dir_name,
                filename=f"{self._prefix(hash)}.json",
                document=document,
            )
            written_name = self._write_index_file(
                directory,
                folder=self._human_readable_dir_name,
                filename=f"{checked}.json",
                document=document,
            )
            return written_name, written_hash

    def drop_pair(
        self, directory: Directory, *, human_readable_name: str
    ) -> dict[str, Any] | None:
        """given (human_readable_name, hash): the call is by the human readable name; the hash is the one stored in that json. If the name is not stored, nothing is deleted. If both copies exist, they must be the same document. Delete the copy in the hash subdirectory first, then the copy in the human_readable subdirectory."""
        self._require_identity_index(directory)
        checked = self.validate_human_readable_name(human_readable_name)
        with self._exclusive_lock(directory):
            name_path = resolve_under_project(
                directory.path,
                f"{self._human_readable_dir_name}/{checked}.json",
                label="name_path",
            )
            if not name_path.is_file():
                return None
            document = self._load_index_document(name_path)
            if document["name"] != checked:
                raise ValueError(
                    f"stored name {document['name']!r} does not match {checked!r}"
                )
            stored_hash = document["hash"]
            if not isinstance(stored_hash, str) or not identity.is_full_sha256(
                stored_hash
            ):
                raise ValueError(
                    f"stored hash for {checked!r} must be a 64-char lowercase SHA-256, "
                    f"got {stored_hash!r}"
                )
            hash_path = resolve_under_project(
                directory.path,
                f"{self._hash_dir_name}/{self._prefix(stored_hash)}.json",
                label="hash_path",
            )
            if hash_path.is_file():
                hash_document = self._load_index_document(hash_path)
                if hash_document != document:
                    raise ValueError(
                        f"name/hash JSON disagree for {checked!r} at {name_path} "
                        f"and {hash_path}"
                    )
                hash_path.unlink()
            name_path.unlink()
            return document

    def pair_state(
        self,
        *,
        identical_to_name: list[dict[str, Any]],
        identical_to_hash: list[dict[str, Any]],
        human_readable_name: str,
        hash: str,
    ) -> str:
        if not isinstance(identical_to_name, list):
            raise TypeError(
                f"identical_to_name must be a list, got {type(identical_to_name)!r}"
            )
        if not isinstance(identical_to_hash, list):
            raise TypeError(
                f"identical_to_hash must be a list, got {type(identical_to_hash)!r}"
            )
        if len(identical_to_name) > 1 or len(identical_to_hash) > 1:
            raise ValueError("corrupt index: more than one document for a path")
        if identical_to_hash and identical_to_hash[0]["name"] != human_readable_name:
            return HashManager.HASH_TAKEN
        if identical_to_name and identical_to_name[0]["hash"] != hash:
            return HashManager.NAME_TAKEN
        if (
            identical_to_name
            and identical_to_hash
            and identical_to_name[0]["name"] == human_readable_name
            and identical_to_name[0]["hash"] == hash
            and identical_to_hash[0] == identical_to_name[0]
        ):
            return HashManager.BOTH_MATCH
        if not identical_to_name and not identical_to_hash:
            return HashManager.BOTH_FREE
        raise ValueError("lists do not form a pair state")

    def _prefix(self, hash: str) -> str:
        return identity.artifact_stem(hash, prefix_len=self._prefix_len)

    def _is_prefix(self, name: str) -> bool:
        return identity.is_hashed_artifact_stem(name, prefix_len=self._prefix_len)

    def _is_identity_index(self, directory: Directory) -> bool:
        marker = directory.path / HashManager.DIR_INDEX_NAME
        hash_dir = directory.path / self._hash_dir_name
        name_dir = directory.path / self._human_readable_dir_name
        return marker.is_file() and hash_dir.is_dir() and name_dir.is_dir()

    def _require_identity_index(self, directory: Directory) -> None:
        if not isinstance(directory, Directory):
            raise TypeError(f"directory must be Directory, got {type(directory)!r}")
        if directory.spec != DirectorySpec.EXISTS:
            raise ValueError(f"directory must exist: {directory.path}")
        if not self._is_identity_index(directory):
            raise ValueError(
                f"directory is not an identity index: {directory.path}"
            )

    def _exclusive_lock(self, directory: Directory) -> ExclusiveDirLock:
        return ExclusiveDirLock(
            directory.path / HashManager.IDENTITY_LOCK_NAME,
            timeout_seconds=self._lock_timeout_seconds,
            poll_seconds=self._lock_poll_seconds,
        )

    def _read_matches(
        self,
        directory: Directory,
        *,
        human_readable_name: str,
        hash: str,
    ) -> tuple[str, Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
        checked = self.validate_human_readable_name(human_readable_name)
        if not identity.is_full_sha256(hash):
            raise ValueError(
                f"hash must be a 64-char lowercase SHA-256, got {hash!r}"
            )
        name_path = resolve_under_project(
            directory.path,
            f"{self._human_readable_dir_name}/{checked}.json",
            label="name_path",
        )
        hash_path = resolve_under_project(
            directory.path,
            f"{self._hash_dir_name}/{self._prefix(hash)}.json",
            label="hash_path",
        )
        identical_to_name: list[dict[str, Any]] = []
        identical_to_hash: list[dict[str, Any]] = []
        if name_path.is_file():
            identical_to_name = [self._load_index_document(name_path)]
        if hash_path.is_file():
            hash_document = self._load_index_document(hash_path)
            if hash_document["hash"] != hash:
                raise ValueError(
                    f"prefix {self._prefix(hash)!r} already used by "
                    f"{hash_document['hash']!r}"
                )
            identical_to_hash = [hash_document]
        return checked, name_path, hash_path, identical_to_name, identical_to_hash

    def _load_index_document(self, path: Path) -> dict[str, Any]:
        document = json_io.read_object(file=File.from_absolute(path=path))
        if "name" not in document:
            raise ValueError(f"identity document at {path} is missing 'name'")
        if "hash" not in document:
            raise ValueError(f"identity document at {path} is missing 'hash'")
        return document

    def _write_index_file(
        self,
        directory: Directory,
        *,
        folder: str,
        filename: str,
        document: dict[str, Any],
    ) -> Path:
        child = Directory.from_under(
            root=directory,
            relative_path=folder,
            policy=DirectoryParsePolicy.MUST_EXIST,
        )
        written = json_io.write_object(
            directory=child,
            name=filename,
            payload=document,
            form=JsonTextForm.DOCUMENT,
        )
        return written.path


def _document_for_state(
    *,
    state: str,
    identical_to_name: list[dict[str, Any]],
    identical_to_hash: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if state == HashManager.BOTH_FREE:
        return None
    if state == HashManager.HASH_TAKEN:
        return identical_to_hash[0]
    return identical_to_name[0]


def _raise_not_free(
    *, checked: str, state: str, document: dict[str, Any] | None
) -> None:
    if state == HashManager.BOTH_MATCH:
        raise FileExistsError(
            f"pair already exists for name {checked!r} with this configuration hash"
        )
    if document is None:
        raise ValueError(f"occupied lookup {state!r} has no document")
    if state == HashManager.NAME_TAKEN:
        raise FileExistsError(
            f"name {checked!r} already exists with hash {document['hash']!r}"
        )
    raise FileExistsError(
        f"configuration hash already saved as name {document['name']!r}"
    )


def _require_prefix_len(raw: object) -> int:
    value = parse_int(raw, label="prefix_len")
    if value < 8 or value > identity.SHA256_HEX_LENGTH:
        raise ValueError(
            f"prefix_len must be in [8, {identity.SHA256_HEX_LENGTH}], got {value}"
        )
    return value


def _require_positive_seconds(raw: object, *, label: str) -> float:
    value = parse_number(raw, label=label)
    if float(value) <= 0:
        raise ValueError(f"{label} must be > 0, got {value}")
    return float(value)


def _require_index_dirname(raw: object, *, prefix_len: int, label: str) -> str:
    text = parse_non_empty_str(raw, label=label)
    text = text.replace("\\", "/")
    if "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be a basename, got {text!r}")
    if identity.is_hashed_artifact_stem(text, prefix_len=prefix_len):
        raise ValueError(f"{label} must not be a prefix, got {text!r}")
    return text


def _require_extra(extra: object) -> dict[str, Any]:
    if not isinstance(extra, Mapping):
        raise TypeError(f"extra must be a JSON object, got {type(extra)!r}")
    payload = dict(extra)
    json_io.dump_object(payload, form=JsonTextForm.CANONICAL)
    if "name" in payload or "hash" in payload:
        raise ValueError("extra must not repeat name or hash")
    return payload
