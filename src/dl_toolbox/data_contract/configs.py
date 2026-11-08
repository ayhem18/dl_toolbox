"""Collection configuration: hashed File and alias File under one Directory."""

from __future__ import annotations

from typing import Any

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import (
    artifact_stem,
    is_full_sha256,
    is_hashed_artifact_stem,
)
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object
from dl_toolbox.data_contract.names import assert_generating_process_name


def write_collection_configuration(
    collection: object, *, manager: HashManager
) -> File:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    name = assert_generating_process_name(
        collection.generating_process_name, manager=manager
    )
    document = collection.to_dict()
    if document["hash"] != collection.collection_hash:
        raise ValueError("collection.to_dict()['hash'] must equal collection_hash")
    if document["generating_process_name"] != name:
        raise ValueError(
            "collection.to_dict()['generating_process_name'] must equal "
            "generating_process_name"
        )
    configs_dir = _require_directory(collection.configs_dir(), label="configs_dir")
    if configs_dir.spec == DirectorySpec.MISSING:
        configs_dir = Directory.from_absolute(
            path=configs_dir.path, policy=DirectoryParsePolicy.CREATE
        )
    hashed_name = _hashed_relative_name(
        collection.collection_hash, manager=manager
    )
    hashed_file = write_object(
        directory=configs_dir,
        name=hashed_name,
        payload=document,
        form=JsonTextForm.DOCUMENT,
    )
    write_object(
        directory=configs_dir,
        name=f"{name}.json",
        payload=document,
        form=JsonTextForm.DOCUMENT,
    )
    return hashed_file


def load_collection_configuration(
    configs_dir: Directory, identity: str, *, manager: HashManager
) -> tuple[File, dict[str, Any]]:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    configs_dir = _require_existing_directory(configs_dir, label="configs_dir")
    text = str(identity).strip()
    if not text:
        raise ValueError("identity is empty")
    file = _file_under(configs_dir, f"{text}.json")
    if file is None and is_full_sha256(text):
        file = _file_under(
            configs_dir, _hashed_relative_name(text, manager=manager)
        )
    if file is None:
        raise FileNotFoundError(
            f"No configuration {text!r} under {configs_dir.path}"
        )
    document = read_object(file=file)
    if "hash" not in document:
        raise ValueError(f"configuration {file.path} has no 'hash' field")
    stored_hash = document["hash"]
    if not isinstance(stored_hash, str) or not is_full_sha256(stored_hash):
        raise ValueError(
            f"configuration {file.path} hash must be a 64-char lowercase SHA-256, "
            f"got {stored_hash!r}"
        )
    if is_hashed_artifact_stem(file.path.stem, prefix_len=manager.prefix_len):
        if not stored_hash.startswith(file.path.stem):
            raise ValueError(
                f"stem {file.path.stem!r} does not match hash {stored_hash!r} "
                f"in {file.path}"
            )
    if is_full_sha256(text) and stored_hash != text:
        raise ValueError(
            f"requested key {text!r} does not match hash {stored_hash!r}"
        )
    if is_hashed_artifact_stem(text, prefix_len=manager.prefix_len):
        if not stored_hash.startswith(text):
            raise ValueError(
                f"requested key {text!r} does not match hash {stored_hash!r}"
            )
    return file, document


def delete_collection_configuration(
    configs_dir: Directory, *, generating_process_name: str, manager: HashManager
) -> bool:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    configs_dir = _require_existing_directory(configs_dir, label="configs_dir")
    name = assert_generating_process_name(
        generating_process_name, manager=manager
    )
    try:
        alias = File.from_under(root=configs_dir, relative_path=f"{name}.json")
    except FileNotFoundError:
        return False
    document = read_object(file=alias)
    stored_hash = document["hash"]
    try:
        hashed = File.from_under(
            root=configs_dir,
            relative_path=_hashed_relative_name(stored_hash, manager=manager),
        )
    except FileNotFoundError:
        hashed = None
    if hashed is not None:
        hashed.path.unlink()
    alias.path.unlink()
    return True


def _hashed_relative_name(full_hash: str, *, manager: HashManager) -> str:
    return f"{artifact_stem(full_hash, prefix_len=manager.prefix_len)}.json"


def _file_under(directory: Directory, relative_path: str) -> File | None:
    try:
        return File.from_under(root=directory, relative_path=relative_path)
    except FileNotFoundError:
        return None


def _require_directory(directory: object, *, label: str) -> Directory:
    if not isinstance(directory, Directory):
        raise TypeError(f"{label} must be Directory, got {type(directory)!r}")
    return directory


def _require_existing_directory(directory: object, *, label: str) -> Directory:
    directory = _require_directory(directory, label=label)
    if directory.spec != DirectorySpec.EXISTS:
        raise ValueError(f"{label} must exist: {directory.path}")
    return directory
