"""Claim, write-last, and incomplete-run cleanup for a process identity index."""

from __future__ import annotations

import shutil

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager, PairLookup
from dl_toolbox.data_contract.configs import delete_collection_configuration


def claim_generating_process(
    manager: HashManager,
    directory: Directory,
    *,
    generating_process_name: str,
    configuration_hash: str,
) -> str:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    name = manager.validate_human_readable_name(generating_process_name)
    lookup = manager.lookup(
        directory, human_readable_name=name, hash=configuration_hash
    )
    if lookup.state != HashManager.BOTH_FREE:
        _raise_occupied(lookup, generating_process_name=name)
    return name


def create_payload_directory(payload: Directory) -> Directory:
    if not isinstance(payload, Directory):
        raise TypeError(f"payload must be Directory, got {type(payload)!r}")
    Directory.from_absolute(
        path=payload.path.parent, policy=DirectoryParsePolicy.CREATE
    )
    if payload.spec == DirectorySpec.EXISTS:
        raise FileExistsError(f"payload directory already exists: {payload.path}")
    payload.path.mkdir(exist_ok=False)
    return Directory.from_absolute(
        path=payload.path, policy=DirectoryParsePolicy.MUST_EXIST
    )


def write_process_configuration(
    manager: HashManager,
    directory: Directory,
    *,
    generating_process_name: str,
    configuration_hash: str,
    generating_process_signature: dict,
) -> File:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    if not isinstance(generating_process_signature, dict):
        raise TypeError("generating_process_signature must be a dict")
    name_path, _hash_path = manager.create_pair(
        directory,
        human_readable_name=generating_process_name,
        hash=configuration_hash,
        extra={"generating_process_signature": generating_process_signature},
    )
    return File.from_absolute(path=name_path)


def delete_incomplete_run(
    *,
    manager: HashManager,
    directory: Directory,
    generating_process_name: str,
    configs_dir: Directory,
    payload: Directory | None,
) -> None:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    if not isinstance(configs_dir, Directory):
        raise TypeError(f"configs_dir must be Directory, got {type(configs_dir)!r}")
    manager.drop_pair(directory, human_readable_name=generating_process_name)
    if payload is not None:
        _rmtree_if_exists(payload)
    if configs_dir.spec == DirectorySpec.EXISTS:
        delete_collection_configuration(
            configs_dir,
            generating_process_name=generating_process_name,
            manager=manager,
        )


def _rmtree_if_exists(payload: Directory) -> None:
    if not isinstance(payload, Directory):
        raise TypeError(f"payload must be Directory, got {type(payload)!r}")
    if payload.spec == DirectorySpec.EXISTS:
        shutil.rmtree(payload.path)


def _raise_occupied(
    lookup: PairLookup, *, generating_process_name: str
) -> None:
    state = lookup.state
    document = lookup.document
    if state == HashManager.BOTH_MATCH:
        raise FileExistsError(
            f"pair already exists for name {generating_process_name!r} "
            "with this configuration hash"
        )
    if document is None:
        raise ValueError(f"occupied lookup {state!r} has no document")
    if state == HashManager.NAME_TAKEN:
        raise FileExistsError(
            f"name {generating_process_name!r} already exists with hash "
            f"{document['hash']!r}"
        )
    raise FileExistsError(
        f"configuration hash already saved as name {document['name']!r}"
    )
