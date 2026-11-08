"""Delete a finished freeze so a later ``process()`` can claim it."""

from __future__ import annotations

import shutil

from dl_toolbox.base.file_management.directory import Directory, DirectorySpec
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.data_contract.configs import delete_collection_configuration


def process_hash_occupant(
    manager: HashManager,
    directory: Directory,
    *,
    configuration_hash: str,
) -> str | None:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    if not isinstance(directory, Directory):
        raise TypeError(f"directory must be Directory, got {type(directory)!r}")
    if not is_full_sha256(configuration_hash):
        raise ValueError(
            f"configuration_hash must be a 64-char lowercase SHA-256, got "
            f"{configuration_hash!r}"
        )
    if directory.spec == DirectorySpec.MISSING:
        return None
    probe = "process_hash_occupant_probe"
    lookup = manager.lookup(
        directory, human_readable_name=probe, hash=configuration_hash
    )
    if lookup.state == HashManager.HASH_TAKEN:
        document = lookup.document
        if document is None:
            raise ValueError("HASH_TAKEN lookup has no document")
        return manager.validate_human_readable_name(document["name"])
    if lookup.state == HashManager.BOTH_MATCH:
        return probe
    return None


def remove_named_freeze(
    manager: HashManager,
    directory: Directory,
    *,
    generating_process_name: str,
    configs_dir: Directory,
    payload: Directory | None,
) -> bool:
    if not isinstance(manager, HashManager):
        raise TypeError(f"manager must be HashManager, got {type(manager)!r}")
    if not isinstance(directory, Directory):
        raise TypeError(f"directory must be Directory, got {type(directory)!r}")
    if not isinstance(configs_dir, Directory):
        raise TypeError(f"configs_dir must be Directory, got {type(configs_dir)!r}")
    name = manager.validate_human_readable_name(generating_process_name)
    dropped = None
    if directory.spec == DirectorySpec.EXISTS:
        dropped = manager.drop_pair(directory, human_readable_name=name)
    collection_removed = False
    if configs_dir.spec == DirectorySpec.EXISTS:
        collection_removed = delete_collection_configuration(
            configs_dir, generating_process_name=name, manager=manager
        )
    payload_exists = payload is not None and payload.spec == DirectorySpec.EXISTS
    if dropped is None and not collection_removed and not payload_exists:
        return False
    if payload is not None and payload.spec == DirectorySpec.EXISTS:
        shutil.rmtree(payload.path)
    return True
