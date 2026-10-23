"""Process-configuration claim, write-last, and incomplete-run cleanup."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from dl_toolbox.data_contract.configs import hashed_artifact_path
from dl_toolbox.data_contract.hashing import (
    DEFAULT_ARTIFACT_PREFIX_LEN,
    is_full_sha256,
)
from dl_toolbox.data_contract.json_io import read_json, write_json
from dl_toolbox.data_contract.names import assert_generating_process_name

HUMAN_READABLE_DIRNAME = "human_readable"
HASHES_DIRNAME = "hashes"


def human_readable_configurations_dir(process_configurations_dir: Path) -> Path:
    return process_configurations_dir / HUMAN_READABLE_DIRNAME


def hash_configurations_dir(process_configurations_dir: Path) -> Path:
    return process_configurations_dir / HASHES_DIRNAME


def process_configuration_paths(
    process_configurations_dir: Path,
    *,
    generating_process_name: str,
    configuration_hash: str,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> tuple[Path, Path]:
    name = assert_generating_process_name(
        generating_process_name, prefix_len=prefix_len
    )
    if not is_full_sha256(configuration_hash):
        raise ValueError(
            f"configuration_hash must be a 64-char lowercase SHA-256, got "
            f"{configuration_hash!r}"
        )
    name_path = (
        human_readable_configurations_dir(process_configurations_dir) / f"{name}.json"
    )
    hash_path = hashed_artifact_path(
        hash_configurations_dir(process_configurations_dir),
        configuration_hash,
        prefix_len=prefix_len,
    )
    return name_path, hash_path


def process_configuration_document(
    *,
    generating_process_name: str,
    configuration_hash: str,
    generating_process_signature: dict,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> dict[str, Any]:
    name = assert_generating_process_name(
        generating_process_name, prefix_len=prefix_len
    )
    if not is_full_sha256(configuration_hash):
        raise ValueError(
            f"configuration_hash must be a 64-char lowercase SHA-256, got "
            f"{configuration_hash!r}"
        )
    if not isinstance(generating_process_signature, dict):
        raise TypeError("generating_process_signature must be a dict")
    return {
        "generating_process_name": name,
        "hash": configuration_hash,
        "generating_process_signature": generating_process_signature,
    }


def assert_generating_process_name_available(
    *,
    generating_process_name: str,
    configuration_hash: str,
    process_configurations_dir: Path,
    payload_dir: Path | None = None,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> str:
    name = assert_generating_process_name(
        generating_process_name, prefix_len=prefix_len
    )
    if payload_dir is not None:
        if not Path(payload_dir).is_absolute():
            raise ValueError(f"payload_dir must be an absolute path, got {payload_dir!s}")
        if payload_dir.exists():
            raise FileExistsError(f"payload directory already exists: {payload_dir}")
    name_path, hash_path = process_configuration_paths(
        process_configurations_dir,
        generating_process_name=name,
        configuration_hash=configuration_hash,
        prefix_len=prefix_len,
    )
    name_exists = name_path.is_file()
    hash_exists = hash_path.is_file()
    if name_exists and hash_exists:
        name_document = read_json(name_path)
        hash_document = read_json(hash_path)
        if (
            name_document != hash_document
            or name_document["generating_process_name"] != name
            or name_document["hash"] != configuration_hash
        ):
            raise ValueError(
                f"process configuration generating_process_name {name!r} and hash "
                f"{configuration_hash} disagree"
            )
        raise FileExistsError(
            f"already saved generating_process_name {name!r} with this "
            "generating_process_signature"
        )
    if name_exists:
        name_document = read_json(name_path)
        raise FileExistsError(
            f"generating_process_name {name!r} already exists with hash "
            f"{name_document['hash']} generating_process_signature="
            f"{name_document['generating_process_signature']}"
        )
    if hash_exists:
        hash_document = read_json(hash_path)
        if hash_document["hash"] != configuration_hash:
            raise ValueError(
                f"hash prefix {hash_path.stem!r} already used by "
                f"{hash_document['hash']}"
            )
        raise FileExistsError(
            "generating_process_signature already saved as generating_process_name "
            f"{hash_document['generating_process_name']!r}"
        )
    return name


def create_payload_directory(payload_dir: Path) -> Path:
    if not payload_dir.is_absolute():
        raise ValueError(f"payload_dir must be an absolute path, got {payload_dir!s}")
    payload_dir.parent.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(exist_ok=False)
    return payload_dir


def save_process_configuration(
    process_configurations_dir: Path,
    *,
    generating_process_name: str,
    configuration_hash: str,
    generating_process_signature: dict,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> Path:
    document = process_configuration_document(
        generating_process_name=generating_process_name,
        configuration_hash=configuration_hash,
        generating_process_signature=generating_process_signature,
        prefix_len=prefix_len,
    )
    name_path, hash_path = process_configuration_paths(
        process_configurations_dir,
        generating_process_name=document["generating_process_name"],
        configuration_hash=configuration_hash,
        prefix_len=prefix_len,
    )
    write_json(hash_path, document)
    write_json(name_path, document)
    return name_path


def delete_incomplete_run(
    *,
    payload_dir: Path | None = None,
    collection_hashed_path: Path | None = None,
    collection_alias_path: Path | None = None,
    process_name_path: Path | None = None,
    process_hash_path: Path | None = None,
) -> None:
    if payload_dir is not None and payload_dir.exists():
        shutil.rmtree(payload_dir)
    for path in (
        collection_hashed_path,
        collection_alias_path,
        process_name_path,
        process_hash_path,
    ):
        if path is not None:
            path.unlink(missing_ok=True)
