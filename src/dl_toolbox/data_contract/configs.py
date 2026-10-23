"""Collection configuration files: hash-prefix filename plus generating_process_name alias."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dl_toolbox.data_contract.hashing import (
    DEFAULT_ARTIFACT_PREFIX_LEN,
    artifact_stem,
    is_full_sha256,
    is_hashed_artifact_stem,
)
from dl_toolbox.data_contract.json_io import read_json, write_json
from dl_toolbox.data_contract.names import assert_generating_process_name


def hashed_artifact_path(
    artifact_dir: Path,
    full_hash: str,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> Path:
    return artifact_dir / f"{artifact_stem(full_hash, prefix_len=prefix_len)}.json"


def list_hashed_artifacts(
    artifact_dir: Path,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> list[Path]:
    if not artifact_dir.is_dir():
        return []
    return sorted(
        path
        for path in artifact_dir.glob("*.json")
        if is_hashed_artifact_stem(path.stem, prefix_len=prefix_len)
    )


def resolve_artifact_path(
    artifact_dir: Path,
    key: str,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> Path:
    text = str(key).strip()
    if not text:
        raise ValueError("identity is empty")
    direct = artifact_dir / f"{text}.json"
    if direct.is_file():
        return direct
    if len(text) >= prefix_len and is_full_sha256(text):
        prefixed = hashed_artifact_path(artifact_dir, text, prefix_len=prefix_len)
        if prefixed.is_file():
            return prefixed
    if len(text) >= prefix_len:
        prefixed = artifact_dir / f"{text[:prefix_len]}.json"
        if prefixed.is_file() and is_hashed_artifact_stem(
            prefixed.stem, prefix_len=prefix_len
        ):
            return prefixed
    raise FileNotFoundError(f"No configuration {text!r} under {artifact_dir}")


def load_artifact(
    artifact_dir: Path,
    key: str,
    *,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> tuple[Path, dict[str, Any]]:
    path = resolve_artifact_path(artifact_dir, key, prefix_len=prefix_len)
    payload = read_json(path)
    if "hash" not in payload:
        raise ValueError(f"configuration {path} has no 'hash' field")
    stored_hash = payload["hash"]
    if not isinstance(stored_hash, str) or not stored_hash:
        raise ValueError(f"configuration {path} has no string 'hash' field")
    if is_hashed_artifact_stem(path.stem, prefix_len=prefix_len):
        if not stored_hash.startswith(path.stem):
            raise ValueError(
                f"stem {path.stem!r} does not match hash {stored_hash!r} in {path}"
            )
    requested = str(key).strip()
    if _is_hex_key(requested, prefix_len=prefix_len):
        if stored_hash != requested and not stored_hash.startswith(requested):
            raise ValueError(
                f"requested key {requested!r} does not match hash {stored_hash!r}"
            )
    return path, payload


def _is_hex_key(key: str, *, prefix_len: int) -> bool:
    if is_hashed_artifact_stem(key, prefix_len=prefix_len):
        return True
    if len(key) <= prefix_len or key != key.lower():
        return False
    try:
        int(key, 16)
    except ValueError:
        return False
    return True


def save_hashed_artifact(
    artifact_dir: Path,
    payload: Mapping[str, Any],
    *,
    full_hash: str,
    generating_process_name: str,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> Path:
    document = dict(payload)
    if document["hash"] != full_hash:
        raise ValueError("payload['hash'] must equal full_hash")
    name = assert_generating_process_name(
        generating_process_name, prefix_len=prefix_len
    )
    if document["generating_process_name"] != name:
        raise ValueError(
            "payload['generating_process_name'] must equal generating_process_name"
        )
    hashed_path = hashed_artifact_path(artifact_dir, full_hash, prefix_len=prefix_len)
    write_json(hashed_path, document)
    write_json(artifact_dir / f"{name}.json", document)
    return hashed_path


def write_collection_configuration(collection: object) -> Path:
    """Process helper: write the collection identity document.

    Collections have ``to_dict`` / ``from_payload`` / ``load``. They do not have
    ``save``. Tests that build a collection without a process use this helper.
    """
    generating_process_name = collection.generating_process_name
    collection_hash = collection.collection_hash
    document = collection.to_dict()
    if document["hash"] != collection_hash:
        raise ValueError("collection.to_dict()['hash'] must equal collection_hash")
    if document["generating_process_name"] != generating_process_name:
        raise ValueError(
            "collection.to_dict()['generating_process_name'] must equal "
            "generating_process_name"
        )
    return save_hashed_artifact(
        collection.configs_dir(),
        document,
        full_hash=collection_hash,
        generating_process_name=generating_process_name,
    )
