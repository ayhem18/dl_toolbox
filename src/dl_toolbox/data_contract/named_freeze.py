"""Delete a finished freeze so a later ``process()`` can claim it.

This is not ``process(generating_process_name=...)``. ``process()`` still has
no overwrite flag.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dl_toolbox.data_contract.configs import hashed_artifact_path, list_hashed_artifacts
from dl_toolbox.base.hashing.identity import (
    DEFAULT_ARTIFACT_PREFIX_LEN,
    is_full_sha256,
)
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.data_contract.names import assert_generating_process_name
from dl_toolbox.data_contract.process_records import (
    hash_configurations_dir,
    human_readable_configurations_dir,
)


def process_hash_occupant(
    *,
    configuration_hash: str,
    process_configurations_dir: Path,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> str | None:
    """Return the ``generating_process_name`` that holds this process hash."""
    if not is_full_sha256(configuration_hash):
        raise ValueError(
            f"configuration_hash must be a 64-char lowercase SHA-256, got "
            f"{configuration_hash!r}"
        )
    hash_path = hashed_artifact_path(
        hash_configurations_dir(process_configurations_dir),
        configuration_hash,
        prefix_len=prefix_len,
    )
    if not hash_path.is_file():
        return None
    document = read_json(hash_path)
    stored_hash = document["hash"]
    if stored_hash != configuration_hash:
        raise ValueError(
            f"hash prefix {hash_path.stem!r} already used by {stored_hash}"
        )
    return assert_generating_process_name(
        document["generating_process_name"], prefix_len=prefix_len
    )


def remove_named_freeze(
    *,
    generating_process_name: str,
    process_configurations_dir: Path,
    collection_configs_dir: Path,
    payload_dir: Path | None = None,
    prefix_len: int = DEFAULT_ARTIFACT_PREFIX_LEN,
) -> bool:
    """Delete process pair, collection alias, hashed collection, payload if any.

    Missing names return ``False``. Payload is a directory the caller already
    resolved; this helper does not read an artifact path from the collection
    document.
    """
    name = assert_generating_process_name(
        generating_process_name, prefix_len=prefix_len
    )
    resolved_payload = _absolute_payload_dir(payload_dir)
    human_readable = human_readable_configurations_dir(process_configurations_dir)
    hashes = hash_configurations_dir(process_configurations_dir)
    process_name_path = human_readable / f"{name}.json"

    process_hash_path: Path | None = None
    if process_name_path.is_file():
        process_document = read_json(process_name_path)
        process_hash_path = hashed_artifact_path(
            hashes,
            _require_full_hash(process_document["hash"], label="process hash"),
            prefix_len=prefix_len,
        )

    collection_alias_path = collection_configs_dir / f"{name}.json"
    collection_hash_path: Path | None = None
    if collection_alias_path.is_file():
        collection_document = read_json(collection_alias_path)
        collection_hash_path = hashed_artifact_path(
            collection_configs_dir,
            _require_full_hash(
                collection_document["hash"], label="collection hash"
            ),
            prefix_len=prefix_len,
        )

    hash_paths_for_name = _process_hash_paths_for_name(
        hashes, name, prefix_len=prefix_len
    )
    payload_exists = resolved_payload is not None and resolved_payload.exists()
    existed = (
        process_name_path.is_file()
        or collection_alias_path.is_file()
        or payload_exists
        or process_hash_path is not None
        or len(hash_paths_for_name) > 0
    )
    if not existed:
        return False

    if resolved_payload is not None and resolved_payload.exists():
        shutil.rmtree(resolved_payload)
    if collection_hash_path is not None:
        collection_hash_path.unlink(missing_ok=True)
    collection_alias_path.unlink(missing_ok=True)
    if process_hash_path is not None:
        _delete_matching_process_hash(process_hash_path, name)
    n_hash_paths = len(hash_paths_for_name)
    for i in range(n_hash_paths):
        _delete_matching_process_hash(hash_paths_for_name[i], name)
    process_name_path.unlink(missing_ok=True)
    return True


def _absolute_payload_dir(payload_dir: Path | None) -> Path | None:
    if payload_dir is None:
        return None
    resolved = Path(payload_dir)
    if not resolved.is_absolute():
        raise ValueError(f"payload_dir must be an absolute path, got {resolved!s}")
    return resolved


def _require_full_hash(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not is_full_sha256(raw):
        raise ValueError(f"{label} must be a 64-char lowercase SHA-256, got {raw!r}")
    return raw


def _process_hash_paths_for_name(
    hashes_dir: Path, name: str, *, prefix_len: int
) -> tuple[Path, ...]:
    paths = list_hashed_artifacts(hashes_dir, prefix_len=prefix_len)
    n_paths = len(paths)
    matches: list[Path] = []
    for i in range(n_paths):
        document = read_json(paths[i])
        if "generating_process_name" not in document:
            continue
        if document["generating_process_name"] == name:
            matches.append(paths[i])
    return tuple(matches)


def _delete_matching_process_hash(path: Path, name: str) -> None:
    if not path.is_file():
        return
    document = read_json(path)
    if (
        "generating_process_name" in document
        and document["generating_process_name"] != name
    ):
        return
    path.unlink()
