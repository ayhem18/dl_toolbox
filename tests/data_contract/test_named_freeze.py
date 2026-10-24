from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.data_contract.configs import hashed_artifact_path
from dl_toolbox.data_contract.hashing import hash_payload
from dl_toolbox.data_contract.json_io import write_json
from dl_toolbox.data_contract.named_freeze import (
    process_hash_occupant,
    remove_named_freeze,
)
from dl_toolbox.data_contract.process_records import (
    assert_generating_process_name_available,
    create_payload_directory,
    save_process_configuration,
)

from _support import N_TRIALS, random_generating_process_name, random_json_object


def _write_collection_pair(
    configs_dir: Path, *, name: str, collection_hash: str
) -> None:
    document = {
        "generating_process_name": name,
        "hash": collection_hash,
        "generating_process_signature": {"strategy": "fixture"},
    }
    write_json(configs_dir / f"{name}.json", document)
    write_json(hashed_artifact_path(configs_dir, collection_hash), document)


def test_missing_name_is_absent(tmp_path: Path, rng: random.Random) -> None:
    for i in range(N_TRIALS):
        removed = remove_named_freeze(
            generating_process_name=random_generating_process_name(rng),
            process_configurations_dir=tmp_path / f"preparation_{i}",
            collection_configs_dir=tmp_path / f"collections_{i}",
        )
        assert removed is False


def test_json_only_name_deletes_process_and_collection_files(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        process_dir = tmp_path / f"preparation_{i}"
        configs_dir = tmp_path / f"collections_{i}"
        signature = {"strategy": "fixture", "args": random_json_object(rng)}
        configuration_hash = hash_payload(signature)
        name = random_generating_process_name(rng)
        save_process_configuration(
            process_dir,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        collection_hash = hash_payload({"members": {name: i}})
        _write_collection_pair(configs_dir, name=name, collection_hash=collection_hash)
        removed = remove_named_freeze(
            generating_process_name=name,
            process_configurations_dir=process_dir,
            collection_configs_dir=configs_dir,
        )
        assert removed is True
        assert list((process_dir / "human_readable").glob("*.json")) == []
        assert list((process_dir / "hashes").glob("*.json")) == []
        assert list(configs_dir.glob("*.json")) == []
        assert_generating_process_name_available(
            generating_process_name=name,
            configuration_hash=configuration_hash,
            process_configurations_dir=process_dir,
        )


def test_file_backed_name_deletes_payload_directory(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        process_dir = tmp_path / f"preparation_{i}"
        configs_dir = tmp_path / f"collections_{i}"
        name = random_generating_process_name(rng)
        payload_dir = tmp_path / "artifacts" / f"{name}_{i}"
        create_payload_directory(payload_dir)
        (payload_dir / "image.npy").write_bytes(rng.randbytes(8))
        signature = {"strategy": "fixture", "args": {"i": i}}
        configuration_hash = hash_payload(signature)
        save_process_configuration(
            process_dir,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        _write_collection_pair(
            configs_dir, name=name, collection_hash=hash_payload({"i": i})
        )
        removed = remove_named_freeze(
            generating_process_name=name,
            process_configurations_dir=process_dir,
            collection_configs_dir=configs_dir,
            payload_dir=payload_dir,
        )
        assert removed is True
        assert not payload_dir.exists()


def test_relative_payload_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        remove_named_freeze(
            generating_process_name="used_name",
            process_configurations_dir=tmp_path / "preparation",
            collection_configs_dir=tmp_path / "collections",
            payload_dir=Path("relative_payload"),
        )


def test_does_not_delete_a_different_name(tmp_path: Path, rng: random.Random) -> None:
    process_dir = tmp_path / "preparation"
    configs_dir = tmp_path / "collections"
    kept_signature = {"strategy": "kept", "args": random_json_object(rng)}
    kept_hash = hash_payload(kept_signature)
    kept_name = random_generating_process_name(rng)
    save_process_configuration(
        process_dir,
        generating_process_name=kept_name,
        configuration_hash=kept_hash,
        generating_process_signature=kept_signature,
    )
    _write_collection_pair(
        configs_dir, name=kept_name, collection_hash=hash_payload({"kept": True})
    )
    deleted_signature = {"strategy": "deleted", "args": random_json_object(rng)}
    deleted_hash = hash_payload(deleted_signature)
    deleted_name = kept_name + "_gone"
    save_process_configuration(
        process_dir,
        generating_process_name=deleted_name,
        configuration_hash=deleted_hash,
        generating_process_signature=deleted_signature,
    )
    _write_collection_pair(
        configs_dir, name=deleted_name, collection_hash=hash_payload({"gone": True})
    )
    remove_named_freeze(
        generating_process_name=deleted_name,
        process_configurations_dir=process_dir,
        collection_configs_dir=configs_dir,
    )
    assert (process_dir / "human_readable" / f"{kept_name}.json").is_file()
    assert hashed_artifact_path(process_dir / "hashes", kept_hash).is_file()
    assert (configs_dir / f"{kept_name}.json").is_file()


def test_process_hash_occupant_returns_the_stored_name(
    tmp_path: Path, rng: random.Random
) -> None:
    process_dir = tmp_path / "preparation"
    configs_dir = tmp_path / "collections"
    signature = {"strategy": "image_sides", "args": random_json_object(rng)}
    configuration_hash = hash_payload(signature)
    occupant = "local_item_collection"
    save_process_configuration(
        process_dir,
        generating_process_name=occupant,
        configuration_hash=configuration_hash,
        generating_process_signature=signature,
    )
    _write_collection_pair(
        configs_dir, name=occupant, collection_hash=hash_payload({"members": 1})
    )
    assert (
        process_hash_occupant(
            configuration_hash=configuration_hash,
            process_configurations_dir=process_dir,
        )
        == occupant
    )
    removed = remove_named_freeze(
        generating_process_name=occupant,
        process_configurations_dir=process_dir,
        collection_configs_dir=configs_dir,
    )
    assert removed is True
    assert (
        process_hash_occupant(
            configuration_hash=configuration_hash,
            process_configurations_dir=process_dir,
        )
        is None
    )
    assert_generating_process_name_available(
        generating_process_name="item_collection",
        configuration_hash=configuration_hash,
        process_configurations_dir=process_dir,
    )


def test_missing_process_hash_has_no_occupant(
    tmp_path: Path, rng: random.Random
) -> None:
    configuration_hash = hash_payload({"missing": rng.randint(0, 1000)})
    assert (
        process_hash_occupant(
            configuration_hash=configuration_hash,
            process_configurations_dir=tmp_path / "preparation",
        )
        is None
    )
