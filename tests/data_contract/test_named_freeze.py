from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import DirectorySpec
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import artifact_stem
from dl_toolbox.data_contract.configs import write_collection_configuration
from dl_toolbox.data_contract.named_freeze import (
    process_hash_occupant,
    remove_named_freeze,
)
from dl_toolbox.data_contract.process_records import (
    claim_generating_process,
    create_payload_directory,
    write_process_configuration,
)

from tests.data_contract.dc_support import (
    N_TRIALS,
    existing_directory,
    hash_manager,
    identity_index,
    missing_directory,
    random_generating_process_name,
    random_json_object,
)


class _Collection:
    def __init__(
        self,
        configs_dir,
        generating_process_name: str,
        *,
        collection_hash: str,
        manager: HashManager,
    ) -> None:
        self._configs_dir = configs_dir
        self.generating_process_name = generating_process_name
        self.collection_hash = collection_hash
        self._manager = manager

    def configs_dir(self):
        return self._configs_dir

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": {"strategy": "fixture"},
        }


def _write_collection(
    configs_dir, *, name: str, collection_hash: str, manager: HashManager
) -> None:
    write_collection_configuration(
        _Collection(
            configs_dir,
            name,
            collection_hash=collection_hash,
            manager=manager,
        ),
        manager=manager,
    )


def test_missing_name_is_absent(tmp_path: Path, rng: random.Random) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        directory = identity_index(manager, tmp_path / f"preparation_{i}")
        configs_dir = missing_directory(tmp_path / f"collections_{i}")
        removed = remove_named_freeze(
            manager,
            directory,
            generating_process_name=random_generating_process_name(rng),
            configs_dir=configs_dir,
            payload=None,
        )
        assert removed is False


def test_json_only_name_deletes_process_and_collection_files(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        directory = identity_index(manager, tmp_path / f"preparation_{i}")
        configs_dir = existing_directory(tmp_path / f"collections_{i}")
        signature = {"strategy": "fixture", "args": random_json_object(rng)}
        configuration_hash = manager.hash_payload(signature)
        name = random_generating_process_name(rng)
        write_process_configuration(
            manager,
            directory,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        collection_hash = manager.hash_payload({"members": {name: i}})
        _write_collection(
            configs_dir, name=name, collection_hash=collection_hash, manager=manager
        )
        removed = remove_named_freeze(
            manager,
            directory,
            generating_process_name=name,
            configs_dir=configs_dir,
            payload=None,
        )
        assert removed is True
        assert (
            manager.lookup(
                directory, human_readable_name=name, hash=configuration_hash
            ).state
            == HashManager.BOTH_FREE
        )
        with pytest.raises(FileNotFoundError):
            File.from_absolute(path=configs_dir.path / f"{name}.json")
        claim_generating_process(
            manager,
            directory,
            generating_process_name=name,
            configuration_hash=configuration_hash,
        )


def test_file_backed_name_deletes_payload_directory(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        directory = identity_index(manager, tmp_path / f"preparation_{i}")
        configs_dir = existing_directory(tmp_path / f"collections_{i}")
        name = random_generating_process_name(rng)
        payload = create_payload_directory(
            missing_directory(tmp_path / "artifacts" / f"{name}_{i}")
        )
        (payload.path / "image.npy").write_bytes(rng.randbytes(8))
        signature = {"strategy": "fixture", "args": {"i": i}}
        configuration_hash = manager.hash_payload(signature)
        write_process_configuration(
            manager,
            directory,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        _write_collection(
            configs_dir,
            name=name,
            collection_hash=manager.hash_payload({"i": i}),
            manager=manager,
        )
        removed = remove_named_freeze(
            manager,
            directory,
            generating_process_name=name,
            configs_dir=configs_dir,
            payload=payload,
        )
        assert removed is True
        assert not payload.path.exists()


def test_does_not_delete_a_different_name(tmp_path: Path, rng: random.Random) -> None:
    manager = hash_manager()
    directory = identity_index(manager, tmp_path / "preparation")
    configs_dir = existing_directory(tmp_path / "collections")
    kept_signature = {"strategy": "kept", "args": random_json_object(rng)}
    kept_hash = manager.hash_payload(kept_signature)
    kept_name = random_generating_process_name(rng)
    write_process_configuration(
        manager,
        directory,
        generating_process_name=kept_name,
        configuration_hash=kept_hash,
        generating_process_signature=kept_signature,
    )
    kept_collection_hash = manager.hash_payload({"kept": True})
    _write_collection(
        configs_dir,
        name=kept_name,
        collection_hash=kept_collection_hash,
        manager=manager,
    )
    deleted_signature = {"strategy": "deleted", "args": random_json_object(rng)}
    deleted_hash = manager.hash_payload(deleted_signature)
    deleted_name = kept_name + "_gone"
    write_process_configuration(
        manager,
        directory,
        generating_process_name=deleted_name,
        configuration_hash=deleted_hash,
        generating_process_signature=deleted_signature,
    )
    _write_collection(
        configs_dir,
        name=deleted_name,
        collection_hash=manager.hash_payload({"gone": True}),
        manager=manager,
    )
    remove_named_freeze(
        manager,
        directory,
        generating_process_name=deleted_name,
        configs_dir=configs_dir,
        payload=None,
    )
    assert (
        manager.lookup(
            directory, human_readable_name=kept_name, hash=kept_hash
        ).state
        == HashManager.BOTH_MATCH
    )
    File.from_absolute(path=configs_dir.path / f"{kept_name}.json")
    File.from_absolute(
        path=configs_dir.path
        / f"{artifact_stem(kept_collection_hash, prefix_len=manager.prefix_len)}.json"
    )


def test_process_hash_occupant_returns_the_stored_name(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    directory = identity_index(manager, tmp_path / "preparation")
    configs_dir = existing_directory(tmp_path / "collections")
    signature = {"strategy": "image_sides", "args": random_json_object(rng)}
    configuration_hash = manager.hash_payload(signature)
    occupant = "local_item_collection"
    write_process_configuration(
        manager,
        directory,
        generating_process_name=occupant,
        configuration_hash=configuration_hash,
        generating_process_signature=signature,
    )
    _write_collection(
        configs_dir,
        name=occupant,
        collection_hash=manager.hash_payload({"members": 1}),
        manager=manager,
    )
    assert (
        process_hash_occupant(
            manager, directory, configuration_hash=configuration_hash
        )
        == occupant
    )
    removed = remove_named_freeze(
        manager,
        directory,
        generating_process_name=occupant,
        configs_dir=configs_dir,
        payload=None,
    )
    assert removed is True
    assert (
        process_hash_occupant(
            manager, directory, configuration_hash=configuration_hash
        )
        is None
    )
    claim_generating_process(
        manager,
        directory,
        generating_process_name="item_collection",
        configuration_hash=configuration_hash,
    )


def test_missing_process_hash_has_no_occupant(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    directory = missing_directory(tmp_path / "preparation")
    assert directory.spec == DirectorySpec.MISSING
    configuration_hash = manager.hash_payload({"missing": rng.randint(0, 1000)})
    assert (
        process_hash_occupant(
            manager, directory, configuration_hash=configuration_hash
        )
        is None
    )
