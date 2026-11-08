from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import artifact_stem
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.data_contract.configs import (
    load_collection_configuration,
    write_collection_configuration,
)

from tests.data_contract.dc_support import (
    N_TRIALS,
    existing_directory,
    hash_manager,
    random_generating_process_name,
    random_json_object,
)


class _Collection:
    def __init__(
        self,
        configs_dir: Directory,
        generating_process_name: str,
        *,
        signature: dict,
        members: dict,
        manager: HashManager,
    ) -> None:
        self._configs_dir = configs_dir
        self.generating_process_name = generating_process_name
        self.generating_process_signature = signature
        self._members = members
        self.collection_hash = manager.hash_payload(
            {
                "generating_process_signature": self.generating_process_signature,
                "members": self._members,
            }
        )

    def configs_dir(self) -> Directory:
        return self._configs_dir

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": self.generating_process_signature,
            "members": self._members,
        }


def test_written_configuration_loads_by_name_hash_and_prefix(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        configs_dir = existing_directory(tmp_path / f"collections_{i}")
        name = random_generating_process_name(rng)
        signature = {"strategy": "example", "args": random_json_object(rng)}
        members = {f"item_{rng.randint(0, 10_000)}": random_json_object(rng)}
        collection = _Collection(
            configs_dir,
            name,
            signature=signature,
            members=members,
            manager=manager,
        )
        hashed_file = write_collection_configuration(collection, manager=manager)
        stored = read_object(file=hashed_file)
        assert stored["hash"] == collection.collection_hash
        assert stored["generating_process_name"] == name
        prefix = artifact_stem(
            collection.collection_hash, prefix_len=manager.prefix_len
        )
        assert hashed_file.path.name == f"{prefix}.json"
        by_name = load_collection_configuration(
            configs_dir, name, manager=manager
        )
        by_hash = load_collection_configuration(
            configs_dir, collection.collection_hash, manager=manager
        )
        by_prefix = load_collection_configuration(
            configs_dir, prefix, manager=manager
        )
        assert by_name[1] == stored
        assert by_hash[0].path == hashed_file.path
        assert by_hash[1] == stored
        assert by_prefix[0].path == hashed_file.path
        assert by_prefix[1] == stored


def test_collection_hash_ignores_generating_process_name(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        signature = {"strategy": "example", "args": random_json_object(rng)}
        members = {"item_1": random_json_object(rng)}
        left = _Collection(
            existing_directory(tmp_path / f"a_{i}"),
            random_generating_process_name(rng),
            signature=signature,
            members=members,
            manager=manager,
        )
        right = _Collection(
            existing_directory(tmp_path / f"b_{i}"),
            random_generating_process_name(rng),
            signature=signature,
            members=members,
            manager=manager,
        )
        assert left.collection_hash == right.collection_hash


def test_truncated_hash_is_rejected(tmp_path: Path) -> None:
    manager = hash_manager()

    class _ShortHash:
        generating_process_name = "run_v1"
        collection_hash = "abcd" * 4

        def configs_dir(self) -> Directory:
            return existing_directory(tmp_path / "short")

        def to_dict(self) -> dict:
            return {
                "hash": self.collection_hash,
                "generating_process_name": self.generating_process_name,
            }

    with pytest.raises(ValueError, match="64-char"):
        write_collection_configuration(_ShortHash(), manager=manager)


def test_name_mismatch_is_rejected(tmp_path: Path, rng: random.Random) -> None:
    manager = hash_manager()
    for _ in range(N_TRIALS):
        left = random_generating_process_name(rng)
        right = random_generating_process_name(rng)
        if left == right:
            right = left + "_b"
        digest = manager.hash_payload(random_json_object(rng))

        class _Mismatch:
            generating_process_name = right
            collection_hash = digest

            def configs_dir(self) -> Directory:
                return existing_directory(tmp_path / "mismatch")

            def to_dict(self) -> dict:
                return {
                    "hash": digest,
                    "generating_process_name": left,
                }

        with pytest.raises(ValueError, match="generating_process_name"):
            write_collection_configuration(_Mismatch(), manager=manager)
