from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.data_contract.configs import (
    hashed_artifact_path,
    list_hashed_artifacts,
    load_artifact,
    resolve_artifact_path,
    save_hashed_artifact,
    write_collection_configuration,
)
from dl_toolbox.data_contract.hashing import artifact_stem, hash_payload
from dl_toolbox.data_contract.json_io import read_json

from _support import N_TRIALS, random_generating_process_name, random_json_object


class _Collection:
    def __init__(
        self,
        configs_dir: Path,
        generating_process_name: str,
        *,
        signature: dict,
        members: dict,
    ) -> None:
        self._configs_dir = configs_dir
        self.generating_process_name = generating_process_name
        self.generating_process_signature = signature
        self._members = members
        self.collection_hash = hash_payload(
            {
                "generating_process_signature": self.generating_process_signature,
                "members": self._members,
            }
        )

    def configs_dir(self) -> Path:
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
    for i in range(N_TRIALS):
        configs_dir = tmp_path / f"collections_{i}"
        name = random_generating_process_name(rng)
        signature = {"strategy": "example", "args": random_json_object(rng)}
        members = {f"item_{rng.randint(0, 10_000)}": random_json_object(rng)}
        collection = _Collection(
            configs_dir,
            name,
            signature=signature,
            members=members,
        )
        hashed_path = write_collection_configuration(collection)
        alias = configs_dir / f"{name}.json"
        stored = read_json(hashed_path)
        assert stored["hash"] == collection.collection_hash
        assert stored["generating_process_name"] == name
        assert hashed_path == hashed_artifact_path(
            configs_dir, collection.collection_hash
        )
        assert resolve_artifact_path(configs_dir, name) == alias
        assert (
            resolve_artifact_path(configs_dir, collection.collection_hash)
            == hashed_path
        )
        assert (
            resolve_artifact_path(
                configs_dir, artifact_stem(collection.collection_hash)
            )
            == hashed_path
        )
        _, loaded = load_artifact(configs_dir, name)
        assert loaded == stored
        assert list_hashed_artifacts(configs_dir) == [hashed_path]


def test_collection_hash_ignores_generating_process_name(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        signature = {"strategy": "example", "args": random_json_object(rng)}
        members = {"item_1": random_json_object(rng)}
        left = _Collection(
            tmp_path / f"a_{i}",
            random_generating_process_name(rng),
            signature=signature,
            members=members,
        )
        right = _Collection(
            tmp_path / f"b_{i}",
            random_generating_process_name(rng),
            signature=signature,
            members=members,
        )
        assert left.collection_hash == right.collection_hash


def test_truncated_identity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64-char"):
        save_hashed_artifact(
            tmp_path,
            {
                "hash": "abcd" * 4,
                "generating_process_name": "run_v1",
            },
            full_hash="abcd" * 4,
            generating_process_name="run_v1",
        )


def test_name_mismatch_is_rejected(tmp_path: Path, rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        digest = hash_payload(random_json_object(rng))
        left = random_generating_process_name(rng)
        right = random_generating_process_name(rng)
        if left == right:
            right = left + "_b"
        with pytest.raises(ValueError, match="generating_process_name"):
            save_hashed_artifact(
                tmp_path,
                {"hash": digest, "generating_process_name": left},
                full_hash=digest,
                generating_process_name=right,
            )
