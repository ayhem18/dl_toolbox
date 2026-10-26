from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.data_contract.configs import hashed_artifact_path
from dl_toolbox.base.hashing.identity import hash_payload
from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.data_contract.process_records import (
    assert_generating_process_name_available,
    create_payload_directory,
    delete_incomplete_run,
    process_configuration_paths,
    save_process_configuration,
)

from tests.data_contract.dc_support import N_TRIALS, random_generating_process_name, random_json_object


def test_saved_record_is_exclusive_for_name_and_signature(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        process_dir = tmp_path / f"preparation_{i}"
        signature = {"strategy": "example", "args": random_json_object(rng)}
        configuration_hash = hash_payload(signature)
        name = random_generating_process_name(rng)
        assert_generating_process_name_available(
            generating_process_name=name,
            configuration_hash=configuration_hash,
            process_configurations_dir=process_dir,
        )
        saved = save_process_configuration(
            process_dir,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        document = read_json(saved)
        assert document["generating_process_name"] == name
        assert document["hash"] == configuration_hash
        assert document["generating_process_signature"] == signature
        with pytest.raises(FileExistsError, match="already saved"):
            assert_generating_process_name_available(
                generating_process_name=name,
                configuration_hash=configuration_hash,
                process_configurations_dir=process_dir,
            )
        other_name = name + "_other"
        with pytest.raises(FileExistsError, match="already saved as"):
            assert_generating_process_name_available(
                generating_process_name=other_name,
                configuration_hash=configuration_hash,
                process_configurations_dir=process_dir,
            )
        other_hash = hash_payload({"strategy": "other", "args": {"i": i}})
        with pytest.raises(FileExistsError, match="already exists with hash"):
            assert_generating_process_name_available(
                generating_process_name=name,
                configuration_hash=other_hash,
                process_configurations_dir=process_dir,
            )


def test_existing_payload_directory_blocks_a_new_claim(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        process_dir = tmp_path / f"preparation_{i}"
        name = random_generating_process_name(rng)
        payload_dir = tmp_path / "artifacts" / name
        payload_dir.mkdir(parents=True)
        with pytest.raises(FileExistsError, match="payload directory"):
            assert_generating_process_name_available(
                generating_process_name=name,
                configuration_hash=hash_payload({"i": i}),
                process_configurations_dir=process_dir,
                payload_dir=payload_dir,
            )


def test_payload_directory_is_created_once(tmp_path: Path, rng: random.Random) -> None:
    for i in range(N_TRIALS):
        payload_dir = tmp_path / "artifacts" / random_generating_process_name(rng) / str(i)
        created = create_payload_directory(payload_dir)
        assert created.is_dir()
        with pytest.raises(FileExistsError):
            create_payload_directory(payload_dir)


def test_incomplete_run_leaves_no_files(tmp_path: Path, rng: random.Random) -> None:
    for i in range(N_TRIALS):
        name = random_generating_process_name(rng)
        payload_dir = tmp_path / "artifacts" / f"{name}_{i}"
        create_payload_directory(payload_dir)
        (payload_dir / "image.npy").write_bytes(rng.randbytes(8))
        process_dir = tmp_path / f"preparation_{i}"
        configuration_hash = hash_payload({"i": i, "noise": rng.randint(0, 1000)})
        name_path, hash_path = process_configuration_paths(
            process_dir,
            generating_process_name=name,
            configuration_hash=configuration_hash,
        )
        write_json(name_path, {"generating_process_name": name})
        configs_dir = tmp_path / f"collections_{i}"
        collection_alias = configs_dir / f"{name}.json"
        collection_hashed = hashed_artifact_path(configs_dir, configuration_hash)
        write_json(collection_alias, {"hash": configuration_hash})
        write_json(collection_hashed, {"hash": configuration_hash})
        delete_incomplete_run(
            payload_dir=payload_dir,
            collection_hashed_path=collection_hashed,
            collection_alias_path=collection_alias,
            process_name_path=name_path,
            process_hash_path=hash_path,
        )
        assert not payload_dir.exists()
        assert not collection_alias.exists()
        assert not collection_hashed.exists()
        assert not name_path.exists()
        assert not hash_path.exists()


def test_hash_prefix_owned_by_a_different_digest_is_rejected(
    tmp_path: Path, rng: random.Random
) -> None:
    for i in range(N_TRIALS):
        process_dir = tmp_path / f"preparation_{i}"
        configuration_hash = hash_payload({"i": i, "noise": rng.randint(0, 1000)})
        name = random_generating_process_name(rng)
        _, hash_path = process_configuration_paths(
            process_dir,
            generating_process_name=name,
            configuration_hash=configuration_hash,
        )
        other_hash = hash_payload({"other": i, "noise": rng.randint(0, 1000)})
        if other_hash == configuration_hash:
            other_hash = hash_payload({"other": i, "extra": True})
        write_json(
            hash_path,
            {
                "generating_process_name": "other",
                "hash": other_hash,
                "generating_process_signature": {"strategy": "other"},
            },
        )
        with pytest.raises(ValueError, match="already used by"):
            assert_generating_process_name_available(
                generating_process_name=name,
                configuration_hash=configuration_hash,
                process_configurations_dir=process_dir,
            )
