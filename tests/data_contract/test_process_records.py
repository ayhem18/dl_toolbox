from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import artifact_stem
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object
from dl_toolbox.data_contract.process_records import (
    claim_generating_process,
    create_payload_directory,
    delete_incomplete_run,
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


def test_saved_record_is_exclusive_for_name_and_signature(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        directory = identity_index(manager, tmp_path / f"preparation_{i}")
        signature = {"strategy": "example", "args": random_json_object(rng)}
        configuration_hash = manager.hash_payload(signature)
        name = random_generating_process_name(rng)
        checked = claim_generating_process(
            manager,
            directory,
            generating_process_name=name,
            configuration_hash=configuration_hash,
        )
        saved = write_process_configuration(
            manager,
            directory,
            generating_process_name=checked,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        document = read_object(file=saved)
        assert document["name"] == checked
        assert document["hash"] == configuration_hash
        assert document["generating_process_signature"] == signature
        with pytest.raises(FileExistsError, match="pair already exists"):
            claim_generating_process(
                manager,
                directory,
                generating_process_name=checked,
                configuration_hash=configuration_hash,
            )
        other_name = checked + "_other"
        with pytest.raises(FileExistsError, match="already saved as name"):
            claim_generating_process(
                manager,
                directory,
                generating_process_name=other_name,
                configuration_hash=configuration_hash,
            )
        other_hash = manager.hash_payload({"strategy": "other", "args": {"i": i}})
        with pytest.raises(FileExistsError, match="already exists with hash"):
            claim_generating_process(
                manager,
                directory,
                generating_process_name=checked,
                configuration_hash=other_hash,
            )


def test_payload_directory_is_created_once(tmp_path: Path, rng: random.Random) -> None:
    for i in range(N_TRIALS):
        payload = missing_directory(
            tmp_path / "artifacts" / random_generating_process_name(rng) / str(i)
        )
        created = create_payload_directory(payload)
        assert created.spec == DirectorySpec.EXISTS
        assert created.path.is_dir()
        with pytest.raises(FileExistsError):
            create_payload_directory(created)


def test_incomplete_run_leaves_no_files(tmp_path: Path, rng: random.Random) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        name = random_generating_process_name(rng)
        payload = create_payload_directory(
            missing_directory(tmp_path / "artifacts" / f"{name}_{i}")
        )
        (payload.path / "image.npy").write_bytes(rng.randbytes(8))
        directory = identity_index(manager, tmp_path / f"preparation_{i}")
        configuration_hash = manager.hash_payload(
            {"i": i, "noise": rng.randint(0, 1000)}
        )
        write_process_configuration(
            manager,
            directory,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature={"i": i},
        )
        configs_dir = existing_directory(tmp_path / f"collections_{i}")
        write_object(
            directory=configs_dir,
            name=f"{name}.json",
            payload={"hash": configuration_hash},
            form=JsonTextForm.DOCUMENT,
        )
        write_object(
            directory=configs_dir,
            name=f"{artifact_stem(configuration_hash, prefix_len=manager.prefix_len)}.json",
            payload={"hash": configuration_hash},
            form=JsonTextForm.DOCUMENT,
        )
        delete_incomplete_run(
            manager=manager,
            directory=directory,
            generating_process_name=name,
            configs_dir=configs_dir,
            payload=payload,
        )
        assert not payload.path.exists()
        lookup = manager.lookup(
            directory, human_readable_name=name, hash=configuration_hash
        )
        assert lookup.state == HashManager.BOTH_FREE
        with pytest.raises(FileNotFoundError):
            File.from_absolute(path=configs_dir.path / f"{name}.json")


def test_hash_prefix_owned_by_a_different_digest_is_rejected(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = hash_manager()
    for i in range(N_TRIALS):
        directory = identity_index(manager, tmp_path / f"preparation_{i}")
        configuration_hash = manager.hash_payload(
            {"i": i, "noise": rng.randint(0, 1000)}
        )
        name = random_generating_process_name(rng)
        other_hash = manager.hash_payload({"other": i, "noise": rng.randint(0, 1000)})
        if other_hash == configuration_hash:
            other_hash = manager.hash_payload({"other": i, "extra": True})
        hashes = Directory.from_under(
            root=directory,
            relative_path=manager.hash_dir_name,
            policy=DirectoryParsePolicy.MUST_EXIST,
        )
        write_object(
            directory=hashes,
            name=f"{artifact_stem(configuration_hash, prefix_len=manager.prefix_len)}.json",
            payload={"name": "other", "hash": other_hash},
            form=JsonTextForm.DOCUMENT,
        )
        with pytest.raises(ValueError, match="already used by"):
            claim_generating_process(
                manager,
                directory,
                generating_process_name=name,
                configuration_hash=configuration_hash,
            )
