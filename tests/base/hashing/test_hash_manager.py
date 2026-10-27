from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.hashing.hash_manager import (
    BOTH_FREE,
    BOTH_MATCH,
    HASH_TAKEN,
    NAME_TAKEN,
    HashManager,
)
from dl_toolbox.base.input_output.json_io import read_json

from tests.base.base_support import N_TRIALS, random_generating_process_name, random_json_object


def _manager() -> HashManager:
    return HashManager(
        prefix_len=12,
        human_readable_dirname="human_readable",
        hashes_dirname="hashes",
    )


def test_write_and_lookup_unique_pair(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = tmp_path / f"records_{i}"
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(random_json_object(rng))
        lookup = manager.lookup_name_hash_json(
            directory, name=name, configuration_hash=configuration_hash
        )
        assert lookup.state == BOTH_FREE
        manager.verify_name_hash_json_unique(
            directory, name=name, configuration_hash=configuration_hash
        )
        name_path, hash_path = manager.write_name_hash_json_unique(
            directory,
            name=name,
            configuration_hash=configuration_hash,
            extra={"note": i},
        )
        document = read_json(name_path)
        assert document == read_json(hash_path)
        assert document["name"] == name
        assert document["hash"] == configuration_hash
        assert document["note"] == i
        matched = manager.lookup_name_hash_json(
            directory, name=name, configuration_hash=configuration_hash
        )
        assert matched.state == BOTH_MATCH
        with pytest.raises(FileExistsError, match="already claimed"):
            manager.verify_name_hash_json_unique(
                directory, name=name, configuration_hash=configuration_hash
            )


def test_name_taken_and_hash_taken(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = tmp_path / f"taken_{i}"
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload({"i": i, "noise": rng.randint(0, 1000)})
        manager.write_name_hash_json_unique(
            directory, name=name, configuration_hash=configuration_hash
        )
        other_hash = manager.hash_payload({"other": i})
        name_state = manager.lookup_name_hash_json(
            directory, name=name, configuration_hash=other_hash
        )
        assert name_state.state == NAME_TAKEN
        with pytest.raises(FileExistsError, match="already exists with hash"):
            manager.verify_name_hash_json_unique(
                directory, name=name, configuration_hash=other_hash
            )
        other_name = name + "_other"
        hash_state = manager.lookup_name_hash_json(
            directory, name=other_name, configuration_hash=configuration_hash
        )
        assert hash_state.state == HASH_TAKEN
        with pytest.raises(FileExistsError, match="already saved as name"):
            manager.verify_name_hash_json_unique(
                directory, name=other_name, configuration_hash=configuration_hash
            )


def test_extra_cannot_repeat_name_or_hash(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    name = random_generating_process_name(rng)
    configuration_hash = manager.hash_payload({"k": 1})
    with pytest.raises(ValueError, match="extra must not repeat"):
        manager.write_name_hash_json_unique(
            tmp_path / "extra",
            name=name,
            configuration_hash=configuration_hash,
            extra={"name": name},
        )


def test_hash_stem_is_rejected_as_name() -> None:
    manager = _manager()
    digest = manager.hash_payload({"k": 1})
    with pytest.raises(ValueError, match="hash stem"):
        manager.assert_name(manager.stem(digest))
