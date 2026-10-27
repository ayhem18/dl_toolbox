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
from dl_toolbox.base.input_output.json_io import read_json, write_json

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


def test_release_name_drops_pair_and_leaves_other_hash(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = tmp_path / f"release_{i}"
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload({"i": i, "noise": rng.randint(0, 1000)})
        manager.write_name_hash_json_unique(
            directory, name=name, configuration_hash=configuration_hash
        )
        other_name = name + "_kept"
        other_hash = manager.hash_payload({"kept": i})
        manager.write_name_hash_json_unique(
            directory, name=other_name, configuration_hash=other_hash
        )
        assert manager.release_name(directory, name=name)["hash"] == configuration_hash
        assert (
            manager.lookup_name_hash_json(
                directory, name=name, configuration_hash=configuration_hash
            ).state
            == BOTH_FREE
        )
        assert (
            manager.lookup_name_hash_json(
                directory, name=other_name, configuration_hash=other_hash
            ).state
            == BOTH_MATCH
        )
        replacement = manager.hash_payload({"replacement": i})
        manager.write_name_hash_json_unique(
            directory, name=name, configuration_hash=replacement
        )
        assert (
            manager.lookup_name_hash_json(
                directory, name=name, configuration_hash=replacement
            ).state
            == BOTH_MATCH
        )
        assert manager.release_name(directory, name=name + "_missing") is None


def test_lookup_cross_pair_is_hash_taken(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    directory = tmp_path / "cross"
    name_a = random_generating_process_name(rng)
    name_b = name_a + "_b"
    hash_a = manager.hash_payload({"who": "a", "n": rng.randint(0, 1000)})
    hash_b = manager.hash_payload({"who": "b", "n": rng.randint(0, 1000)})
    manager.write_name_hash_json_unique(
        directory, name=name_a, configuration_hash=hash_a
    )
    manager.write_name_hash_json_unique(
        directory, name=name_b, configuration_hash=hash_b
    )
    crossed = manager.lookup_name_hash_json(
        directory, name=name_b, configuration_hash=hash_a
    )
    assert crossed.state == HASH_TAKEN
    with pytest.raises(FileExistsError, match="already saved as name"):
        manager.verify_name_hash_json_unique(
            directory, name=name_b, configuration_hash=hash_a
        )


def test_release_name_refuses_disagreeing_pair(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = _manager()
    directory = tmp_path / "disagree"
    name = random_generating_process_name(rng)
    configuration_hash = manager.hash_payload({"k": 1})
    other_hash = manager.hash_payload({"k": 2})
    name_path, hash_path = manager.name_hash_paths(
        directory, name=name, configuration_hash=configuration_hash
    )
    write_json(name_path, {"name": name, "hash": configuration_hash})
    write_json(hash_path, {"name": name, "hash": other_hash})
    with pytest.raises(ValueError, match="disagree"):
        manager.release_name(directory, name=name)
