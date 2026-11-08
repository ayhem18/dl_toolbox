from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.hashing.identity import artifact_stem
from dl_toolbox.base.input_output import json_io
from dl_toolbox.base.input_output.json_io import JsonTextForm

from tests.base.base_support import N_TRIALS, random_generating_process_name, random_json_object


def _manager() -> HashManager:
    return HashManager(
        prefix_len=12,
        human_readable_dir_name="human_readable",
        hash_dir_name="hashes",
        lock_timeout_seconds=2,
        lock_poll_seconds=0.05,
    )


def _index(manager: HashManager, path: Path) -> Directory:
    directory = Directory.from_absolute(
        path=path, policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    return manager.create_identity_index(directory)


def test_write_and_lookup_unique_pair(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = _index(manager, tmp_path / f"records_{i}")
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(random_json_object(rng))
        lookup = manager.lookup(
            directory, human_readable_name=name, hash=configuration_hash
        )
        assert lookup.state == HashManager.BOTH_FREE
        assert lookup.document is None
        name_path, hash_path = manager.create_pair(
            directory,
            human_readable_name=name,
            hash=configuration_hash,
            extra={"note": i},
        )
        document = json_io.read_object(file=File.from_absolute(path=name_path))
        assert document == json_io.read_object(file=File.from_absolute(path=hash_path))
        assert document["name"] == name
        assert document["hash"] == configuration_hash
        assert document["note"] == i
        matched = manager.lookup(
            directory, human_readable_name=name, hash=configuration_hash
        )
        assert matched.state == HashManager.BOTH_MATCH
        assert matched.document == document
        with pytest.raises(FileExistsError, match="pair already exists"):
            manager.create_pair(
                directory,
                human_readable_name=name,
                hash=configuration_hash,
                extra={},
            )


def test_name_taken_and_hash_taken(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = _index(manager, tmp_path / f"taken_{i}")
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(
            {"i": i, "noise": rng.randint(0, 1000)}
        )
        manager.create_pair(
            directory,
            human_readable_name=name,
            hash=configuration_hash,
            extra={},
        )
        other_hash = manager.hash_payload({"other": i})
        name_state = manager.lookup(
            directory, human_readable_name=name, hash=other_hash
        )
        assert name_state.state == HashManager.NAME_TAKEN
        assert name_state.document is not None
        assert name_state.document["hash"] == configuration_hash
        with pytest.raises(FileExistsError, match="already exists with hash"):
            manager.create_pair(
                directory,
                human_readable_name=name,
                hash=other_hash,
                extra={},
            )
        other_name = name + "_other"
        hash_state = manager.lookup(
            directory, human_readable_name=other_name, hash=configuration_hash
        )
        assert hash_state.state == HashManager.HASH_TAKEN
        assert hash_state.document is not None
        assert hash_state.document["name"] == name
        with pytest.raises(FileExistsError, match="already saved as name"):
            manager.create_pair(
                directory,
                human_readable_name=other_name,
                hash=configuration_hash,
                extra={},
            )


def test_extra_cannot_repeat_name_or_hash(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    directory = _index(manager, tmp_path / "extra")
    name = random_generating_process_name(rng)
    configuration_hash = manager.hash_payload({"k": 1})
    with pytest.raises(ValueError, match="extra must not repeat"):
        manager.create_pair(
            directory,
            human_readable_name=name,
            hash=configuration_hash,
            extra={"name": name},
        )


def test_prefix_is_rejected_as_name() -> None:
    manager = _manager()
    digest = manager.hash_payload({"k": 1})
    prefix = artifact_stem(digest, prefix_len=manager.prefix_len)
    with pytest.raises(ValueError, match="prefix"):
        manager.validate_human_readable_name(prefix)


def test_drop_pair_drops_pair_and_leaves_other_hash(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = _index(manager, tmp_path / f"release_{i}")
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(
            {"i": i, "noise": rng.randint(0, 1000)}
        )
        manager.create_pair(
            directory,
            human_readable_name=name,
            hash=configuration_hash,
            extra={},
        )
        other_name = name + "_kept"
        other_hash = manager.hash_payload({"kept": i})
        manager.create_pair(
            directory,
            human_readable_name=other_name,
            hash=other_hash,
            extra={},
        )
        released = manager.drop_pair(directory, human_readable_name=name)
        assert released is not None
        assert released["hash"] == configuration_hash
        assert (
            manager.lookup(
                directory, human_readable_name=name, hash=configuration_hash
            ).state
            == HashManager.BOTH_FREE
        )
        assert (
            manager.lookup(
                directory, human_readable_name=other_name, hash=other_hash
            ).state
            == HashManager.BOTH_MATCH
        )
        replacement = manager.hash_payload({"replacement": i})
        manager.create_pair(
            directory,
            human_readable_name=name,
            hash=replacement,
            extra={},
        )
        assert (
            manager.lookup(
                directory, human_readable_name=name, hash=replacement
            ).state
            == HashManager.BOTH_MATCH
        )
        assert manager.drop_pair(directory, human_readable_name=name + "_missing") is None


def test_lookup_cross_pair_is_hash_taken(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    directory = _index(manager, tmp_path / "cross")
    name_a = random_generating_process_name(rng)
    name_b = name_a + "_b"
    hash_a = manager.hash_payload({"who": "a", "n": rng.randint(0, 1000)})
    hash_b = manager.hash_payload({"who": "b", "n": rng.randint(0, 1000)})
    manager.create_pair(directory, human_readable_name=name_a, hash=hash_a, extra={})
    manager.create_pair(directory, human_readable_name=name_b, hash=hash_b, extra={})
    crossed = manager.lookup(directory, human_readable_name=name_b, hash=hash_a)
    assert crossed.state == HashManager.HASH_TAKEN
    assert crossed.document is not None
    assert crossed.document["name"] == name_a
    with pytest.raises(FileExistsError, match="already saved as name"):
        manager.create_pair(
            directory, human_readable_name=name_b, hash=hash_a, extra={}
        )


def test_drop_pair_refuses_disagreeing_pair(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = _manager()
    directory = _index(manager, tmp_path / "disagree")
    name = random_generating_process_name(rng)
    configuration_hash = manager.hash_payload({"k": 1})
    other_hash = manager.hash_payload({"k": 2})
    manager.create_pair(
        directory,
        human_readable_name=name,
        hash=configuration_hash,
        extra={},
    )
    hashes = Directory.from_under(
        root=directory,
        relative_path=manager.hash_dir_name,
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    json_io.write_object(
        directory=hashes,
        name=f"{artifact_stem(configuration_hash, prefix_len=manager.prefix_len)}.json",
        payload={"name": name, "hash": other_hash},
        form=JsonTextForm.DOCUMENT,
    )
    with pytest.raises(ValueError, match="disagree"):
        manager.drop_pair(directory, human_readable_name=name)


def test_layout_rejects_identical_dirnames_and_prefixes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        HashManager(
            prefix_len=12,
            human_readable_dir_name="same",
            hash_dir_name="same",
            lock_timeout_seconds=2,
            lock_poll_seconds=0.05,
        )
    with pytest.raises(ValueError, match="prefix_len"):
        HashManager(
            prefix_len=4,
            human_readable_dir_name="human_readable",
            hash_dir_name="hashes",
            lock_timeout_seconds=2,
            lock_poll_seconds=0.05,
        )
    digest = HashManager(
        prefix_len=12,
        human_readable_dir_name="human_readable",
        hash_dir_name="hashes",
        lock_timeout_seconds=2,
        lock_poll_seconds=0.05,
    ).hash_payload({"k": 1})
    prefix = artifact_stem(digest, prefix_len=12)
    with pytest.raises(ValueError, match="prefix"):
        HashManager(
            prefix_len=12,
            human_readable_dir_name=prefix,
            hash_dir_name="hashes",
            lock_timeout_seconds=2,
            lock_poll_seconds=0.05,
        )


def test_lookup_on_empty_index_is_both_free(tmp_path: Path) -> None:
    manager = _manager()
    directory = _index(manager, tmp_path / "empty")
    digest = manager.hash_payload({"k": 1})
    lookup = manager.lookup(
        directory, human_readable_name="run_free", hash=digest
    )
    assert lookup.state == HashManager.BOTH_FREE
    assert list((directory.path / manager.human_readable_dir_name).iterdir()) == []
    assert list((directory.path / manager.hash_dir_name).iterdir()) == []


def test_lookup_requires_identity_index(tmp_path: Path) -> None:
    manager = _manager()
    directory = Directory.from_absolute(
        path=tmp_path / "not_index", policy=DirectoryParsePolicy.CREATE
    )
    digest = manager.hash_payload({"k": 1})
    with pytest.raises(ValueError, match="not an identity index"):
        manager.lookup(directory, human_readable_name="run_free", hash=digest)


def test_create_identity_index_is_idempotent_and_refuses_unmarked(
    tmp_path: Path,
) -> None:
    manager = _manager()
    missing = Directory.from_absolute(
        path=tmp_path / "idx", policy=DirectoryParsePolicy.ALLOW_MISSING
    )
    created = manager.create_identity_index(missing)
    assert (created.path / HashManager.DIR_INDEX_NAME).is_file()
    assert (created.path / manager.hash_dir_name).is_dir()
    assert (created.path / manager.human_readable_dir_name).is_dir()
    again = manager.create_identity_index(created)
    assert again.path == created.path
    unmarked = Directory.from_absolute(
        path=tmp_path / "random_folder", policy=DirectoryParsePolicy.CREATE
    )
    with pytest.raises(ValueError, match="not an identity index"):
        manager.create_identity_index(unmarked)


def test_prefix_collision_is_error_not_occupancy(tmp_path: Path) -> None:
    manager = _manager()
    directory = _index(manager, tmp_path / "prefix")
    digest = manager.hash_payload({"k": 1})
    other = "0" * 64
    if other == digest:
        other = "1" * 64
    hashes = Directory.from_under(
        root=directory,
        relative_path=manager.hash_dir_name,
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    json_io.write_object(
        directory=hashes,
        name=f"{artifact_stem(digest, prefix_len=manager.prefix_len)}.json",
        payload={"name": "other_run", "hash": other},
        form=JsonTextForm.DOCUMENT,
    )
    with pytest.raises(ValueError, match="already used by"):
        manager.lookup(
            directory, human_readable_name="run_free", hash=digest
        )


def test_lock_is_gone_after_lookup(tmp_path: Path) -> None:
    manager = _manager()
    directory = _index(manager, tmp_path / "lock")
    digest = manager.hash_payload({"k": 1})
    manager.lookup(directory, human_readable_name="run_free", hash=digest)
    assert not (directory.path / HashManager.IDENTITY_LOCK_NAME).exists()
