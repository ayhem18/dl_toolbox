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
from dl_toolbox.base.input_output.json_io import JsonTextForm, read, write

from tests.base.base_support import N_TRIALS, random_generating_process_name, random_json_object


def _manager() -> HashManager:
    return HashManager(
        prefix_len=12,
        human_readable_dirname="human_readable",
        hashes_dirname="hashes",
    )


def _index(path: Path) -> Directory:
    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.CREATE)


def test_write_and_lookup_unique_pair(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = _index(tmp_path / f"records_{i}")
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(random_json_object(rng))
        lookup = manager.lookup(
            directory, name=name, configuration_hash=configuration_hash
        )
        assert lookup.state == HashManager.BOTH_FREE
        assert lookup.document is None
        name_path, hash_path = manager.claim(
            directory,
            name=name,
            configuration_hash=configuration_hash,
            extra={"note": i},
        )
        document = read(file=File.from_absolute(path=name_path))
        assert document == read(file=File.from_absolute(path=hash_path))
        assert document["name"] == name
        assert document["hash"] == configuration_hash
        assert document["note"] == i
        matched = manager.lookup(
            directory, name=name, configuration_hash=configuration_hash
        )
        assert matched.state == HashManager.BOTH_MATCH
        assert matched.document == document
        with pytest.raises(FileExistsError, match="already claimed"):
            manager.claim(
                directory,
                name=name,
                configuration_hash=configuration_hash,
                extra={},
            )


def test_name_taken_and_hash_taken(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = _index(tmp_path / f"taken_{i}")
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(
            {"i": i, "noise": rng.randint(0, 1000)}
        )
        manager.claim(
            directory,
            name=name,
            configuration_hash=configuration_hash,
            extra={},
        )
        other_hash = manager.hash_payload({"other": i})
        name_state = manager.lookup(
            directory, name=name, configuration_hash=other_hash
        )
        assert name_state.state == HashManager.NAME_TAKEN
        assert name_state.document is not None
        assert name_state.document["hash"] == configuration_hash
        with pytest.raises(FileExistsError, match="already exists with hash"):
            manager.claim(
                directory,
                name=name,
                configuration_hash=other_hash,
                extra={},
            )
        other_name = name + "_other"
        hash_state = manager.lookup(
            directory, name=other_name, configuration_hash=configuration_hash
        )
        assert hash_state.state == HashManager.HASH_TAKEN
        assert hash_state.document is not None
        assert hash_state.document["name"] == name
        with pytest.raises(FileExistsError, match="already saved as name"):
            manager.claim(
                directory,
                name=other_name,
                configuration_hash=configuration_hash,
                extra={},
            )


def test_extra_cannot_repeat_name_or_hash(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    directory = _index(tmp_path / "extra")
    name = random_generating_process_name(rng)
    configuration_hash = manager.hash_payload({"k": 1})
    with pytest.raises(ValueError, match="extra must not repeat"):
        manager.claim(
            directory,
            name=name,
            configuration_hash=configuration_hash,
            extra={"name": name},
        )


def test_hash_stem_is_rejected_as_name() -> None:
    manager = _manager()
    digest = manager.hash_payload({"k": 1})
    stem = artifact_stem(digest, prefix_len=manager.prefix_len)
    with pytest.raises(ValueError, match="hash stem"):
        manager.assert_name(stem)


def test_release_name_drops_pair_and_leaves_other_hash(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = _manager()
    for i in range(N_TRIALS):
        directory = _index(tmp_path / f"release_{i}")
        name = random_generating_process_name(rng)
        configuration_hash = manager.hash_payload(
            {"i": i, "noise": rng.randint(0, 1000)}
        )
        manager.claim(
            directory,
            name=name,
            configuration_hash=configuration_hash,
            extra={},
        )
        other_name = name + "_kept"
        other_hash = manager.hash_payload({"kept": i})
        manager.claim(
            directory,
            name=other_name,
            configuration_hash=other_hash,
            extra={},
        )
        released = manager.release_name(directory, name=name)
        assert released is not None
        assert released["hash"] == configuration_hash
        assert (
            manager.lookup(
                directory, name=name, configuration_hash=configuration_hash
            ).state
            == HashManager.BOTH_FREE
        )
        assert (
            manager.lookup(
                directory, name=other_name, configuration_hash=other_hash
            ).state
            == HashManager.BOTH_MATCH
        )
        replacement = manager.hash_payload({"replacement": i})
        manager.claim(
            directory,
            name=name,
            configuration_hash=replacement,
            extra={},
        )
        assert (
            manager.lookup(
                directory, name=name, configuration_hash=replacement
            ).state
            == HashManager.BOTH_MATCH
        )
        assert manager.release_name(directory, name=name + "_missing") is None


def test_lookup_cross_pair_is_hash_taken(tmp_path: Path, rng: random.Random) -> None:
    manager = _manager()
    directory = _index(tmp_path / "cross")
    name_a = random_generating_process_name(rng)
    name_b = name_a + "_b"
    hash_a = manager.hash_payload({"who": "a", "n": rng.randint(0, 1000)})
    hash_b = manager.hash_payload({"who": "b", "n": rng.randint(0, 1000)})
    manager.claim(directory, name=name_a, configuration_hash=hash_a, extra={})
    manager.claim(directory, name=name_b, configuration_hash=hash_b, extra={})
    crossed = manager.lookup(directory, name=name_b, configuration_hash=hash_a)
    assert crossed.state == HashManager.HASH_TAKEN
    assert crossed.document is not None
    assert crossed.document["name"] == name_a
    with pytest.raises(FileExistsError, match="already saved as name"):
        manager.claim(
            directory, name=name_b, configuration_hash=hash_a, extra={}
        )


def test_release_name_refuses_disagreeing_pair(
    tmp_path: Path, rng: random.Random
) -> None:
    manager = _manager()
    directory = _index(tmp_path / "disagree")
    name = random_generating_process_name(rng)
    configuration_hash = manager.hash_payload({"k": 1})
    other_hash = manager.hash_payload({"k": 2})
    manager.claim(
        directory,
        name=name,
        configuration_hash=configuration_hash,
        extra={},
    )
    hashes = Directory.from_under(
        root=directory,
        relative_path=manager.hashes_dirname,
        policy=DirectoryParsePolicy.MUST_EXIST,
    )
    write(
        directory=hashes,
        name=f"{artifact_stem(configuration_hash, prefix_len=manager.prefix_len)}.json",
        payload={"name": name, "hash": other_hash},
        form=JsonTextForm.DOCUMENT,
    )
    with pytest.raises(ValueError, match="disagree"):
        manager.release_name(directory, name=name)


def test_layout_rejects_identical_dirnames_and_hash_stems() -> None:
    with pytest.raises(ValueError, match="distinct"):
        HashManager(
            prefix_len=12,
            human_readable_dirname="same",
            hashes_dirname="same",
        )
    with pytest.raises(ValueError, match="prefix_len"):
        HashManager(
            prefix_len=4,
            human_readable_dirname="human_readable",
            hashes_dirname="hashes",
        )
    digest = HashManager(
        prefix_len=12,
        human_readable_dirname="human_readable",
        hashes_dirname="hashes",
    ).hash_payload({"k": 1})
    stem = artifact_stem(digest, prefix_len=12)
    with pytest.raises(ValueError, match="hash stem"):
        HashManager(
            prefix_len=12,
            human_readable_dirname=stem,
            hashes_dirname="hashes",
        )


def test_lookup_does_not_create_index_folders(tmp_path: Path) -> None:
    manager = _manager()
    directory = _index(tmp_path / "empty")
    digest = manager.hash_payload({"k": 1})
    lookup = manager.lookup(
        directory, name="run_free", configuration_hash=digest
    )
    assert lookup.state == HashManager.BOTH_FREE
    assert not (directory.path / manager.human_readable_dirname).exists()
    assert not (directory.path / manager.hashes_dirname).exists()


def test_prefix_collision_is_error_not_occupancy(tmp_path: Path) -> None:
    manager = _manager()
    directory = _index(tmp_path / "prefix")
    digest = manager.hash_payload({"k": 1})
    other = "0" * 64
    if other == digest:
        other = "1" * 64
    hashes = Directory.from_under(
        root=directory,
        relative_path=manager.hashes_dirname,
        policy=DirectoryParsePolicy.CREATE,
    )
    write(
        directory=hashes,
        name=f"{artifact_stem(digest, prefix_len=manager.prefix_len)}.json",
        payload={"name": "other_run", "hash": other},
        form=JsonTextForm.DOCUMENT,
    )
    with pytest.raises(ValueError, match="already used by"):
        manager.lookup(
            directory, name="run_free", configuration_hash=digest
        )
