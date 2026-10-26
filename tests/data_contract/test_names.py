from __future__ import annotations

import random

import pytest

from dl_toolbox.base.hashing.identity import artifact_stem, hash_payload
from dl_toolbox.data_contract.names import assert_generating_process_name

from _support import N_TRIALS, random_generating_process_name, random_json_object


def test_valid_names_round_trip(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        name = random_generating_process_name(rng)
        padded = f"  {name}  "
        assert assert_generating_process_name(padded) == name
        assert assert_generating_process_name(f"{name}.json") == name


def test_paths_and_empty_are_rejected(rng: random.Random) -> None:
    with pytest.raises(ValueError, match="empty"):
        assert_generating_process_name("   ")
    for _ in range(N_TRIALS):
        name = random_generating_process_name(rng)
        with pytest.raises(ValueError, match="basename"):
            assert_generating_process_name(f"{name}/nested")
        with pytest.raises(ValueError, match="basename"):
            assert_generating_process_name("..")


def test_hash_stems_are_rejected(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        digest = hash_payload(random_json_object(rng))
        with pytest.raises(ValueError, match="hash stem"):
            assert_generating_process_name(artifact_stem(digest))
        with pytest.raises(ValueError, match="hash stem"):
            assert_generating_process_name(digest)
