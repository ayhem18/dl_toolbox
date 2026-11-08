from __future__ import annotations

import random

import pytest

from dl_toolbox.base.hashing.identity import artifact_stem
from dl_toolbox.data_contract.names import assert_generating_process_name

from tests.data_contract.dc_support import (
    N_TRIALS,
    hash_manager,
    random_generating_process_name,
    random_json_object,
)


def test_valid_names_round_trip(rng: random.Random) -> None:
    manager = hash_manager()
    for _ in range(N_TRIALS):
        name = random_generating_process_name(rng)
        padded = f"  {name}  "
        assert assert_generating_process_name(padded, manager=manager) == name
        assert assert_generating_process_name(f"{name}.json", manager=manager) == name


def test_paths_and_empty_are_rejected(rng: random.Random) -> None:
    manager = hash_manager()
    with pytest.raises(ValueError, match="empty"):
        assert_generating_process_name("   ", manager=manager)
    for _ in range(N_TRIALS):
        name = random_generating_process_name(rng)
        with pytest.raises(ValueError, match="basename"):
            assert_generating_process_name(f"{name}/nested", manager=manager)
        with pytest.raises(ValueError, match="basename"):
            assert_generating_process_name("..", manager=manager)


def test_hash_stems_are_rejected(rng: random.Random) -> None:
    manager = hash_manager()
    for _ in range(N_TRIALS):
        digest = manager.hash_payload(random_json_object(rng))
        with pytest.raises(ValueError, match="prefix"):
            assert_generating_process_name(
                artifact_stem(digest, prefix_len=manager.prefix_len),
                manager=manager,
            )
        with pytest.raises(ValueError, match="prefix"):
            assert_generating_process_name(digest, manager=manager)
