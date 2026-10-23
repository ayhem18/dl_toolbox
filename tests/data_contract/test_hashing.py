from __future__ import annotations

import random

import pytest

from dl_toolbox.data_contract.hashing import (
    DEFAULT_ARTIFACT_PREFIX_LEN,
    artifact_stem,
    hash_payload,
    is_full_sha256,
    is_hashed_artifact_stem,
)

from _support import N_TRIALS, random_json_object, shuffled_mapping


def test_same_object_same_hash_under_key_shuffles(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        payload = random_json_object(rng)
        left = hash_payload(payload)
        right = hash_payload(shuffled_mapping(rng, payload))
        assert left == right
        assert is_full_sha256(left)
        stem = artifact_stem(left)
        assert left.startswith(stem)
        assert len(stem) == DEFAULT_ARTIFACT_PREFIX_LEN
        assert is_hashed_artifact_stem(stem)
        assert is_hashed_artifact_stem(left)


def test_different_objects_different_hashes(rng: random.Random) -> None:
    hashes: set[str] = set()
    for i in range(N_TRIALS):
        digest = hash_payload({"trial": i, "noise": rng.randint(0, 1_000_000)})
        assert digest not in hashes
        hashes.add(digest)


def test_truncated_identity_is_rejected(rng: random.Random) -> None:
    full = hash_payload({"k": 1})
    for _ in range(N_TRIALS):
        cut = rng.randint(8, 32)
        with pytest.raises(ValueError, match="64-char"):
            artifact_stem(full[:cut])


def test_ordinary_names_are_not_hash_stems(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        name = "run_" + "".join(rng.choice("abcdefxyz012") for _ in range(8))
        assert not is_hashed_artifact_stem(name)
