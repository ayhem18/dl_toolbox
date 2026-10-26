from __future__ import annotations

import random
import string

import pytest

from dl_toolbox.base.parsing.parse_iterable import (
    parse_mapping,
    parse_sequence,
    parse_unique_str_tuple,
)

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def test_parse_mapping() -> None:
    payload = {"a": 1}
    assert parse_mapping(payload, label="args") is payload
    with pytest.raises(TypeError, match="args"):
        parse_mapping(["a"], label="args")


def test_parse_sequence_rejects_str() -> None:
    assert list(parse_sequence(["A", "B"], label="names")) == ["A", "B"]
    with pytest.raises(ValueError, match="names"):
        parse_sequence("AB", label="names")
    with pytest.raises(ValueError, match="names"):
        parse_sequence(b"AB", label="names")


def test_parse_unique_str_tuple(rng: random.Random) -> None:
    assert parse_unique_str_tuple([" A ", "B"], label="trusted_classes") == ("A", "B")
    with pytest.raises(ValueError, match="trusted_classes"):
        parse_unique_str_tuple([], label="trusted_classes")
    with pytest.raises(ValueError, match="duplicate"):
        parse_unique_str_tuple(["A", "A"], label="trusted_classes")
    with pytest.raises(ValueError, match="entries"):
        parse_unique_str_tuple(["A", "  "], label="trusted_classes")
    for _ in range(N_TRIALS):
        n = rng.randint(1, 8)
        names: list[str] = [""] * n
        seen: set[str] = set()
        for i in range(n):
            name = "n" + "".join(
                rng.choice(string.ascii_lowercase) for _ in range(4)
            )
            while name in seen:
                name = "n" + "".join(
                    rng.choice(string.ascii_lowercase) for _ in range(4)
                )
            seen.add(name)
            names[i] = name
        assert parse_unique_str_tuple(names, label="trusted_classes") == tuple(names)
