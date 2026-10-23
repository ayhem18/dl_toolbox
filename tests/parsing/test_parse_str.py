from __future__ import annotations

import random
import string

import pytest

from dl_toolbox.parsing.parse_str import parse_non_empty_str

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def test_strips_non_empty(rng: random.Random) -> None:
    assert parse_non_empty_str("  pore  ", label="recipe") == "pore"
    for _ in range(N_TRIALS):
        core = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 12)))
        padded = (" " * rng.randint(0, 3)) + core + (" " * rng.randint(0, 3))
        assert parse_non_empty_str(padded, label="recipe") == core


def test_empty_and_non_str_fail(rng: random.Random) -> None:
    with pytest.raises(ValueError, match="recipe"):
        parse_non_empty_str("   ", label="recipe")
    with pytest.raises(ValueError, match="recipe"):
        parse_non_empty_str(None, label="recipe")
    for _ in range(N_TRIALS):
        with pytest.raises(ValueError, match="recipe"):
            parse_non_empty_str(" " * rng.randint(0, 5), label="recipe")
