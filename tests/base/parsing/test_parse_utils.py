from __future__ import annotations

import random
import string

import pytest

from dl_toolbox.base.parsing.parse_utils import (
    expected_vs_actual,
    require_exact_keys,
    require_key,
)

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def test_require_key_returns_the_value(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        key = "k_" + "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
        value = rng.randint(-1000, 1000)
        payload = {key: value, "other": 1}
        assert require_key(payload, key, label="row") == value


def test_missing_key_names_expected_and_actual() -> None:
    with pytest.raises(ValueError) as ctx:
        require_key({"b": 1}, "a", label="row")
    message = str(ctx.value)
    assert "a" in message
    assert "b" in message


def test_expected_vs_actual_message() -> None:
    error = expected_vs_actual("kind", expected="'matched'", actual="pair")
    assert "kind" in str(error)
    assert "matched" in str(error)
    assert "pair" in str(error)


def test_exact_keys_pass() -> None:
    require_exact_keys({"a": 1, "b": 2}, ("a", "b"), label="args")


def test_extra_or_missing_fails() -> None:
    with pytest.raises(ValueError, match="args"):
        require_exact_keys({"a": 1, "c": 3}, ("a", "b"), label="args")
    with pytest.raises(ValueError, match="args"):
        require_exact_keys({"a": 1}, ("a", "b"), label="args")
