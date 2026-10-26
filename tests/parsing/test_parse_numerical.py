from __future__ import annotations

import random

import pytest

from dl_toolbox.base.parsing.parse_numerical import (
    parse_bool,
    parse_half_open_unit_interval,
    parse_int,
    parse_non_negative_int,
    parse_non_negative_number,
    parse_number,
    parse_percentile,
    parse_positive_int,
    parse_positive_number,
)

N_TRIALS = 40


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def test_parse_bool_accepts_bools_and_rejects_ints(rng: random.Random) -> None:
    assert parse_bool(True, label="flag") is True
    assert parse_bool(False, label="flag") is False
    for _ in range(N_TRIALS):
        with pytest.raises(ValueError, match="flag"):
            parse_bool(rng.choice((0, 1, "true", None)), label="flag")


def test_parse_int_rejects_bool_and_float(rng: random.Random) -> None:
    for _ in range(N_TRIALS):
        value = rng.randint(-10_000, 10_000)
        assert parse_int(value, label="floor") == value
    with pytest.raises(ValueError, match="floor"):
        parse_int(True, label="floor")
    with pytest.raises(ValueError, match="floor"):
        parse_int(1.5, label="floor")


def test_non_negative_and_positive(rng: random.Random) -> None:
    assert parse_non_negative_int(0, label="offset") == 0
    for _ in range(N_TRIALS):
        value = rng.randint(1, 10_000)
        assert parse_non_negative_int(value, label="offset") == value
        assert parse_positive_int(value, label="gate") == value
        with pytest.raises(ValueError, match="offset"):
            parse_non_negative_int(-value, label="offset")
    with pytest.raises(ValueError, match="gate"):
        parse_positive_int(0, label="gate")


def test_parse_percentile() -> None:
    assert parse_percentile(1, label="low") == 1
    assert parse_percentile(99.5, label="high") == 99.5
    with pytest.raises(ValueError, match="low"):
        parse_percentile(0, label="low")
    with pytest.raises(ValueError, match="high"):
        parse_percentile(100, label="high")
    with pytest.raises(ValueError, match="ratio"):
        parse_number(True, label="ratio")


def test_positive_and_unit_interval_numbers(rng: random.Random) -> None:
    assert parse_non_negative_number(0, label="wd") == 0.0
    assert parse_positive_number(1e-7, label="lr") == 1e-7
    assert parse_half_open_unit_interval(0.05, label="smooth") == 0.05
    with pytest.raises(ValueError, match="lr"):
        parse_positive_number(0, label="lr")
    with pytest.raises(ValueError, match="smooth"):
        parse_half_open_unit_interval(1, label="smooth")
    for _ in range(N_TRIALS):
        value = rng.random() * 0.99
        assert parse_half_open_unit_interval(value, label="smooth") == float(value)
        with pytest.raises(ValueError, match="smooth"):
            parse_half_open_unit_interval(1.0 + rng.random(), label="smooth")
