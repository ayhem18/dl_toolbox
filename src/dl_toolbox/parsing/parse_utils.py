"""Shared JSON-object checks."""

from collections.abc import Mapping, Sequence


def require_key(payload: Mapping, key: str, *, label: str) -> object:
    if key not in payload:
        raise ValueError(
            f"{label}: expected key {key!r}; got keys {sorted(payload)}"
        )
    return payload[key]


def require_exact_keys(
    payload: Mapping, keys: Sequence[str], *, label: str
) -> None:
    expected = tuple(keys)
    actual = set(payload)
    if actual != set(expected):
        raise ValueError(f"{label} keys must be {expected}, got {sorted(actual)}")


def expected_vs_actual(where: str, *, expected: str, actual: object) -> ValueError:
    return ValueError(
        f"{where}: expected {expected}; got {type(actual).__name__}={actual!r}"
    )
