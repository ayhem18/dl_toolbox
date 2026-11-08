"""JSON-object key checks and scalar membership."""

from collections.abc import Callable, Mapping, Sequence


def require_key(payload: Mapping, key: str, *, label: str) -> object:
    if key not in payload:
        raise ValueError(
            f"{label}: expected key {key!r}; got keys {sorted(payload)}"
        )
    return payload[key]


def require_exact_keys(
    payload: Mapping, keys: Sequence[str], *, label: str) -> None:
    expected = tuple(keys)
    actual = set(payload)
    if actual != set(expected):
        raise ValueError(f"{label} keys must be {expected}, got {sorted(actual)}")


def expected_vs_actual(where: str, *, expected: str, actual: object) -> ValueError:
    return ValueError(
        f"{where}: expected {expected}; got {type(actual).__name__}={actual!r}"
    )


def one_of_message(
    *,
    value: object,
    allowed: Sequence[object],
    label: str,
    allowed_text: str) -> str:
    del allowed
    return f"{label} must be {allowed_text}, got {value!r}"


def parse_one_of(
    value: object,
    *,
    allowed: Sequence[object],
    label: str,
    error_type: type[BaseException],
    error_message: Callable[..., str]) -> object:
    if isinstance(allowed, (str, bytes)) or not isinstance(allowed, Sequence):
        raise TypeError(
            f"allowed must be a non-string Sequence, got {type(allowed)!r}"
        )
    if len(allowed) == 0:
        raise ValueError("allowed must be non-empty")
    if not isinstance(label, str) or not label:
        raise ValueError(f"label must be a non-empty str, got {label!r}")
    if not isinstance(error_type, type) or not issubclass(error_type, BaseException):
        raise TypeError(
            f"error_type must be an exception class, got {error_type!r}"
        )
    if not callable(error_message):
        raise TypeError(
            f"error_message must be callable, got {type(error_message)!r}"
        )
    if value in allowed:
        return value
    allowed_text = _format_allowed(allowed)
    message = error_message(
        value=value,
        allowed=allowed,
        label=label,
        allowed_text=allowed_text,
    )
    if not isinstance(message, str):
        raise TypeError(
            f"error_message must return str, got {type(message)!r}"
        )
    raise error_type(message)


def _format_allowed(allowed: Sequence[object]) -> str:
    items = [repr(item) for item in allowed]
    n = len(items)
    if n == 1:
        return items[0]
    if n == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"
