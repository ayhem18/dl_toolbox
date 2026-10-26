"""Bools, ints, and floats from JSON. Rejects bool where a number is required."""


def parse_bool(raw: object, *, label: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{label} must be a bool")
    return raw


def parse_int(raw: object, *, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{label} must be an int, got {raw!r}")
    return raw


def parse_non_negative_int(raw: object, *, label: str) -> int:
    value = parse_int(raw, label=label)
    if value < 0:
        raise ValueError(f"{label} must be a non-negative int, got {raw!r}")
    return value


def parse_positive_int(raw: object, *, label: str) -> int:
    value = parse_int(raw, label=label)
    if value <= 0:
        raise ValueError(f"{label} must be a positive int, got {raw!r}")
    return value


def parse_number(raw: object, *, label: str) -> int | float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} must be a number, got {raw!r}")
    return raw


def parse_percentile(raw: object, *, label: str) -> int | float:
    value = parse_number(raw, label=label)
    if not 0 < float(value) < 100:
        raise ValueError(f"{label} must be between 0 and 100, got {raw!r}")
    return value


def parse_non_negative_number(raw: object, *, label: str) -> float:
    value = float(parse_number(raw, label=label))
    if value < 0.0:
        raise ValueError(f"{label} must be >= 0, got {raw!r}")
    return value


def parse_positive_number(raw: object, *, label: str) -> float:
    value = float(parse_number(raw, label=label))
    if value <= 0.0:
        raise ValueError(f"{label} must be > 0, got {raw!r}")
    return value


def parse_half_open_unit_interval(raw: object, *, label: str) -> float:
    value = float(parse_number(raw, label=label))
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{label} must be in [0, 1), got {raw!r}")
    return value
