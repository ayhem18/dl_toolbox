"""JSON objects and lists. Rejects str and bytes as lists."""

from collections.abc import Sequence

from dl_toolbox.parsing.parse_str import parse_non_empty_str


def parse_mapping(raw: object, *, label: str) -> dict:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a dict")
    return raw


def parse_sequence(raw: object, *, label: str) -> Sequence:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{label} must be a list")
    return raw


def parse_unique_str_tuple(raw: object, *, label: str) -> tuple[str, ...]:
    sequence = parse_sequence(raw, label=label)
    if not sequence:
        raise ValueError(f"{label} must not be empty")
    n = len(sequence)
    names: list[str] = [""] * n
    seen: dict[str, None] = {}
    for i in range(n):
        name = parse_non_empty_str(sequence[i], label=f"{label} entries")
        if name in seen:
            raise ValueError(f"duplicate {label} value {name!r}")
        seen[name] = None
        names[i] = name
    return tuple(names)
