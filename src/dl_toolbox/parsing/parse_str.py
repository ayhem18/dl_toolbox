"""Non-empty strings from JSON."""


def parse_non_empty_str(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty str")
    return raw.strip()
