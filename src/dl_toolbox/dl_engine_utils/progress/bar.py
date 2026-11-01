"""Known-length tqdm range. tqdm is a required dependency."""

from __future__ import annotations

from tqdm import tqdm


def range_progress(
    num_iter: int,
    *,
    enabled: bool,
    desc: str = "",
    unit: str = "it",
) -> tqdm:
    return tqdm(range(num_iter), desc=desc, unit=unit, disable=not enabled)
