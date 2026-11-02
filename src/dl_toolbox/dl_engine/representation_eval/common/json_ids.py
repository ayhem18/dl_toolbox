"""Make FeatureBank ids JSON-serializable."""

from __future__ import annotations


def jsonable_id(sample_id: object) -> object:
    if isinstance(sample_id, tuple):
        n = len(sample_id)
        out: list[object] = [None] * n
        for i in range(n):
            out[i] = sample_id[i]
        return out
    return sample_id
