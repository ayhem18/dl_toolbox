"""Rebuild a DataLoader for frozen-feature extract: no shuffle, no drop_last."""

from __future__ import annotations

from torch.utils.data import DataLoader


def loader_for_extract(
    source: DataLoader, *, batch_size: int | None = None
) -> DataLoader:
    size = int(source.batch_size or 1) if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    kwargs: dict = {
        "batch_size": size,
        "shuffle": False,
        "num_workers": source.num_workers,
        "pin_memory": bool(source.pin_memory),
        "drop_last": False,
        "collate_fn": source.collate_fn,
    }
    if source.num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(source.dataset, **kwargs)
