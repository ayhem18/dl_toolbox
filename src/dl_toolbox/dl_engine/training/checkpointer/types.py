"""Checkpoint request, record, catalog-derived fit state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

KIND_EPOCH = "epoch"
KIND_RANK = "rank"
KIND_VALUE = "value"
KIND_LAST = "last"
FIT_CLAIMED = "claimed"
FIT_RUNNING = "running"
FIT_FINISHED = "finished"
CHECKPOINTS_DIRNAME = "checkpoints"
LIVE_DIRNAME = "live"
CATALOG_FILENAME = "catalog.json"
LOCK_DIRNAME = "catalog.lock"


class CheckpointLookupError(ValueError):
    """Selector does not resolve to a retained file."""


class CheckpointSaveError(ValueError):
    """Fit cannot write a checkpoint (missing monitor, bad trainer)."""


@dataclass(frozen=True)
class CheckpointPolicy:
    save_top_k: int
    save_last: bool


@dataclass(frozen=True)
class CheckpointRequest:
    kind: str
    epoch: int | None = None
    rank: int | None = None
    value: float | None = None

    @staticmethod
    def parse(selector: str) -> CheckpointRequest:
        text = parse_non_empty_str(selector, label="selector")
        if text == KIND_LAST:
            return CheckpointRequest(kind=KIND_LAST)
        if text.startswith("epoch="):
            return CheckpointRequest(kind=KIND_EPOCH, epoch=_parse_int_suffix(text, "epoch"))
        if text.startswith("rank="):
            rank = _parse_int_suffix(text, "rank")
            if rank < 1:
                raise CheckpointLookupError(f"rank must be >= 1, got {rank}")
            return CheckpointRequest(kind=KIND_RANK, rank=rank)
        if text.startswith("value="):
            return CheckpointRequest(kind=KIND_VALUE, value=_parse_float_suffix(text))
        raise CheckpointLookupError(
            f"selector must be 'last', 'rank=N', 'epoch=N', or 'value=S', got {text!r}"
        )


@dataclass(frozen=True)
class CheckpointRecord:
    epoch: int
    path: Path
    rank: int | None
    score: float
    kind: str
    checkpoint_id: str


@dataclass(frozen=True)
class FitRunState:
    status: str
    latest_epoch: int | None
    current_best_epoch: int | None
    max_epochs: int | None


def _parse_int_suffix(text: str, label: str) -> int:
    suffix = text.split("=", 1)[1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise CheckpointLookupError(f"{label} selector must be an int, got {text!r}") from exc


def _parse_float_suffix(text: str) -> float:
    suffix = text.split("=", 1)[1]
    try:
        return float(suffix)
    except ValueError as exc:
        raise CheckpointLookupError(f"value selector must be a float, got {text!r}") from exc
