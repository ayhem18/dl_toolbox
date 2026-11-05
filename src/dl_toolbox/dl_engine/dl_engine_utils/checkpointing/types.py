"""Checkpoint directory document, selectors, and fit state."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dl_toolbox.base.hashing.identity import is_full_sha256
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
DDP_JOIN_FILENAME = ".ddp_join.ckpt"

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class CheckpointLookupError(ValueError):
    """Selector does not resolve to a retained file."""


class CheckpointSaveError(ValueError):
    """Fit cannot write a checkpoint (missing file after write)."""


class CheckpointPolicy(BaseModel):
    model_config = _STRICT

    save_top_k: int = Field(ge=-1)
    save_last: bool


class CheckpointMonitor(BaseModel):
    model_config = _STRICT

    key: str
    mode: Literal["min", "max"]

    @field_validator("key")
    @classmethod
    def _key_non_empty(cls, value: str) -> str:
        return parse_non_empty_str(value, label="monitor.key")


class CatalogLastPointer(BaseModel):
    model_config = _STRICT

    epoch: int
    id: str


class CatalogLiveRow(BaseModel):
    """One retained ``live/{id}.ckpt`` as named in ``catalog.json``."""

    model_config = _STRICT

    epoch: int
    id: str
    path: str
    score: float
    sha256: str
    bytes: int = Field(ge=0)
    rank: int | None = None

    @field_validator("sha256")
    @classmethod
    def _full_digest(cls, value: str) -> str:
        if not is_full_sha256(value):
            raise ValueError("sha256 must be a 64-char lowercase digest")
        return value

    @field_validator("rank")
    @classmethod
    def _rank_positive(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1:
            raise ValueError(f"rank must be >= 1, got {value}")
        return value


class CatalogHistoryEntry(BaseModel):
    model_config = _STRICT

    id: str
    score: float
    rank_then: int
    sha256: str
    retained: bool

    @field_validator("sha256")
    @classmethod
    def _full_digest(cls, value: str) -> str:
        if not is_full_sha256(value):
            raise ValueError("sha256 must be a 64-char lowercase digest")
        return value


class CatalogFit(BaseModel):
    model_config = _STRICT

    status: Literal["running", "finished"]
    latest_epoch: int | None
    max_epochs: int = Field(ge=1)


class CatalogCurrent(BaseModel):
    model_config = _STRICT

    files: list[CatalogLiveRow]
    last: CatalogLastPointer | None


class CatalogDocument(BaseModel):
    """``catalog.json``: the named state of the Checkpoint Directory."""

    model_config = _STRICT

    monitor: CheckpointMonitor
    policy: CheckpointPolicy
    fit: CatalogFit
    current: CatalogCurrent
    history: dict[str, CatalogHistoryEntry]


class CheckpointRequest(BaseModel):
    model_config = _STRICT

    kind: Literal["last", "epoch", "rank", "value"]
    epoch: int | None = None
    rank: int | None = None
    value: float | None = None

    @classmethod
    def parse(cls, selector: str) -> Self:
        text = parse_non_empty_str(selector, label="selector")
        try:
            if text == KIND_LAST:
                return cls(kind=KIND_LAST)
            if text.startswith("epoch="):
                return cls(kind=KIND_EPOCH, epoch=int(text.split("=", 1)[1]))
            if text.startswith("rank="):
                rank = int(text.split("=", 1)[1])
                if rank < 1:
                    raise CheckpointLookupError(f"rank must be >= 1, got {rank}")
                return cls(kind=KIND_RANK, rank=rank)
            if text.startswith("value="):
                return cls(kind=KIND_VALUE, value=float(text.split("=", 1)[1]))
        except (TypeError, ValueError) as exc:
            if isinstance(exc, CheckpointLookupError):
                raise
            raise CheckpointLookupError(
                "selector must be 'last', 'rank=N', 'epoch=N', or 'value=S', "
                f"got {text!r}"
            ) from exc
        except ValidationError as exc:
            raise CheckpointLookupError(
                "selector must be 'last', 'rank=N', 'epoch=N', or 'value=S', "
                f"got {text!r}"
            ) from exc
        raise CheckpointLookupError(
            "selector must be 'last', 'rank=N', 'epoch=N', or 'value=S', "
            f"got {text!r}"
        )


class CheckpointRecord(BaseModel):
    model_config = _STRICT

    epoch: int
    path: Path
    rank: int | None
    score: float
    kind: str
    checkpoint_id: str


class FitRunState(BaseModel):
    model_config = _STRICT

    status: str
    latest_epoch: int | None
    current_best_epoch: int | None
    max_epochs: int | None
