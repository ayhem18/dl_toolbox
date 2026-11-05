"""Keep / prune ranking. No disk."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CatalogLastPointer,
    CheckpointPolicy,
)

_CANDIDATE_ID = "__policy_candidate__"
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class ScoredCheckpoint(BaseModel):
    """One checkpoint as the policy sees it: id + score (+ epoch for last)."""

    model_config = _STRICT

    id: str
    score: float
    epoch: int


class RankedCheckpoint(BaseModel):
    model_config = _STRICT

    id: str
    score: float
    epoch: int
    rank: int = Field(ge=1)


class PolicyApplyResult(BaseModel):
    model_config = _STRICT

    ranked: tuple[RankedCheckpoint, ...]
    keep_ids: frozenset[str]
    last: CatalogLastPointer | None
    ordered: tuple[RankedCheckpoint, ...]


def decide_keep(
    *,
    current_files: Sequence[ScoredCheckpoint],
    score: float,
    policy: CheckpointPolicy,
    mode: str,
) -> bool:
    """Whether a new score belongs in the keep set (top-k and/or last)."""
    n_current = len(current_files)
    files: list[ScoredCheckpoint] = []
    for i in range(n_current):
        files.append(current_files[i])
    files.append(
        ScoredCheckpoint(id=_CANDIDATE_ID, score=float(score), epoch=-1)
    )
    last_id = _CANDIDATE_ID if policy.save_last else None
    last_epoch = -1 if policy.save_last else None
    result = apply_policy(
        files=files,
        policy=policy,
        mode=mode,
        last_id=last_id,
        last_epoch=last_epoch,
    )
    return _CANDIDATE_ID in result.keep_ids


def apply_policy(
    *,
    files: Sequence[ScoredCheckpoint],
    policy: CheckpointPolicy,
    mode: str,
    last_id: str | None,
    last_epoch: int | None,
) -> PolicyApplyResult:
    """Rank ``files``. ``last_*`` is the just-saved checkpoint when save_last."""
    if mode not in ("min", "max"):
        raise ValueError(f"monitor.mode must be 'min' or 'max', got {mode!r}")
    n = len(files)
    ordered_src: list[ScoredCheckpoint] = []
    for i in range(n):
        ordered_src.append(files[i])
    reverse = mode == "max"
    ordered_src.sort(key=_score_of, reverse=reverse)
    ordered: list[RankedCheckpoint] = []
    for i in range(n):
        item = ordered_src[i]
        ordered.append(
            RankedCheckpoint(
                id=item.id,
                score=item.score,
                epoch=item.epoch,
                rank=i + 1,
            )
        )
    if policy.save_top_k == -1:
        keep_n = n
    else:
        keep_n = policy.save_top_k
    if keep_n < 0:
        ranked: list[RankedCheckpoint] = []
    else:
        ranked = ordered[:keep_n]
    keep_ids: set[str] = set()
    n_ranked = len(ranked)
    for i in range(n_ranked):
        keep_ids.add(ranked[i].id)
    last_row = None
    if policy.save_last:
        if last_id is None or last_epoch is None:
            raise ValueError("save_last requires last_id and last_epoch")
        last_row = CatalogLastPointer(epoch=last_epoch, id=last_id)
        keep_ids.add(str(last_id))
    return PolicyApplyResult(
        ranked=tuple(ranked),
        keep_ids=frozenset(keep_ids),
        last=last_row,
        ordered=tuple(ordered),
    )


def _score_of(item: ScoredCheckpoint) -> float:
    return item.score
