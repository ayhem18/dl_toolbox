# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.policy import (
    ScoredCheckpoint,
    decide_keep,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import CheckpointPolicy


def _row(*, checkpoint_id: str, score: float, epoch: int) -> ScoredCheckpoint:
    return ScoredCheckpoint(id=checkpoint_id, score=score, epoch=epoch)


def test_empty_catalog_keeps_first_when_top_k_positive() -> None:
    kept = decide_keep(
        current_files=[],
        score=0.5,
        policy=CheckpointPolicy(save_top_k=1, save_last=False),
        mode="min",
    )
    assert kept is True


def test_worse_min_score_is_skipped_when_top_k_full() -> None:
    current = [_row(checkpoint_id="aaa", score=0.1, epoch=0)]
    kept = decide_keep(
        current_files=current,
        score=0.9,
        policy=CheckpointPolicy(save_top_k=1, save_last=False),
        mode="min",
    )
    assert kept is False


def test_better_min_score_is_kept() -> None:
    current = [_row(checkpoint_id="aaa", score=0.5, epoch=0)]
    kept = decide_keep(
        current_files=current,
        score=0.1,
        policy=CheckpointPolicy(save_top_k=1, save_last=False),
        mode="min",
    )
    assert kept is True


def test_save_last_keeps_even_when_worse_than_top_k() -> None:
    current = [_row(checkpoint_id="aaa", score=0.1, epoch=0)]
    kept = decide_keep(
        current_files=current,
        score=0.9,
        policy=CheckpointPolicy(save_top_k=1, save_last=True),
        mode="min",
    )
    assert kept is True


def test_save_top_k_all_always_keeps() -> None:
    current = [_row(checkpoint_id="aaa", score=0.1, epoch=0)]
    kept = decide_keep(
        current_files=current,
        score=9.0,
        policy=CheckpointPolicy(save_top_k=-1, save_last=False),
        mode="min",
    )
    assert kept is True


def test_max_mode_keeps_higher_score() -> None:
    current = [_row(checkpoint_id="aaa", score=0.5, epoch=0)]
    better = decide_keep(
        current_files=current,
        score=0.9,
        policy=CheckpointPolicy(save_top_k=1, save_last=False),
        mode="max",
    )
    worse = decide_keep(
        current_files=current,
        score=0.1,
        policy=CheckpointPolicy(save_top_k=1, save_last=False),
        mode="max",
    )
    assert better is True
    assert worse is False


def test_save_top_k_zero_without_last_skips() -> None:
    kept = decide_keep(
        current_files=[],
        score=0.1,
        policy=CheckpointPolicy(save_top_k=0, save_last=False),
        mode="min",
    )
    assert kept is False
