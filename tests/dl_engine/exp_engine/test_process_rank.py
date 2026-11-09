from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.exp_engine.process_rank import ProcessRank, is_ddp_worker


@pytest.fixture(autouse=True)
def _clear_rank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        ProcessRank.RANK_ENV,
        ProcessRank.GLOBAL_RANK_ENV,
        ProcessRank.NODE_RANK_ENV,
        ProcessRank.GROUP_RANK_ENV,
        ProcessRank.LOCAL_RANK_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_named_rank_env_constants() -> None:
    assert ProcessRank.RANK_ENV == "RANK"
    assert ProcessRank.GLOBAL_RANK_ENV == "GLOBAL_RANK"
    assert ProcessRank.NODE_RANK_ENV == "NODE_RANK"
    assert ProcessRank.GROUP_RANK_ENV == "GROUP_RANK"
    assert ProcessRank.LOCAL_RANK_ENV == "LOCAL_RANK"


def test_empty_environment_is_not_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        ProcessRank.RANK_ENV,
        ProcessRank.GLOBAL_RANK_ENV,
        ProcessRank.NODE_RANK_ENV,
        ProcessRank.GROUP_RANK_ENV,
        ProcessRank.LOCAL_RANK_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    assert is_ddp_worker() is False
    assert ProcessRank.is_ddp_worker() is False


@pytest.mark.parametrize(
    "name",
    [
        ProcessRank.RANK_ENV,
        ProcessRank.GLOBAL_RANK_ENV,
        ProcessRank.NODE_RANK_ENV,
        ProcessRank.GROUP_RANK_ENV,
        ProcessRank.LOCAL_RANK_ENV,
    ],
)
def test_positive_rank_is_a_worker(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "1")
    assert is_ddp_worker() is True


def test_rank_zero_is_not_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ProcessRank.RANK_ENV, "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert is_ddp_worker() is False


def test_junk_and_empty_are_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ProcessRank.RANK_ENV, "gpu")
    assert is_ddp_worker() is False
    monkeypatch.setenv(ProcessRank.RANK_ENV, "  ")
    assert is_ddp_worker() is False


def test_any_one_worker_signal_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ProcessRank.RANK_ENV, "0")
    monkeypatch.setenv(ProcessRank.LOCAL_RANK_ENV, "1")
    assert is_ddp_worker() is True
