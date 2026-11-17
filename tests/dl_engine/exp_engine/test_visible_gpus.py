from __future__ import annotations

import os

import pytest

from dl_toolbox.dl_engine.exp_engine.visible_gpus import (
    CUDA_VISIBLE_DEVICES,
    bind_visible_gpus,
    candidate_indices,
    select_freest_gpus,
)


def test_select_freest_gpus_orders_by_free_memory() -> None:
    selected = select_freest_gpus(
        count=2,
        gpus=((0, 1000), (1, 20000), (2, 9000), (3, 15000)),
        candidates=None,
    )
    assert selected == (1, 3)


def test_select_freest_gpus_restricts_to_cuda_visible_devices() -> None:
    selected = select_freest_gpus(
        count=1,
        gpus=((0, 20000), (1, 40000), (2, 10000)),
        candidates=(0, 2),
    )
    assert selected == (0,)


def test_select_freest_gpus_errors_when_pool_is_too_small() -> None:
    with pytest.raises(ValueError, match="need 2 GPU"):
        select_freest_gpus(
            count=2,
            gpus=((0, 1000),),
            candidates=None,
        )


def test_candidate_indices_parses_env() -> None:
    assert candidate_indices(None) is None
    assert candidate_indices("") is None
    assert candidate_indices("0, 2,3") == (0, 2, 3)
    with pytest.raises(ValueError, match="duplicate"):
        candidate_indices("1,1")


def test_bind_visible_gpus_writes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.exp_engine.visible_gpus.query_gpu_free_mib",
        lambda: ((4, 12000), (7, 30000), (1, 25000)),
    )
    selected = bind_visible_gpus(count=2)
    assert selected == (7, 1)
    assert os.environ[CUDA_VISIBLE_DEVICES] == "7,1"
