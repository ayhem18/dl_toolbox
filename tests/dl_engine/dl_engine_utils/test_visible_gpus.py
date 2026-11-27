from __future__ import annotations

import os

import pytest

from dl_toolbox.dl_engine.dl_engine_utils.visible_gpus import (
    VisibleGpus,
    bind_visible_gpus,
    candidate_indices,
    free_gpu_indices,
    gib_to_mib,
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
    assert candidate_indices(env_value=None) is None
    assert candidate_indices(env_value="") is None
    assert candidate_indices(env_value="0, 2,3") == (0, 2, 3)
    with pytest.raises(ValueError, match="duplicate"):
        candidate_indices(env_value="1,1")


def test_bind_visible_gpus_writes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VisibleGpus.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.dl_engine_utils.visible_gpus.query_gpu_free_mib",
        lambda: ((4, 12000), (7, 30000), (1, 25000)),
    )
    selected = bind_visible_gpus(count=2)
    assert selected == (7, 1)
    assert os.environ[VisibleGpus.CUDA_VISIBLE_DEVICES] == "7,1"


def test_gib_to_mib() -> None:
    assert gib_to_mib(min_free_gib=20) == 20 * VisibleGpus.MIB_PER_GIB


def test_gib_to_mib_rejects_zero() -> None:
    with pytest.raises(ValueError, match="min_free_gib"):
        gib_to_mib(min_free_gib=0)


def test_free_threshold_filters_gpus() -> None:
    gpus = ((0, 19 * 1024), (1, 20 * 1024), (2, 40 * 1024))
    assert free_gpu_indices(
        gpus=gpus,
        candidates=None,
        min_free_mib=gib_to_mib(min_free_gib=20),
    ) == (2, 1)
