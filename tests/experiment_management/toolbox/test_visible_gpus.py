from __future__ import annotations

import os

import pytest

from dl_toolbox.experiment_management.toolbox.trainer.parser import TrainerConfigParser
from dl_toolbox.experiment_management.toolbox.trainer.visible_gpus import (
    CUDA_VISIBLE_DEVICES,
    MIN_FREE_MIB,
    bind_visible_gpus,
    candidate_indices,
    free_gpu_indices,
    select_freest_gpus,
)


def _gpu_payload(*, devices: int) -> dict:
    return {
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": devices,
        "precision": 32,
        "enable_progress_bar": False,
    }


def test_selects_top_n_by_free_memory() -> None:
    selected = select_freest_gpus(
        count=2,
        gpus=((0, 1000), (1, 20000), (2, 9000), (3, 15000)),
        candidates=None,
    )
    assert selected == (1, 3)


def test_free_gpu_indices_are_sorted_by_free_memory() -> None:
    indices = free_gpu_indices(
        gpus=((0, 1000), (1, 20000), (2, MIN_FREE_MIB), (3, 15000)),
        candidates=None,
    )
    assert indices == (1, 3, 2)


def test_skips_gpus_below_min_free() -> None:
    with pytest.raises(ValueError, match="need 2 GPU"):
        select_freest_gpus(
            count=2,
            gpus=((0, MIN_FREE_MIB - 1), (1, MIN_FREE_MIB)),
            candidates=None,
        )


def test_restricts_to_existing_cuda_visible_devices() -> None:
    selected = select_freest_gpus(
        count=1,
        gpus=((0, 20000), (1, 40000), (2, 10000)),
        candidates=(0, 2),
    )
    assert selected == (0,)


def test_candidate_indices_parses_env() -> None:
    assert candidate_indices(None) is None
    assert candidate_indices("") is None
    assert candidate_indices("0, 2,3") == (0, 2, 3)
    with pytest.raises(ValueError, match="duplicate"):
        candidate_indices("1,1")


def test_parser_sets_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr(
        "dl_toolbox.experiment_management.toolbox.trainer.visible_gpus.query_gpu_free_mib",
        lambda: ((0, 9000), (1, 22000), (2, 500), (3, 18000)),
    )
    parsed = TrainerConfigParser(_gpu_payload(devices=2))
    assert parsed.trainer_config.devices == 2
    assert os.environ[CUDA_VISIBLE_DEVICES] == "1,3"


def test_parser_raises_when_not_enough_free_gpus(monkeypatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr(
        "dl_toolbox.experiment_management.toolbox.trainer.visible_gpus.query_gpu_free_mib",
        lambda: ((0, 9000), (1, 1000)),
    )
    with pytest.raises(ValueError, match="need 4 GPU"):
        TrainerConfigParser(_gpu_payload(devices=4))


def test_cpu_parser_does_not_bind_gpus(monkeypatch) -> None:
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "keep-me")

    def _boom() -> tuple[tuple[int, int], ...]:
        raise AssertionError("cpu trainer must not query GPUs")

    monkeypatch.setattr(
        "dl_toolbox.experiment_management.toolbox.trainer.visible_gpus.query_gpu_free_mib",
        _boom,
    )
    parsed = TrainerConfigParser(
        {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
            "enable_progress_bar": False,
        }
    )
    assert parsed.trainer_config.accelerator == "cpu"
    assert os.environ[CUDA_VISIBLE_DEVICES] == "keep-me"


def test_bind_visible_gpus_writes_env(monkeypatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr(
        "dl_toolbox.experiment_management.toolbox.trainer.visible_gpus.query_gpu_free_mib",
        lambda: ((4, 12000), (7, 30000), (1, 25000)),
    )
    selected = bind_visible_gpus(count=2)
    assert selected == (7, 1)
    assert os.environ[CUDA_VISIBLE_DEVICES] == "7,1"
