from __future__ import annotations

import math

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dl_toolbox.dl_engine.dl_engine_utils.optimization.warmup_cosine import (
    make_warmup_cosine_scheduler,
    warmup_cosine_lambda,
)

MAX_EPOCHS = 200
WARMUP = 10
BASE_LR = 1.5e-4
MIN_LR = 0.0


def test_warmup_cosine_200_epochs_10_warmup() -> None:
    assert warmup_cosine_lambda(
        0,
        warmup_epochs=WARMUP,
        max_epochs=MAX_EPOCHS,
        min_lr=MIN_LR,
        base_lr=BASE_LR,
    ) == 0.0
    assert warmup_cosine_lambda(
        WARMUP,
        warmup_epochs=WARMUP,
        max_epochs=MAX_EPOCHS,
        min_lr=MIN_LR,
        base_lr=BASE_LR,
    ) == pytest.approx(1.0)
    last = warmup_cosine_lambda(
        MAX_EPOCHS - 1,
        warmup_epochs=WARMUP,
        max_epochs=MAX_EPOCHS,
        min_lr=MIN_LR,
        base_lr=BASE_LR,
    )
    progress = float((MAX_EPOCHS - 1) - WARMUP) / float(MAX_EPOCHS - WARMUP)
    expected = 0.5 * (1.0 + math.cos(math.pi * progress))
    assert last == pytest.approx(expected)
    assert last < 0.01


def test_lambda_scheduler_steps_match_warmup_then_peak() -> None:
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = AdamW([param], lr=BASE_LR)
    factory = make_warmup_cosine_scheduler(
        warmup_epochs=WARMUP,
        max_epochs=MAX_EPOCHS,
        min_lr=MIN_LR,
        base_lr=BASE_LR,
    )
    scheduler = factory(optimizer)
    assert isinstance(scheduler, LambdaLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    for _ in range(WARMUP):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR)
