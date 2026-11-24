"""Linear learning-rate scale."""

from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.dl_engine_utils.optimization.linear_lr_scale import (
    effective_batch_size,
    scaled_learning_rate,
)


def test_linear_lr_scaling_uses_global_batch() -> None:
    global_batch = effective_batch_size(
        batch_size=1, devices=4, accumulate_grad_batches=1
    )
    assert global_batch == 4
    assert scaled_learning_rate(
        base_learning_rate=0.00015,
        global_batch_size=global_batch,
        batch_reference=256,
    ) == pytest.approx(0.00015 * 4 / 256)
    assert scaled_learning_rate(
        base_learning_rate=0.00015,
        global_batch_size=256,
        batch_reference=256,
    ) == pytest.approx(0.00015)
