from __future__ import annotations

import pytest
import torch

from dl_toolbox.dl_engine.metrics.accumulator import MetricAccumulator
from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject, concat_dim0, metric_log_name


def _identity_metric() -> MetricObject:
    return MetricObject(
        name="vec",
        required_keys=frozenset({KEY_OUTPUT, KEY_TARGET}),
        metric_function=lambda tensors: tensors[KEY_OUTPUT].reshape(-1),
        combiner=concat_dim0,
        reducers={"mean": lambda values: values.mean()},
    )


def test_accumulator_concatenates_batches_then_reduces() -> None:
    metric = _identity_metric()
    acc = MetricAccumulator([(metric, ("mean",), ("mean",))], split="val")
    first = acc.update(
        {KEY_OUTPUT: torch.tensor([1.0, 3.0]), KEY_TARGET: torch.zeros(2)}
    )
    second = acc.update(
        {KEY_OUTPUT: torch.tensor([8.0]), KEY_TARGET: torch.zeros(1)}
    )
    assert first[metric_log_name("vec", "val", "batch", "mean")] == pytest.approx(2.0)
    assert second[metric_log_name("vec", "val", "batch", "mean")] == pytest.approx(8.0)
    epoch = acc.epoch_scalars()[metric_log_name("vec", "val", "epoch", "mean")]
    assert epoch == pytest.approx(4.0)
    mean_of_batch_means = (2.0 + 8.0) / 2.0
    assert epoch != pytest.approx(mean_of_batch_means)


def test_unknown_reducer_is_an_error() -> None:
    metric = _identity_metric()
    with pytest.raises(ValueError, match="reducer"):
        MetricAccumulator([(metric, (), ("missing",))], split="train")
