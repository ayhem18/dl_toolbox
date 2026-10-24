from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor

from dl_toolbox.dl_engine.metrics.accumulator import MetricAccumulator
from dl_toolbox.dl_engine.metrics.classification.classification import (
    binary_bce_loss_metric,
    binary_classification_accuracy_metric,
    multiclass_cross_entropy_loss_metric,
    multiclass_top1_accuracy_metric,
)
from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_BATCH, LEVEL_EPOCH, metric_log_name

N_TRIALS = 40
N_CLASSES = 5


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def _mapping(output: Tensor, target: Tensor) -> dict[str, Tensor]:
    return {KEY_OUTPUT: output, KEY_TARGET: target}


def _one_hot_logits(labels: Tensor, n_classes: int, *, correct: bool, scale: float) -> Tensor:
    logits = torch.zeros(labels.shape[0], n_classes)
    for i, label in enumerate(labels.tolist()):
        peak = int(label) if correct else (int(label) + 1) % n_classes
        logits[i, peak] = scale
    return logits


def test_ce_metric_function_keeps_item_axis(rng: random.Random) -> None:
    metric = multiclass_cross_entropy_loss_metric()
    for _ in range(N_TRIALS):
        batch_size = rng.randint(2, 16)
        labels = torch.tensor([rng.randrange(N_CLASSES) for _ in range(batch_size)])
        logits = torch.randn(batch_size, N_CLASSES)
        values = metric.metric_function(_mapping(logits, labels))
        assert tuple(values.shape) == (batch_size,)


def test_bce_metric_function_keeps_item_axis(rng: random.Random) -> None:
    metric = binary_bce_loss_metric()
    for _ in range(N_TRIALS):
        batch_size = rng.randint(2, 16)
        labels = torch.tensor([rng.choice([0.0, 1.0]) for _ in range(batch_size)])
        logits = torch.randn(batch_size, 1)
        values = metric.metric_function(_mapping(logits, labels))
        assert tuple(values.shape) == (batch_size,)


def test_accuracy_metric_function_is_per_item_counts(rng: random.Random) -> None:
    metric = multiclass_top1_accuracy_metric()
    for _ in range(N_TRIALS):
        batch_size = rng.randint(2, 16)
        labels = torch.tensor([rng.randrange(N_CLASSES) for _ in range(batch_size)])
        logits = torch.randn(batch_size, N_CLASSES)
        values = metric.metric_function(_mapping(logits, labels))
        assert tuple(values.shape) == (batch_size, 2)
        assert torch.all(values[:, 1] == 1.0)


def test_epoch_ce_is_item_mean_not_mean_of_batch_means() -> None:
    metric = multiclass_cross_entropy_loss_metric()
    acc = MetricAccumulator([(metric, ("mean",), ("mean",))], split="val")
    easy_labels = torch.zeros(8, dtype=torch.long)
    hard_labels = torch.zeros(3, dtype=torch.long)
    easy = _mapping(_one_hot_logits(easy_labels, N_CLASSES, correct=True, scale=8.0), easy_labels)
    hard = _mapping(_one_hot_logits(hard_labels, N_CLASSES, correct=False, scale=8.0), hard_labels)
    batch_easy = acc.update(easy)[metric_log_name("loss", "val", LEVEL_BATCH, "mean")]
    batch_hard = acc.update(hard)[metric_log_name("loss", "val", LEVEL_BATCH, "mean")]
    epoch = acc.epoch_scalars()[metric_log_name("loss", "val", LEVEL_EPOCH, "mean")]
    mean_of_means = (batch_easy + batch_hard) / 2.0
    item_mean = (8.0 * batch_easy + 3.0 * batch_hard) / 11.0
    assert epoch == pytest.approx(item_mean, rel=1e-5, abs=1e-6)
    assert abs(epoch - mean_of_means) > 0.5


def test_epoch_accuracy_is_micro_over_items() -> None:
    metric = multiclass_top1_accuracy_metric()
    acc = MetricAccumulator([(metric, ("acc",), ("acc",))], split="val")
    correct_labels = torch.arange(8) % N_CLASSES
    wrong_labels = torch.zeros(2, dtype=torch.long)
    acc.update(
        _mapping(
            _one_hot_logits(correct_labels, N_CLASSES, correct=True, scale=5.0),
            correct_labels,
        )
    )
    acc.update(
        _mapping(
            _one_hot_logits(wrong_labels, N_CLASSES, correct=False, scale=5.0),
            wrong_labels,
        )
    )
    epoch = acc.epoch_scalars()[metric_log_name("accuracy", "val", LEVEL_EPOCH, "acc")]
    assert epoch == pytest.approx(8.0 / 10.0)


def test_binary_accuracy_threshold_separates_scores() -> None:
    metric = binary_classification_accuracy_metric(threshold=0.5)
    logits = torch.tensor([[4.0], [-4.0], [0.2], [-0.2]])
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    acc = MetricAccumulator([(metric, (), ("acc",))], split="train")
    acc.update(_mapping(logits, labels))
    assert acc.epoch_scalars()[metric_log_name("accuracy", "train", LEVEL_EPOCH, "acc")] == pytest.approx(
        1.0
    )


def test_same_reducer_on_batch_and_combined_epoch(rng: random.Random) -> None:
    metric = multiclass_cross_entropy_loss_metric()
    for _ in range(N_TRIALS):
        size_a = rng.randint(2, 9)
        size_b = rng.randint(2, 9)
        labels_a = torch.tensor([rng.randrange(N_CLASSES) for _ in range(size_a)])
        labels_b = torch.tensor([rng.randrange(N_CLASSES) for _ in range(size_b)])
        logits_a = torch.randn(size_a, N_CLASSES)
        logits_b = torch.randn(size_b, N_CLASSES)
        acc = MetricAccumulator([(metric, ("mean",), ("mean",))], split="train")
        acc.update(_mapping(logits_a, labels_a))
        acc.update(_mapping(logits_b, labels_b))
        combined = metric.combiner(
            metric.metric_function(_mapping(logits_a, labels_a)),
            metric.metric_function(_mapping(logits_b, labels_b)),
        )
        expected = float(metric.reducers["mean"](combined).item())
        assert acc.epoch_scalars()[metric_log_name("loss", "train", LEVEL_EPOCH, "mean")] == pytest.approx(
            expected, rel=1e-5, abs=1e-6
        )
