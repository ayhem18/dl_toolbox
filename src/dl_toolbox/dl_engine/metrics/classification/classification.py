from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from dl_toolbox.dl_engine.metrics.contracts import ACCURACY_KEYS, KEY_OUTPUT, KEY_TARGET, LOSS_KEYS
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject


def stack_sum_combiner(previous: Tensor, value: Tensor) -> Tensor:
    return previous + value


def _ratio_from_counts(counts: Tensor) -> Tensor:
    correct, total = counts[0], counts[1].clamp_min(1.0)
    return correct / total


def binary_accuracy_counts(
    output: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    if target.dim() == 1:
        target = target.unsqueeze(1)
    target = target.to(device=output.device, dtype=torch.float32)
    predicted = (torch.sigmoid(output) > threshold).float()
    correct = (predicted == target).float().sum()
    total = torch.tensor(float(target.numel()), device=output.device, dtype=correct.dtype)
    return torch.stack([correct, total])


def multiclass_accuracy_counts(output: Tensor, target: Tensor) -> Tensor:
    labels = target.to(device=output.device).long().view(-1)
    preds = output.argmax(dim=-1)
    correct = (preds == labels).float().sum()
    total = torch.tensor(float(labels.numel()), device=output.device, dtype=correct.dtype)
    return torch.stack([correct, total])


def multiclass_cross_entropy_loss_scalar(output: Tensor, target: Tensor) -> Tensor:
    labels = target.to(device=output.device).long().view(-1)
    return F.cross_entropy(output, labels)


def binary_bce_loss_scalar(output: Tensor, target: Tensor) -> Tensor:
    if target.dim() == 1:
        target = target.unsqueeze(1)
    target = target.to(device=output.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(output, target)


def binary_classification_accuracy_metric(
    *,
    threshold: float = 0.5,
) -> MetricObject:
    return MetricObject(
        name="accuracy",
        required_keys=ACCURACY_KEYS,
        metric_function=lambda tensors: binary_accuracy_counts(
            tensors[KEY_OUTPUT],
            tensors[KEY_TARGET],
            threshold=threshold,
        ),
        combiner=stack_sum_combiner,
        reducers={"acc": _ratio_from_counts},
    )


def multiclass_top1_accuracy_metric() -> MetricObject:
    return MetricObject(
        name="accuracy",
        required_keys=ACCURACY_KEYS,
        metric_function=lambda tensors: multiclass_accuracy_counts(
            tensors[KEY_OUTPUT],
            tensors[KEY_TARGET],
        ),
        combiner=stack_sum_combiner,
        reducers={"acc": _ratio_from_counts},
    )


def multiclass_cross_entropy_loss_metric() -> MetricObject:
    def measure(tensors: dict[str, Tensor]) -> Tensor:
        loss = multiclass_cross_entropy_loss_scalar(
            tensors[KEY_OUTPUT],
            tensors[KEY_TARGET],
        ).reshape(())
        count = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)
        return torch.stack([loss, count])

    def mean_value(values: Tensor) -> Tensor:
        return values[0] / values[1].clamp_min(1.0)

    return MetricObject(
        name="loss",
        required_keys=LOSS_KEYS,
        metric_function=measure,
        combiner=stack_sum_combiner,
        reducers={"mean": mean_value},
    )


def binary_bce_loss_metric() -> MetricObject:
    def measure(tensors: dict[str, Tensor]) -> Tensor:
        loss = binary_bce_loss_scalar(tensors[KEY_OUTPUT], tensors[KEY_TARGET]).reshape(())
        count = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)
        return torch.stack([loss, count])

    def mean_value(values: Tensor) -> Tensor:
        return values[0] / values[1].clamp_min(1.0)

    return MetricObject(
        name="loss",
        required_keys=LOSS_KEYS,
        metric_function=measure,
        combiner=stack_sum_combiner,
        reducers={"mean": mean_value},
    )
