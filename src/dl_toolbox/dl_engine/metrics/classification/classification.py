from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from dl_toolbox.dl_engine.metrics.contracts import ACCURACY_KEYS, KEY_OUTPUT, KEY_TARGET, LOSS_KEYS
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject, concat_dim0


def mean_items(values: Tensor) -> Tensor:
    return values.mean()


def micro_from_counts(counts: Tensor) -> Tensor:
    totals = counts.reshape(-1, 2).sum(dim=0)
    return totals[0] / totals[1].clamp_min(1.0)


def binary_accuracy_counts(
    output: Tensor,
    target: Tensor,
    *,
    threshold: float,
) -> Tensor:
    logits, labels = _align_binary_shapes(output, target)
    predicted = (torch.sigmoid(logits) > threshold).to(dtype=torch.float32)
    equal = (predicted == labels).to(dtype=torch.float32)
    correct = equal.reshape(equal.shape[0], -1).sum(dim=1)
    total = torch.full(
        (equal.shape[0],),
        float(equal[0].numel()),
        device=equal.device,
        dtype=correct.dtype,
    )
    return torch.stack((correct, total), dim=1)


def multiclass_accuracy_counts(output: Tensor, target: Tensor) -> Tensor:
    labels = _multiclass_labels(output, target)
    preds = output.argmax(dim=1)
    equal = (preds == labels).to(dtype=torch.float32)
    correct = equal.reshape(equal.shape[0], -1).sum(dim=1)
    total = torch.full(
        (equal.shape[0],),
        float(equal[0].numel()),
        device=equal.device,
        dtype=correct.dtype,
    )
    return torch.stack((correct, total), dim=1)


def multiclass_cross_entropy_per_item(output: Tensor, target: Tensor) -> Tensor:
    labels = _multiclass_labels(output, target)
    per_elem = F.cross_entropy(output, labels, reduction="none")
    return _mean_non_batch(per_elem)


def binary_bce_per_item(output: Tensor, target: Tensor) -> Tensor:
    logits, labels = _align_binary_shapes(output, target)
    per_elem = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return _mean_non_batch(per_elem)


def binary_classification_accuracy_metric(*, threshold: float = 0.5) -> MetricObject:
    score = float(threshold)

    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        return binary_accuracy_counts(
            tensors[KEY_OUTPUT],
            tensors[KEY_TARGET],
            threshold=score,
        )

    return MetricObject(
        name="accuracy",
        required_keys=ACCURACY_KEYS,
        metric_function=measure,
        combiner=concat_dim0,
        reducers={"acc": micro_from_counts},
    )


def multiclass_top1_accuracy_metric() -> MetricObject:
    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        return multiclass_accuracy_counts(tensors[KEY_OUTPUT], tensors[KEY_TARGET])

    return MetricObject(
        name="accuracy",
        required_keys=ACCURACY_KEYS,
        metric_function=measure,
        combiner=concat_dim0,
        reducers={"acc": micro_from_counts},
    )


def multiclass_cross_entropy_loss_metric() -> MetricObject:
    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        return multiclass_cross_entropy_per_item(tensors[KEY_OUTPUT], tensors[KEY_TARGET])

    return MetricObject(
        name="loss",
        required_keys=LOSS_KEYS,
        metric_function=measure,
        combiner=concat_dim0,
        reducers={"mean": mean_items},
    )


def binary_bce_loss_metric() -> MetricObject:
    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        return binary_bce_per_item(tensors[KEY_OUTPUT], tensors[KEY_TARGET])

    return MetricObject(
        name="loss",
        required_keys=LOSS_KEYS,
        metric_function=measure,
        combiner=concat_dim0,
        reducers={"mean": mean_items},
    )


def _align_binary_shapes(output: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    logits = output if output.ndim > 1 else output.unsqueeze(1)
    labels = target if target.ndim > 1 else target.unsqueeze(1)
    labels = labels.to(device=logits.device, dtype=torch.float32)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"batch size mismatch: output {tuple(logits.shape)} vs target {tuple(labels.shape)}"
        )
    return logits, labels


def _multiclass_labels(output: Tensor, target: Tensor) -> Tensor:
    labels = target.to(device=output.device).long()
    if labels.ndim == output.ndim and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.shape[0] != output.shape[0]:
        raise ValueError(
            f"batch size mismatch: output {tuple(output.shape)} vs target {tuple(labels.shape)}"
        )
    return labels


def _mean_non_batch(values: Tensor) -> Tensor:
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1).mean(dim=1)
