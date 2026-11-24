"""Scale a base learning rate by the global batch size."""

from __future__ import annotations


def effective_batch_size(
    *, batch_size: int, devices: int, accumulate_grad_batches: int
) -> int:
    """Global batch: per-device batch times devices times accumulation steps."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if devices < 1:
        raise ValueError(f"devices must be >= 1, got {devices}")
    if accumulate_grad_batches < 1:
        raise ValueError(
            f"accumulate_grad_batches must be >= 1, got {accumulate_grad_batches}"
        )
    return batch_size * devices * accumulate_grad_batches


def scaled_learning_rate(
    *,
    base_learning_rate: float,
    global_batch_size: int,
    batch_reference: int,
) -> float:
    """Linear rule: ``base_learning_rate * global_batch_size / batch_reference``."""
    if base_learning_rate <= 0.0:
        raise ValueError(
            f"base_learning_rate must be > 0, got {base_learning_rate}"
        )
    if global_batch_size < 1:
        raise ValueError(f"global_batch_size must be >= 1, got {global_batch_size}")
    if batch_reference < 1:
        raise ValueError(f"batch_reference must be >= 1, got {batch_reference}")
    return base_learning_rate * global_batch_size / float(batch_reference)
