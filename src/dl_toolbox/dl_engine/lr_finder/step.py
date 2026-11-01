"""One range-test step: prepare tensors, forward, loss. No Lightning module."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True)
class LrFinderStep:
    prepare_tensors_fn: Callable[[Any], Mapping[str, Any]]
    forward_fn: Callable[[nn.Module, Mapping[str, Any]], Tensor]
    loss_fn: Callable[[Tensor, Mapping[str, Any]], Tensor]
