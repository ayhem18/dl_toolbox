"""Mixed-precision policy for one range-test session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from dl_toolbox.dl_engine.lr_finder.runtime import LrFinderRuntime
from dl_toolbox.dl_engine.lr_finder.step import LrFinderStep

_FP16 = frozenset({16, "16", "16-mixed"})
_BF16 = frozenset({"bf16", "bf16-mixed"})


@dataclass(frozen=True)
class AmpPolicy:
    dtype: torch.dtype | None
    use_scaler: bool

    @classmethod
    def from_runtime(
        cls, runtime: LrFinderRuntime, device: torch.device
    ) -> AmpPolicy:
        if device.type != "cuda":
            return cls(dtype=None, use_scaler=False)
        if runtime.precision in _FP16:
            return cls(dtype=torch.float16, use_scaler=True)
        if runtime.precision in _BF16:
            return cls(dtype=torch.bfloat16, use_scaler=False)
        return cls(dtype=None, use_scaler=False)


class AmpSession:
    def __init__(self, policy: AmpPolicy, device: torch.device) -> None:
        self._policy = policy
        self._device = device
        self._scaler = (
            torch.amp.GradScaler("cuda") if policy.use_scaler else None
        )

    @classmethod
    def from_runtime(
        cls, runtime: LrFinderRuntime, device: torch.device
    ) -> AmpSession:
        return cls(AmpPolicy.from_runtime(runtime, device), device)

    def compute_loss(
        self,
        model: nn.Module,
        prepared: Mapping[str, Any],
        step: LrFinderStep,
    ) -> Tensor:
        dtype = self._policy.dtype
        if dtype is None:
            output = step.forward_fn(model, prepared)
            return step.loss_fn(output, prepared)
        with torch.amp.autocast(device_type=self._device.type, dtype=dtype):
            output = step.forward_fn(model, prepared)
            return step.loss_fn(output, prepared)

    def optimizer_step(self, loss: Tensor, optimizer: Optimizer) -> None:
        scaler = self._scaler
        if scaler is None:
            loss.backward()
            optimizer.step()
            return
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
