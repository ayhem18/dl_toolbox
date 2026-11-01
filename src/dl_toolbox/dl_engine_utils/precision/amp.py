"""Mixed-precision policy and one training-step AMP session."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

_FP16 = frozenset({16, "16", "16-mixed"})
_BF16 = frozenset({"bf16", "bf16-mixed"})


@dataclass(frozen=True)
class AmpPolicy:
    dtype: torch.dtype | None
    use_scaler: bool

    @classmethod
    def from_precision(
        cls, precision: int | str, device: torch.device
    ) -> AmpPolicy:
        if device.type != "cuda":
            return cls(dtype=None, use_scaler=False)
        if precision in _FP16:
            return cls(dtype=torch.float16, use_scaler=True)
        if precision in _BF16:
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
    def from_precision(
        cls, precision: int | str, device: torch.device
    ) -> AmpSession:
        return cls(AmpPolicy.from_precision(precision, device), device)

    def autocast(self) -> AbstractContextManager[None]:
        dtype = self._policy.dtype
        if dtype is None:
            return nullcontext()
        return torch.amp.autocast(device_type=self._device.type, dtype=dtype)

    def optimizer_step(self, loss: Tensor, optimizer: Optimizer) -> None:
        scaler = self._scaler
        if scaler is None:
            loss.backward()
            optimizer.step()
            return
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
