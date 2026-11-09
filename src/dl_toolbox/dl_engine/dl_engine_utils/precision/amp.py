"""Mixed-precision policy and one training-step AMP session."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch
from torch import Tensor
from torch.optim import Optimizer


class AmpPolicy:
    FP16 = frozenset({16, "16", "16-mixed"})
    BF16 = frozenset({"bf16", "bf16-mixed"})

    def __init__(self, *, dtype: torch.dtype | None, use_scaler: bool) -> None:
        if not isinstance(use_scaler, bool):
            raise TypeError(f"use_scaler must be bool, got {type(use_scaler)!r}")
        self.dtype = dtype
        self.use_scaler = use_scaler

    @classmethod
    def from_precision(
        cls, precision: int | str, device: torch.device
    ) -> AmpPolicy:
        if not isinstance(device, torch.device):
            raise TypeError(f"device must be torch.device, got {type(device)!r}")
        if device.type != "cuda":
            return cls(dtype=None, use_scaler=False)
        if precision in cls.FP16:
            return cls(dtype=torch.float16, use_scaler=True)
        if precision in cls.BF16:
            return cls(dtype=torch.bfloat16, use_scaler=False)
        return cls(dtype=None, use_scaler=False)


class AmpSession:
    def __init__(self, policy: AmpPolicy, device: torch.device) -> None:
        if not isinstance(policy, AmpPolicy):
            raise TypeError(f"policy must be AmpPolicy, got {type(policy)!r}")
        if not isinstance(device, torch.device):
            raise TypeError(f"device must be torch.device, got {type(device)!r}")
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
        if not isinstance(loss, Tensor):
            raise TypeError(f"loss must be Tensor, got {type(loss)!r}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
        scaler = self._scaler
        if scaler is None:
            loss.backward()
            optimizer.step()
            return
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
