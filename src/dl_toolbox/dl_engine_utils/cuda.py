"""CUDA runtime helpers used by inspect / extract loops."""

from __future__ import annotations

import torch


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    return "out of memory" in str(exc).lower()


def resolve_torch_device(name: str | None) -> torch.device:
    if name is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)
