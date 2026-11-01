"""Accelerator and precision. Device resolution lives here, AMP does not."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dl_toolbox.dl_engine.lr_finder.errors import LrFinderError

_CPU = "cpu"
_GPU = "gpu"
_CUDA = "cuda"
_ACCELERATORS = frozenset({_CPU, _GPU, _CUDA})


@dataclass(frozen=True)
class LrFinderRuntime:
    accelerator: str
    precision: int | str


def validate_runtime(runtime: LrFinderRuntime) -> None:
    name = _accelerator_name(runtime.accelerator)
    if name not in _ACCELERATORS:
        raise LrFinderError(
            f"accelerator must be one of {sorted(_ACCELERATORS)}, "
            f"got {runtime.accelerator!r}"
        )


def resolve_device(runtime: LrFinderRuntime) -> torch.device:
    validate_runtime(runtime)
    name = _accelerator_name(runtime.accelerator)
    if name == _CPU:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise LrFinderError("accelerator is gpu but CUDA is not available")
    return torch.device("cuda")


def _accelerator_name(accelerator: str) -> str:
    return str(accelerator).strip().lower()
