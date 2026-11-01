from __future__ import annotations

import torch

from dl_toolbox.dl_engine.lr_finder.precision import AmpPolicy
from dl_toolbox.dl_engine.lr_finder.runtime import LrFinderRuntime


def test_cpu_disables_amp() -> None:
    runtime = LrFinderRuntime(accelerator="cpu", precision=16)
    policy = AmpPolicy.from_runtime(runtime, torch.device("cpu"))
    assert policy.dtype is None
    assert policy.use_scaler is False


def test_cuda_fp16_uses_scaler() -> None:
    runtime = LrFinderRuntime(accelerator="gpu", precision=16)
    policy = AmpPolicy.from_runtime(runtime, torch.device("cuda"))
    assert policy.dtype == torch.float16
    assert policy.use_scaler is True


def test_cuda_bf16_has_no_scaler() -> None:
    runtime = LrFinderRuntime(accelerator="gpu", precision="bf16")
    policy = AmpPolicy.from_runtime(runtime, torch.device("cuda"))
    assert policy.dtype == torch.bfloat16
    assert policy.use_scaler is False
