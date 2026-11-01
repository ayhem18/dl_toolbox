from __future__ import annotations

import torch
from torch import nn

from dl_toolbox.dl_engine_utils.precision import AmpPolicy, AmpSession


def test_cpu_disables_amp() -> None:
    policy = AmpPolicy.from_precision(16, torch.device("cpu"))
    assert policy.dtype is None
    assert policy.use_scaler is False


def test_cuda_fp16_uses_scaler() -> None:
    policy = AmpPolicy.from_precision(16, torch.device("cuda"))
    assert policy.dtype == torch.float16
    assert policy.use_scaler is True


def test_cuda_bf16_has_no_scaler() -> None:
    policy = AmpPolicy.from_precision("bf16", torch.device("cuda"))
    assert policy.dtype == torch.bfloat16
    assert policy.use_scaler is False


def test_cuda_fp32_disables_amp() -> None:
    policy = AmpPolicy.from_precision(32, torch.device("cuda"))
    assert policy.dtype is None
    assert policy.use_scaler is False


def test_cpu_session_steps_optimizer() -> None:
    device = torch.device("cpu")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = AmpSession.from_precision(32, device)
    before = model.weight.detach().clone()
    optimizer.zero_grad()
    with session.autocast():
        loss = model(torch.ones(1, 2)).sum()
    session.optimizer_step(loss, optimizer)
    assert not torch.equal(before, model.weight.detach())
