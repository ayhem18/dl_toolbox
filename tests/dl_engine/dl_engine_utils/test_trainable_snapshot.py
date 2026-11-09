from __future__ import annotations

import pytest
import torch
from torch import nn

from dl_toolbox.dl_engine.dl_engine_utils.trainable_snapshot import (
    restore_trainable,
    snapshot_trainable,
)


def test_snapshot_restore_puts_weights_and_optimizer_back() -> None:
    net = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    snapshot = snapshot_trainable(model=net, optimizer=optimizer)
    before = net.weight.detach().clone()
    optimizer.zero_grad()
    loss = net(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, net.weight.detach())
    restore_trainable(
        model=net,
        optimizer=optimizer,
        snapshot=snapshot,
        device=torch.device("cpu"),
    )
    assert torch.equal(before, net.weight.detach())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
