"""Return type of experiment.build_find_lr. Not JSON."""

from __future__ import annotations

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.runs.lr_finder_run.spec import FindLrSteps


class FindLrParts:
    def __init__(
        self,
        *,
        net: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        steps: FindLrSteps,
    ) -> None:
        if not isinstance(net, nn.Module):
            raise TypeError(f"net must be nn.Module, got {type(net)!r}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be Optimizer, got {type(optimizer)!r}")
        if not isinstance(train_loader, DataLoader):
            raise TypeError(
                f"train_loader must be DataLoader, got {type(train_loader)!r}"
            )
        if not isinstance(steps, FindLrSteps):
            raise TypeError(f"steps must be FindLrSteps, got {type(steps)!r}")
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.steps = steps
