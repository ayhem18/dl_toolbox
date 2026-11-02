"""Tiny binary-linear fixtures for lr_finder tests. Not collected by pytest."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.lr_finder import (
    LrFinderConfig,
    LrFinderRuntime,
    LrFinderSpec,
    LrFinderStep,
    RangeTestInput,
)
from dl_toolbox.dl_engine.metrics.contracts import KEY_TARGET


class LinearBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def prepare_tensors(batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
    features, labels = batch
    return {"input": features, KEY_TARGET: labels}


def forward(net: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    return net(batch["input"])


def loss_fn(output: Tensor, batch: dict[str, Tensor]) -> Tensor:
    target = batch[KEY_TARGET].float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(output, target)


def make_loader(*, n: int = 16, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_spec() -> LrFinderSpec:
    return LrFinderSpec(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=4,
        smooth_f=0.05,
        diverge_th=4.0,
    )


def make_step() -> LrFinderStep:
    return LrFinderStep(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward,
        loss_fn=loss_fn,
    )


def make_inputs(
    *,
    model: nn.Module | None = None,
    loader: DataLoader | None = None,
    spec: LrFinderSpec | None = None,
    loss: object | None = None,
    num_iterations_per_lr: int = 1,
) -> RangeTestInput:
    net = LinearBinary() if model is None else model
    step = make_step()
    if loss is not None:
        step = LrFinderStep(
            prepare_tensors_fn=prepare_tensors,
            forward_fn=forward,
            loss_fn=loss,
        )
    return RangeTestInput(
        model=net,
        optimizer=torch.optim.Adam(net.parameters(), lr=1e-4),
        train_loader=make_loader() if loader is None else loader,
        step=step,
        config=LrFinderConfig(
            spec=make_spec() if spec is None else spec,
            runtime=LrFinderRuntime(
                accelerator="cpu",
                precision=32,
                enable_progress_bar=False,
            ),
            num_iterations_per_lr=num_iterations_per_lr,
        ),
    )
