from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory, DirectoryParsePolicy
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrRuntime,
    FindLrSpec,
    FindLrSteps,
)
from dl_toolbox.dl_engine.runs.common.run_config import Accelerator
from tests.dl_engine.runs.support import (
    LinearBinary,
    RecordingLogger,
    binary_loader,
    forward_fn,
    loss_fn,
    prepare_tensors,
    run_log,
)


class FindLrLogger(RecordingLogger):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str, object, int]] = []
        self.tables: list[tuple[str, list[Mapping[str, Any]], int]] = []

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        self.images.append((tag, image, int(step)))

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.tables.append((tag, [dict(row) for row in rows], int(step)))


def range_spec() -> FindLrSpec:
    return FindLrSpec.range(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=4,
        smooth_f=0.05,
        diverge_th=4.0,
    )


def full_ds_spec(*, num_iter: int = 4, num_iterations_per_lr: int = 1) -> FindLrSpec:
    return FindLrSpec.full_ds(
        start_lr=1e-4,
        end_lr=1e-1,
        num_iter=num_iter,
        num_iterations_per_lr=num_iterations_per_lr,
    )


def cpu_runtime() -> FindLrRuntime:
    return FindLrRuntime(
        accelerator=Accelerator.CPU,
        precision=32,
        enable_progress_bar=False,
    )


def finder_steps() -> FindLrSteps:
    return FindLrSteps(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
    )


def artifacts_directory(tmp_path) -> Directory:
    return Directory.from_absolute(
        path=tmp_path, policy=DirectoryParsePolicy.MUST_EXIST
    )


def net_optimizer_loader(
    *, n: int = 16, batch_size: int = 4
) -> tuple[nn.Module, Optimizer, DataLoader]:
    import torch

    net = LinearBinary()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loader = binary_loader(n=n, batch_size=batch_size)
    return net, optimizer, loader


__all__ = [
    "FindLrLogger",
    "artifacts_directory",
    "cpu_runtime",
    "finder_steps",
    "full_ds_spec",
    "net_optimizer_loader",
    "range_spec",
    "run_log",
]
