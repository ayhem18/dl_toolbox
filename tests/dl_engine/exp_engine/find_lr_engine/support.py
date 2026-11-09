from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dl_toolbox.base.file_management.file import File
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.default_layout import (
    DefaultLayoutFindLrEngine,
)
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.experiment import FindLrParts
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    IdentityOverwritePolicy,
)
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import FindLrSteps
from tests.dl_engine.exp_engine.support import (
    layout_from_root,
    scaffold_project,
    write_recipe,
)
from tests.dl_engine.runs.support import LinearBinary, binary_loader, steps


def find_lr_payload(
    *,
    name: str,
    run_group: list[str],
    seed: int,
    experiment_config: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    logger_experiment: str | None = None,
) -> dict[str, Any]:
    logger: dict[str, Any] = {}
    if tags is not None:
        logger["tags"] = tags
    if logger_experiment is not None:
        logger["experiment"] = logger_experiment
    return {
        "human_readable_name": name,
        "run_group": list(run_group),
        "seed": seed,
        "lr_finder": {
            "kind": "range",
            "start_lr": 1e-4,
            "end_lr": 1e-1,
            "num_iter": 4,
            "smooth_f": 0.05,
            "diverge_th": 4.0,
        },
        "runtime": {
            "accelerator": "cpu",
            "precision": 32,
            "enable_progress_bar": False,
        },
        "logger": logger,
        "experiment_config": (
            experiment_config if experiment_config is not None else {"width": 4}
        ),
    }


class LinearBinaryFindLrExperiment:
    def __init__(self) -> None:
        self.initial_net_state: dict[str, torch.Tensor] | None = None
        self.initial_lr: float | None = None

    def build_find_lr(
        self, experiment_config: dict[str, Any], *, project_root: object
    ) -> FindLrParts:
        del experiment_config, project_root
        net = LinearBinary()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        self.initial_net_state = {
            key: value.detach().clone() for key, value in net.state_dict().items()
        }
        self.initial_lr = float(optimizer.param_groups[0]["lr"])
        return FindLrParts(
            net=net,
            optimizer=optimizer,
            train_loader=binary_loader(n=32, batch_size=8),
            steps=FindLrSteps.from_fit_steps(steps()),
        )


def find_engine(
    root,
    *,
    project_name: str,
    recipe: File,
    force: str,
    experiment: LinearBinaryFindLrExperiment | None = None,
    backend: str = "tensorboard",
) -> DefaultLayoutFindLrEngine:
    layout = layout_from_root(root, name=project_name, backend=backend)
    bound = experiment if experiment is not None else LinearBinaryFindLrExperiment()
    return DefaultLayoutFindLrEngine(
        layout=layout,
        recipe=recipe,
        experiment=bound,
        force=force,
    )


__all__ = [
    "IdentityOverwritePolicy",
    "LinearBinaryFindLrExperiment",
    "Path",
    "find_engine",
    "find_lr_payload",
    "scaffold_project",
    "write_recipe",
]
