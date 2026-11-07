from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import JsonTextForm, write
from dl_toolbox.dl_engine.exp_engine.train_val_engine.experiment import (
    TrainValEvalParts,
    TrainValFitParts,
)
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_config import EvalContext, EvalLoaders, LogItem
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import TrainValLoaders
from dl_toolbox.project_layout import ProjectLayout, TrackingBackend
from tests.dl_engine.runs.support import (
    LinearBinary,
    binary_loader,
    binary_specs,
    steps,
    train_val_metrics,
)


def existing_root(path: Path) -> Directory:
    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.CREATE)


def scaffold_project(tmp_path: Path, *, name: str, backend: str) -> Directory:
    root = existing_root(tmp_path / name)
    ProjectLayout.scaffold(
        root=root, project_name=name, tracking_backend=backend
    )
    return root


def layout_from_root(root: Directory, *, name: str, backend: str) -> ProjectLayout:
    return ProjectLayout.from_start(
        start=root, project_name=name, tracking_backend=backend
    )


def cpu_fit_payload(
    *,
    name: str,
    run_group: list[str],
    seed: int,
    experiment_config: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    max_epochs: int = 1,
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
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "cpu",
            "devices": [],
            "precision": 32,
            "enable_progress_bar": False,
            "accumulate_grad_batches": 1,
        },
        "checkpoint_policy": {"save_top_k": -1, "save_last": True},
        "early_stopping": {"name": "none"},
        "logger": logger,
        "experiment_config": (
            experiment_config if experiment_config is not None else {"width": 4}
        ),
    }


def gpu_fit_payload(
    *,
    name: str,
    devices: int | list[int],
    seed: int,
) -> dict[str, Any]:
    payload = cpu_fit_payload(name=name, run_group=[], seed=seed)
    trainer = dict(payload["trainer"])
    trainer["accelerator"] = "gpu"
    trainer["devices"] = devices
    payload["trainer"] = trainer
    return payload


def write_recipe(root: Directory, *, relative_path: str, payload: dict[str, Any]) -> File:
    parent_relative, name = relative_path.rsplit("/", 1)
    directory = Directory.from_under(
        root=root,
        relative_path=parent_relative,
        policy=DirectoryParsePolicy.CREATE,
    )
    return write(
        directory=directory,
        name=name,
        payload=payload,
        form=JsonTextForm.DOCUMENT,
    )


def inspect_payload(*, selector: str = "last") -> dict[str, Any]:
    return {"selector": selector, "evaluate": {"split": "val"}}


class LinearBinaryExperiment:
    def build_fit(self, experiment_config: dict[str, Any], *, project_root: Directory) -> TrainValFitParts:
        del experiment_config, project_root
        net = LinearBinary()
        loader = binary_loader(n=32, batch_size=8)
        return TrainValFitParts(
            net=net,
            loaders=TrainValLoaders(train=loader, val=loader),
            steps=steps(),
            metrics=train_val_metrics(binary_specs()),
            optimization=OptimizationConfig(
                optimizer=torch.optim.Adam(net.parameters(), lr=0.05),
                scheduler=OptimizationConfig.ABSENT,
            ),
        )

    def build_eval(
        self,
        experiment_config: dict[str, Any],
        evaluate: dict[str, Any],
        *,
        project_root: Directory,
        payload_dir: Directory,
    ) -> TrainValEvalParts:
        del experiment_config, evaluate, project_root
        loader = binary_loader(n=16, batch_size=8)

        def evaluate_fn(context: EvalContext) -> list[LogItem]:
            marker = payload_dir.path / "inspect.txt"
            marker.write_text(f"step={context.step}\n", encoding="utf-8")
            return [
                LogItem.scalars(
                    step=context.step, values={"inspect/ok": 1.0}
                )
            ]

        return TrainValEvalParts(
            loaders=EvalLoaders(named={"val": loader}),
            evaluate_fn=evaluate_fn,
        )
