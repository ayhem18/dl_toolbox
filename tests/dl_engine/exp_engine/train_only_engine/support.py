from __future__ import annotations

from typing import Any

import torch

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.file_management.file import File
from dl_toolbox.dl_engine.exp_engine.train_only_engine.default_layout import (
    DefaultLayoutTrainOnlyEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_only_engine.experiment import (
    TrainOnlyEvalParts,
    TrainOnlyFitParts,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
)
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_config import EvalContext, EvalLoaders, LogItem
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import TrainOnlyLoaders
from tests.dl_engine.exp_engine.support import layout_from_root
from tests.dl_engine.runs.support import (
    LinearBinary,
    binary_loader,
    binary_specs,
    steps,
    train_only_metrics,
)


class LinearBinaryTrainOnlyExperiment:
    def build_train_only(
        self, experiment_config: dict[str, Any], *, project_root: Directory
    ) -> TrainOnlyFitParts:
        del experiment_config, project_root
        net = LinearBinary()
        loader = binary_loader(n=32, batch_size=8)
        return TrainOnlyFitParts(
            net=net,
            loaders=TrainOnlyLoaders(train=loader),
            steps=steps(),
            metrics=train_only_metrics(binary_specs()),
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
    ) -> TrainOnlyEvalParts:
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

        return TrainOnlyEvalParts(
            loaders=EvalLoaders(named={"val": loader}),
            evaluate_fn=evaluate_fn,
        )


def fit_engine(root, *, project_name: str, recipe: File, force: str):
    layout = layout_from_root(root, name=project_name, backend="tensorboard")
    return DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=recipe,
        experiment=LinearBinaryTrainOnlyExperiment(),
        force=force,
    )
