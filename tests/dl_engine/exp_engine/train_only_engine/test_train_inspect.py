from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dl_toolbox.dl_engine.exp_engine.train_only_engine.default_layout import (
    DefaultLayoutTrainOnlyEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_only_engine.train_only_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)
from tests.dl_engine.exp_engine.support import (
    cpu_fit_payload,
    inspect_payload,
    layout_from_root,
    scaffold_project,
    write_recipe,
)
from tests.dl_engine.exp_engine.train_only_engine.support import (
    LinearBinaryTrainOnlyExperiment,
)


def test_train_cpu_then_inspect_last(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_to", backend="tensorboard")
    train_recipe = write_recipe(
        root,
        relative_path="run_recipes/fit.json",
        payload=cpu_fit_payload(
            name="linear_binary", run_group=["pretrain"], seed=7, max_epochs=1
        ),
    )
    inspect_recipe = write_recipe(
        root,
        relative_path="run_recipes/inspect.json",
        payload=inspect_payload(selector="last"),
    )
    layout = layout_from_root(root, name="proj_to", backend="tensorboard")
    DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=train_recipe,
        experiment=LinearBinaryTrainOnlyExperiment(),
        force=IdentityOverwritePolicy.OFF,
    ).train()
    inspect_engine = DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        train_recipe=train_recipe,
        experiment=LinearBinaryTrainOnlyExperiment(),
        inspect_recipe=inspect_recipe,
        eval_device=torch.device("cpu"),
    )
    inspect_engine.inspect()
    artifacts = inspect_engine.get_artifacts_directory().path
    payload_dirs = list((artifacts / "inspection_runs").glob("epoch=*"))
    assert payload_dirs
    inspect_notes = list(payload_dirs[0].glob("inspect.txt"))
    assert inspect_notes
    catalog = artifacts / "checkpoints" / "catalog.json"
    assert catalog.is_file()
    text = catalog.read_text(encoding="utf-8")
    assert '"status": "finished"' in text


def test_mlflow_train_then_inspect(tmp_path: Path) -> None:
    pytest.importorskip("mlflow.tracking")
    root = scaffold_project(tmp_path, name="proj_ml", backend="mlflow")
    train_recipe = write_recipe(
        root,
        relative_path="run_recipes/fit.json",
        payload=cpu_fit_payload(
            name="ml_run",
            run_group=[],
            seed=2,
            max_epochs=1,
            logger_experiment="unit",
        ),
    )
    inspect_recipe = write_recipe(
        root,
        relative_path="run_recipes/inspect.json",
        payload=inspect_payload(),
    )
    layout = layout_from_root(root, name="proj_ml", backend="mlflow")
    DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=train_recipe,
        experiment=LinearBinaryTrainOnlyExperiment(),
        force=IdentityOverwritePolicy.OFF,
    ).train()
    DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        train_recipe=train_recipe,
        experiment=LinearBinaryTrainOnlyExperiment(),
        inspect_recipe=inspect_recipe,
        eval_device=torch.device("cpu"),
    ).inspect()
