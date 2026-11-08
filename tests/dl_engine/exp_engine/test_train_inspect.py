from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)
from tests.dl_engine.exp_engine.support import (
    LinearBinaryExperiment,
    cpu_fit_payload,
    inspect_payload,
    layout_from_root,
    scaffold_project,
    write_recipe,
)


def test_train_cpu_then_inspect_last(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_tv", backend="tensorboard")
    fit_recipe = write_recipe(
        root,
        relative_path="run_recipes/fit.json",
        payload=cpu_fit_payload(
            name="linear_binary", run_group=[], seed=7, max_epochs=1
        ),
    )
    inspect_recipe = write_recipe(
        root,
        relative_path="run_recipes/inspect.json",
        payload=inspect_payload(selector="last"),
    )
    layout = layout_from_root(root, name="proj_tv", backend="tensorboard")
    DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=fit_recipe,
        experiment=LinearBinaryExperiment(),
        force=IdentityOverwritePolicy.OFF,
    ).train()
    inspect_engine = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        fit_recipe=fit_recipe,
        experiment=LinearBinaryExperiment(),
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
    fit_recipe = write_recipe(
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
    DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=fit_recipe,
        experiment=LinearBinaryExperiment(),
        force=IdentityOverwritePolicy.OFF,
    ).train()
    DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        fit_recipe=fit_recipe,
        experiment=LinearBinaryExperiment(),
        inspect_recipe=inspect_recipe,
        eval_device=torch.device("cpu"),
    ).inspect()
