from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
    devices_and_strategy,
    pick_visible_gpu_indices,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)
from dl_toolbox.dl_engine.runs.common.run_config import Accelerator, Strategy
from tests.dl_engine.exp_engine.support import (
    LinearBinaryExperiment,
    cpu_fit_payload,
    gpu_fit_payload,
    inspect_payload,
    layout_from_root,
    scaffold_project,
    write_recipe,
)


def test_cpu_devices_are_empty_tuple() -> None:
    devices, strategy = devices_and_strategy(
        accelerator=Accelerator.CPU, devices_field=[]
    )
    assert devices == ()
    assert strategy == Strategy.SINGLE_DEVICE


def test_gpu_count_picks_first_visible_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout.torch.cuda.device_count",
        lambda: 4,
    )
    devices, strategy = devices_and_strategy(
        accelerator=Accelerator.GPU, devices_field=2
    )
    assert devices == (0, 1)
    assert strategy == Strategy.DDP
    one, one_strategy = devices_and_strategy(
        accelerator=Accelerator.GPU, devices_field=1
    )
    assert one == (0,)
    assert one_strategy == Strategy.SINGLE_DEVICE


def test_gpu_count_errors_when_not_enough_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout.torch.cuda.device_count",
        lambda: 1,
    )
    with pytest.raises(ValueError, match="need 3 visible CUDA devices"):
        pick_visible_gpu_indices(count=3)


def test_explicit_gpu_indices_skip_the_pick() -> None:
    devices, strategy = devices_and_strategy(
        accelerator=Accelerator.GPU, devices_field=[3, 1]
    )
    assert devices == (3, 1)
    assert strategy == Strategy.DDP


def test_get_trainer_converts_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout.torch.cuda.device_count",
        lambda: 2,
    )
    root = scaffold_project(tmp_path, name="proj_gpu", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/gpu.json",
        payload=gpu_fit_payload(name="gpu_run", devices=2, seed=1),
    )
    layout = layout_from_root(root, name="proj_gpu", backend="tensorboard")
    engine = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=recipe,
        experiment=LinearBinaryExperiment(),
        force=IdentityOverwritePolicy.OFF,
    )
    trainer = engine.get_trainer()
    assert trainer.accelerator == Accelerator.GPU
    assert trainer.devices == (0, 1)
    assert trainer.strategy == Strategy.DDP


def test_identity_payload_excludes_logger_labels(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_hash", backend="tensorboard")
    first = write_recipe(
        root,
        relative_path="run_recipes/a.json",
        payload=cpu_fit_payload(
            name="tagged", run_group=[], seed=4, tags={"task": "one"}
        ),
    )
    second = write_recipe(
        root,
        relative_path="run_recipes/b.json",
        payload=cpu_fit_payload(
            name="tagged", run_group=[], seed=4, tags={"task": "two"}
        ),
    )
    layout = layout_from_root(root, name="proj_hash", backend="tensorboard")
    left = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=first,
        experiment=LinearBinaryExperiment(),
        force=IdentityOverwritePolicy.OFF,
    )
    right = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=second,
        experiment=LinearBinaryExperiment(),
        force=IdentityOverwritePolicy.OFF,
    )
    assert left.get_identity_payload() == right.get_identity_payload()
    assert "tags" not in left.get_identity_payload()
    left.process_run_identity()
    with pytest.raises(FileExistsError, match="already claimed"):
        right.process_run_identity()


def test_purpose_mismatch(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_p", backend="tensorboard")
    fit = write_recipe(
        root,
        relative_path="run_recipes/fit.json",
        payload=cpu_fit_payload(name="p", run_group=[], seed=1),
    )
    inspect = write_recipe(
        root,
        relative_path="run_recipes/inspect.json",
        payload=inspect_payload(),
    )
    layout = layout_from_root(root, name="proj_p", backend="tensorboard")
    fit_engine = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=fit,
        experiment=LinearBinaryExperiment(),
        force=IdentityOverwritePolicy.OFF,
    )
    inspect_engine = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        fit_recipe=fit,
        experiment=LinearBinaryExperiment(),
        inspect_recipe=inspect,
        eval_device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="inspect requires purpose"):
        fit_engine.inspect()
    with pytest.raises(ValueError, match="train requires purpose"):
        inspect_engine.train()


def test_cpu_devices_reject_gpu_indices(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_bad", backend="tensorboard")
    payload = cpu_fit_payload(name="bad", run_group=[], seed=1)
    payload["trainer"] = dict(payload["trainer"])
    payload["trainer"]["devices"] = [0]
    recipe = write_recipe(
        root, relative_path="run_recipes/bad.json", payload=payload
    )
    layout = layout_from_root(root, name="proj_bad", backend="tensorboard")
    with pytest.raises(ValueError, match="cpu trainer.devices"):
        DefaultLayoutTrainValEngine(
            purpose=EnginePurpose.FIT,
            layout=layout,
            fit_recipe=recipe,
            experiment=LinearBinaryExperiment(),
            force=IdentityOverwritePolicy.OFF,
        )
