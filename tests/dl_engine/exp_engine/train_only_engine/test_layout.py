from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from dl_toolbox.dl_engine.exp_engine.train_only_engine.default_layout import (
    DefaultLayoutTrainOnlyEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_only_engine.train_only_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
    TrainOnlyEngine,
)
from dl_toolbox.dl_engine.runs.common.run_config import Accelerator, Strategy
from tests.dl_engine.exp_engine.support import (
    cpu_fit_payload,
    gpu_fit_payload,
    inspect_payload,
    layout_from_root,
    scaffold_project,
    write_recipe,
)
from tests.dl_engine.exp_engine.train_only_engine.support import (
    LinearBinaryTrainOnlyExperiment,
    fit_engine,
)


def test_constructor_keys_and_no_defaults() -> None:
    assert DefaultLayoutTrainOnlyEngine.FIT_CONSTRUCTOR_KEYS == ("force",)
    assert DefaultLayoutTrainOnlyEngine.INSPECT_CONSTRUCTOR_KEYS == (
        "inspect_recipe",
        "eval_device",
    )
    assert DefaultLayoutTrainOnlyEngine.RESOLVED_CONFIG_FILENAME == "resolved_config.json"
    for method in (TrainOnlyEngine.train, TrainOnlyEngine.inspect):
        params = inspect.signature(method).parameters
        assert list(params) == ["self"]
    for name, param in inspect.signature(
        DefaultLayoutTrainOnlyEngine.__init__
    ).parameters.items():
        if name in ("self", "fields"):
            continue
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, name


def test_constructor_requires_force(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_keys", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/keys.json",
        payload=cpu_fit_payload(name="keys", run_group=[], seed=1),
    )
    layout = layout_from_root(root, name="proj_keys", backend="tensorboard")
    with pytest.raises(ValueError, match="keys must be"):
        DefaultLayoutTrainOnlyEngine(
            purpose=EnginePurpose.FIT,
            layout=layout,
            train_recipe=recipe,
            experiment=LinearBinaryTrainOnlyExperiment(),
        )


def test_experiment_must_have_build_train_only(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_exp", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/exp.json",
        payload=cpu_fit_payload(name="exp", run_group=[], seed=1),
    )
    layout = layout_from_root(root, name="proj_exp", backend="tensorboard")

    class NoTrainOnly:
        def build_eval(self) -> None:
            return None

    with pytest.raises(TypeError, match="build_train_only"):
        DefaultLayoutTrainOnlyEngine(
            purpose=EnginePurpose.FIT,
            layout=layout,
            train_recipe=recipe,
            experiment=NoTrainOnly(),
            force=IdentityOverwritePolicy.OFF,
        )


def test_experiment_must_have_build_eval(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_eval", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/eval.json",
        payload=cpu_fit_payload(name="eval", run_group=[], seed=1),
    )
    layout = layout_from_root(root, name="proj_eval", backend="tensorboard")

    class NoEval:
        def build_train_only(self) -> None:
            return None

    with pytest.raises(TypeError, match="build_eval"):
        DefaultLayoutTrainOnlyEngine(
            purpose=EnginePurpose.FIT,
            layout=layout,
            train_recipe=recipe,
            experiment=NoEval(),
            force=IdentityOverwritePolicy.OFF,
        )


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
    left = fit_engine(
        root, project_name="proj_hash", recipe=first, force=IdentityOverwritePolicy.OFF
    )
    right = fit_engine(
        root, project_name="proj_hash", recipe=second, force=IdentityOverwritePolicy.OFF
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
    fit_eng = DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=fit,
        experiment=LinearBinaryTrainOnlyExperiment(),
        force=IdentityOverwritePolicy.OFF,
    )
    inspect_engine = DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        train_recipe=fit,
        experiment=LinearBinaryTrainOnlyExperiment(),
        inspect_recipe=inspect,
        eval_device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="inspect requires purpose"):
        fit_eng.inspect()
    with pytest.raises(ValueError, match="train requires purpose"):
        inspect_engine.train()


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
    engine = DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=recipe,
        experiment=LinearBinaryTrainOnlyExperiment(),
        force=IdentityOverwritePolicy.OFF,
    )
    trainer = engine.get_trainer()
    assert trainer.accelerator == Accelerator.GPU
    assert trainer.devices == (0, 1)
    assert trainer.strategy == Strategy.DDP


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
        DefaultLayoutTrainOnlyEngine(
            purpose=EnginePurpose.FIT,
            layout=layout,
            train_recipe=recipe,
            experiment=LinearBinaryTrainOnlyExperiment(),
            force=IdentityOverwritePolicy.OFF,
        )
