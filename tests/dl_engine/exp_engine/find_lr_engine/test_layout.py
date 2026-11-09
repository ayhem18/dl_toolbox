from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from dl_toolbox.dl_engine.exp_engine.find_lr_engine.default_layout import (
    DefaultLayoutFindLrEngine,
)
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.experiment import FindLrParts
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.find_lr_engine import FindLrEngine
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    IdentityOverwritePolicy,
)
from dl_toolbox.dl_engine.runs.lr_finder_run.find_lr_run import FindLrRun
from tests.dl_engine.exp_engine.find_lr_engine.support import (
    LinearBinaryFindLrExperiment,
    find_engine,
    find_lr_payload,
    scaffold_project,
    write_recipe,
)
from tests.dl_engine.exp_engine.support import layout_from_root


def test_constructor_keys_and_no_defaults() -> None:
    assert DefaultLayoutFindLrEngine.CONSTRUCTOR_KEYS == ("force",)
    assert DefaultLayoutFindLrEngine.RESOLVED_CONFIG_FILENAME == "resolved_config.json"
    params = inspect.signature(FindLrEngine.find).parameters
    assert list(params) == ["self"]
    for name, param in inspect.signature(
        DefaultLayoutFindLrEngine.__init__
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
        payload=find_lr_payload(name="keys", run_group=[], seed=1),
    )
    layout = layout_from_root(root, name="proj_keys", backend="tensorboard")
    with pytest.raises(ValueError, match="keys must be"):
        DefaultLayoutFindLrEngine(
            layout=layout,
            recipe=recipe,
            experiment=LinearBinaryFindLrExperiment(),
        )
    with pytest.raises(ValueError, match="keys must be"):
        DefaultLayoutFindLrEngine(
            layout=layout,
            recipe=recipe,
            experiment=LinearBinaryFindLrExperiment(),
            force=IdentityOverwritePolicy.OFF,
            inspect_recipe=recipe,
        )


def test_experiment_must_have_build_find_lr(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_exp", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/exp.json",
        payload=find_lr_payload(name="exp", run_group=[], seed=1),
    )
    layout = layout_from_root(root, name="proj_exp", backend="tensorboard")

    class NoFinder:
        pass

    with pytest.raises(TypeError, match="build_find_lr"):
        DefaultLayoutFindLrEngine(
            layout=layout,
            recipe=recipe,
            experiment=NoFinder(),
            force=IdentityOverwritePolicy.OFF,
        )


def test_parts_require_identity(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_id", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/id.json",
        payload=find_lr_payload(name="id", run_group=[], seed=1),
    )
    engine = find_engine(
        root, project_name="proj_id", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    with pytest.raises(RuntimeError, match="claim_fit_identity"):
        engine.get_logger()
    with pytest.raises(RuntimeError, match="claim_fit_identity"):
        engine.get_run_log()
    with pytest.raises(RuntimeError, match="claim_fit_identity"):
        engine.get_net()
    engine.claim_fit_identity()
    assert isinstance(engine.get_net(), torch.nn.Module)
    log = engine.get_run_log()
    assert log.filename == DefaultLayoutFindLrEngine.RESOLVED_CONFIG_FILENAME
    assert log.payload["human_readable_name"] == "id"


def test_find_writes_curve_and_restores(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_find", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/find.json",
        payload=find_lr_payload(name="find", run_group=["lr_finder"], seed=7),
    )
    experiment = LinearBinaryFindLrExperiment()
    engine = find_engine(
        root,
        project_name="proj_find",
        recipe=recipe,
        force=IdentityOverwritePolicy.OFF,
        experiment=experiment,
    )
    result = engine.find()
    assert result.curve.path.name == FindLrRun.CURVE_FILENAME
    assert result.plot.path.name == FindLrRun.PLOT_FILENAME
    assert result.curve.path.is_file()
    assert result.plot.path.is_file()
    assert result.kind == engine.get_spec().kind
    assert experiment.initial_net_state is not None
    after = engine.get_net().state_dict()
    keys = list(experiment.initial_net_state)
    for i in range(len(keys)):
        key = keys[i]
        assert torch.equal(experiment.initial_net_state[key], after[key])
    assert engine.get_optimizer().param_groups[0]["lr"] == pytest.approx(
        experiment.initial_lr
    )
    again = find_engine(
        root, project_name="proj_find", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        again.find()


def test_find_lr_parts_types() -> None:
    with pytest.raises(TypeError, match="net must be"):
        FindLrParts(
            net=object(),
            optimizer=object(),
            train_loader=object(),
            steps=object(),
        )
