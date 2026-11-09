from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import DirectoryParsePolicy
from dl_toolbox.base.input_output.json_io import JsonTextForm, write_object
from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)
from dl_toolbox.project_layout.names import (
    CHECKPOINT_CONFIG_FILENAME,
    PROJECT_LEVEL_CONFIGS_DIRNAME,
)
from tests.dl_engine.exp_engine.support import (
    LinearBinaryExperiment,
    cpu_fit_payload,
    inspect_payload,
    layout_from_root,
    scaffold_project,
    write_recipe,
)


def _fit_engine(root, *, project_name: str, recipe, force: str):
    layout = layout_from_root(
        root, name=project_name, backend="tensorboard"
    )
    return DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=recipe,
        experiment=LinearBinaryExperiment(),
        force=force,
    )


def test_both_free_then_both_match(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_a", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/alpha.json",
        payload=cpu_fit_payload(name="alpha", run_group=[], seed=1),
    )
    engine = _fit_engine(root, project_name="proj_a", recipe=recipe, force=IdentityOverwritePolicy.OFF)
    engine.process_run_identity()
    artifacts = engine.get_artifacts_directory()
    assert artifacts.path.is_dir()
    again = _fit_engine(root, project_name="proj_a", recipe=recipe, force=IdentityOverwritePolicy.OFF)
    with pytest.raises(FileExistsError, match="already claimed"):
        again.process_run_identity()
    overwrite = _fit_engine(
        root, project_name="proj_a", recipe=recipe, force=IdentityOverwritePolicy.OVERWRITE_NAME
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        overwrite.process_run_identity()


def test_hash_taken_ignores_force(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_b", backend="tensorboard")
    first = write_recipe(
        root,
        relative_path="run_recipes/one.json",
        payload=cpu_fit_payload(name="one", run_group=[], seed=3),
    )
    second_payload = cpu_fit_payload(name="two", run_group=[], seed=3)
    second = write_recipe(
        root, relative_path="run_recipes/two.json", payload=second_payload
    )
    _fit_engine(root, project_name="proj_b", recipe=first, force=IdentityOverwritePolicy.OFF).process_run_identity()
    with pytest.raises(FileExistsError, match="configuration hash"):
        _fit_engine(
            root, project_name="proj_b", recipe=second, force=IdentityOverwritePolicy.OVERWRITE_NAME
        ).process_run_identity()


def test_name_taken_overwrite_releases(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_c", backend="tensorboard")
    first = write_recipe(
        root,
        relative_path="run_recipes/same.json",
        payload=cpu_fit_payload(name="same", run_group=[], seed=1),
    )
    engine = _fit_engine(root, project_name="proj_c", recipe=first, force=IdentityOverwritePolicy.OFF)
    engine.process_run_identity()
    marker = engine.get_artifacts_directory().path / "stale.txt"
    marker.write_text("stale\n", encoding="utf-8")
    other = write_recipe(
        root,
        relative_path="run_recipes/same_other.json",
        payload=cpu_fit_payload(name="same", run_group=[], seed=2),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _fit_engine(
            root, project_name="proj_c", recipe=other, force=IdentityOverwritePolicy.OFF
        ).process_run_identity()
    _fit_engine(
        root, project_name="proj_c", recipe=other, force=IdentityOverwritePolicy.OVERWRITE_NAME
    ).process_run_identity()
    assert not marker.is_file()
    assert engine.get_artifacts_directory().path.is_dir()


def test_run_group_isolates_the_same_name(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_d", backend="tensorboard")
    flat = write_recipe(
        root,
        relative_path="run_recipes/flat.json",
        payload=cpu_fit_payload(name="shared", run_group=[], seed=1),
    )
    nested = write_recipe(
        root,
        relative_path="run_recipes/nested.json",
        payload=cpu_fit_payload(
            name="shared", run_group=["learning_rate_grid", "less_than10"], seed=1
        ),
    )
    _fit_engine(root, project_name="proj_d", recipe=flat, force=IdentityOverwritePolicy.OFF).process_run_identity()
    nested_engine = _fit_engine(
        root, project_name="proj_d", recipe=nested, force=IdentityOverwritePolicy.OFF
    )
    nested_engine.process_run_identity()
    artifacts = nested_engine.get_artifacts_directory().path
    assert artifacts.as_posix().endswith(
        "run_artifacts/learning_rate_grid/less_than10/shared"
    )


def test_orphan_artifacts_are_an_error(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_e", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/orphan.json",
        payload=cpu_fit_payload(name="orphan", run_group=[], seed=1),
    )
    engine = _fit_engine(root, project_name="proj_e", recipe=recipe, force=IdentityOverwritePolicy.OFF)
    engine.get_artifacts_directory().path.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="orphan"):
        engine.process_run_identity()


def test_inspect_missing_pair_is_immediate(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_f", backend="tensorboard")
    fit = write_recipe(
        root,
        relative_path="run_recipes/fit.json",
        payload=cpu_fit_payload(name="ghost", run_group=[], seed=1),
    )
    inspect = write_recipe(
        root,
        relative_path="run_recipes/inspect.json",
        payload=inspect_payload(),
    )
    layout = layout_from_root(root, name="proj_f", backend="tensorboard")
    engine = DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        fit_recipe=fit,
        experiment=LinearBinaryExperiment(),
        inspect_recipe=inspect,
        eval_device=torch_device_cpu(),
    )
    with pytest.raises(FileNotFoundError):
        engine.inspect()


def test_worker_attaches_to_claimed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = scaffold_project(tmp_path, name="proj_g", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/ddp.json",
        payload=cpu_fit_payload(name="ddp", run_group=[], seed=1),
    )
    _fit_engine(root, project_name="proj_g", recipe=recipe, force=IdentityOverwritePolicy.OFF).process_run_identity()
    monkeypatch.setenv("RANK", "1")
    worker = _fit_engine(
        root, project_name="proj_g", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    worker.process_run_identity()
    assert worker._identity_ready
    assert not worker._claimed


def test_worker_times_out_when_pair_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = scaffold_project(tmp_path, name="proj_h", backend="tensorboard")
    configs = root.path / PROJECT_LEVEL_CONFIGS_DIRNAME
    write_object(
        directory=_must_exist(configs),
        name=CHECKPOINT_CONFIG_FILENAME,
        payload={"lock_timeout_seconds": 0.2},
        form=JsonTextForm.DOCUMENT,
    )
    recipe = write_recipe(
        root,
        relative_path="run_recipes/wait.json",
        payload=cpu_fit_payload(name="wait", run_group=[], seed=1),
    )
    monkeypatch.setenv("RANK", "1")
    engine = _fit_engine(
        root, project_name="proj_h", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    with pytest.raises(FileNotFoundError):
        engine.process_run_identity()


def test_constructor_omits_illegal_purpose_fields(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_i", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/fit.json",
        payload=cpu_fit_payload(name="keys", run_group=[], seed=1),
    )
    inspect = write_recipe(
        root,
        relative_path="run_recipes/inspect.json",
        payload=inspect_payload(),
    )
    layout = layout_from_root(root, name="proj_i", backend="tensorboard")
    with pytest.raises(ValueError, match="keys must be"):
        DefaultLayoutTrainValEngine(
            purpose=EnginePurpose.FIT,
            layout=layout,
            fit_recipe=recipe,
            experiment=LinearBinaryExperiment(),
            force=IdentityOverwritePolicy.OFF,
            inspect_recipe=inspect,
        )
    with pytest.raises(ValueError, match="keys must be"):
        DefaultLayoutTrainValEngine(
            purpose=EnginePurpose.INSPECT,
            layout=layout,
            fit_recipe=recipe,
            experiment=LinearBinaryExperiment(),
            inspect_recipe=inspect,
            eval_device=torch_device_cpu(),
            force=IdentityOverwritePolicy.OFF,
        )


def torch_device_cpu():
    import torch

    return torch.device("cpu")


def _must_exist(path: Path):
    from dl_toolbox.base.file_management.directory import Directory

    return Directory.from_absolute(path=path, policy=DirectoryParsePolicy.MUST_EXIST)
