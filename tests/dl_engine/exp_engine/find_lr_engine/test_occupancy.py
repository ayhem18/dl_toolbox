from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.dl_engine.exp_engine.find_lr_engine.find_lr_engine import (
    FindLrEngineError,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    IdentityOverwritePolicy,
)
from tests.dl_engine.exp_engine.find_lr_engine.support import (
    find_engine,
    find_lr_payload,
    scaffold_project,
    write_recipe,
)


def test_both_free_then_both_match(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_a", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/alpha.json",
        payload=find_lr_payload(name="alpha", run_group=[], seed=1),
    )
    engine = find_engine(
        root, project_name="proj_a", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    engine.claim_fit_identity()
    artifacts = engine.get_artifacts_directory()
    assert artifacts.path.is_dir()
    again = find_engine(
        root, project_name="proj_a", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        again.claim_fit_identity()
    overwrite = find_engine(
        root,
        project_name="proj_a",
        recipe=recipe,
        force=IdentityOverwritePolicy.OVERWRITE_NAME,
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        overwrite.claim_fit_identity()


def test_hash_taken_ignores_force(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_b", backend="tensorboard")
    first = write_recipe(
        root,
        relative_path="run_recipes/one.json",
        payload=find_lr_payload(name="one", run_group=[], seed=3),
    )
    second = write_recipe(
        root,
        relative_path="run_recipes/two.json",
        payload=find_lr_payload(name="two", run_group=[], seed=3),
    )
    find_engine(
        root, project_name="proj_b", recipe=first, force=IdentityOverwritePolicy.OFF
    ).claim_fit_identity()
    with pytest.raises(FileExistsError, match="configuration hash"):
        find_engine(
            root,
            project_name="proj_b",
            recipe=second,
            force=IdentityOverwritePolicy.OVERWRITE_NAME,
        ).claim_fit_identity()


def test_name_taken_overwrite_releases(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_c", backend="tensorboard")
    first = write_recipe(
        root,
        relative_path="run_recipes/same.json",
        payload=find_lr_payload(name="same", run_group=[], seed=1),
    )
    engine = find_engine(
        root, project_name="proj_c", recipe=first, force=IdentityOverwritePolicy.OFF
    )
    engine.claim_fit_identity()
    marker = engine.get_artifacts_directory().path / "stale.txt"
    marker.write_text("stale\n", encoding="utf-8")
    other = write_recipe(
        root,
        relative_path="run_recipes/same_other.json",
        payload=find_lr_payload(name="same", run_group=[], seed=2),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        find_engine(
            root, project_name="proj_c", recipe=other, force=IdentityOverwritePolicy.OFF
        ).claim_fit_identity()
    find_engine(
        root,
        project_name="proj_c",
        recipe=other,
        force=IdentityOverwritePolicy.OVERWRITE_NAME,
    ).claim_fit_identity()
    assert not marker.is_file()
    assert engine.get_artifacts_directory().path.is_dir()


def test_run_group_isolates_the_same_name(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_d", backend="tensorboard")
    flat = write_recipe(
        root,
        relative_path="run_recipes/flat.json",
        payload=find_lr_payload(name="shared", run_group=[], seed=1),
    )
    nested = write_recipe(
        root,
        relative_path="run_recipes/nested.json",
        payload=find_lr_payload(
            name="shared", run_group=["lr_finder", "less_than10"], seed=1
        ),
    )
    find_engine(
        root, project_name="proj_d", recipe=flat, force=IdentityOverwritePolicy.OFF
    ).claim_fit_identity()
    nested_engine = find_engine(
        root, project_name="proj_d", recipe=nested, force=IdentityOverwritePolicy.OFF
    )
    nested_engine.claim_fit_identity()
    artifacts = nested_engine.get_artifacts_directory().path
    assert artifacts.as_posix().endswith("run_artifacts/lr_finder/less_than10/shared")


def test_orphan_artifacts_are_an_error(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_e", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/orphan.json",
        payload=find_lr_payload(name="orphan", run_group=[], seed=1),
    )
    engine = find_engine(
        root, project_name="proj_e", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    engine.get_artifacts_directory().path.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="orphan"):
        engine.claim_fit_identity()


def test_worker_find_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = scaffold_project(tmp_path, name="proj_g", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/ddp.json",
        payload=find_lr_payload(name="ddp", run_group=[], seed=1),
    )
    monkeypatch.setenv("RANK", "1")
    engine = find_engine(
        root, project_name="proj_g", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    with pytest.raises(FindLrEngineError, match="DDP worker"):
        engine.find()
    assert not engine._identity_ready
    assert not engine._claimed
