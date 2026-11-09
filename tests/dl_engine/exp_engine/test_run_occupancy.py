from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream
from tests.dl_engine.exp_engine.find_lr_engine.support import (
    find_engine,
    find_lr_payload,
)
from tests.dl_engine.exp_engine.support import (
    LinearBinaryExperiment,
    cpu_fit_payload,
    layout_from_root,
    scaffold_project,
    write_recipe,
)


def _fit_engine(root, *, project_name: str, recipe, force: str):
    layout = layout_from_root(root, name=project_name, backend="tensorboard")
    return DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=recipe,
        experiment=LinearBinaryExperiment(),
        force=force,
    )


def _pair_state(engine) -> str:
    manager = engine.get_hash_manager()
    payload = engine.get_identity_payload()
    lookup = manager.lookup(
        engine.get_identity_index(),
        human_readable_name=engine.get_run_name(),
        hash=manager.hash_payload(payload),
    )
    return lookup.state


def test_cleanup_failed_claim_requires_claimed(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_clean", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/clean.json",
        payload=cpu_fit_payload(name="clean", run_group=[], seed=1),
    )
    engine = _fit_engine(
        root, project_name="proj_clean", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    with pytest.raises(RuntimeError, match="claimed"):
        engine.cleanup_failed_claim()


def test_cleanup_failed_claim_drops_pair_and_artifacts(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_drop", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/drop.json",
        payload=cpu_fit_payload(name="drop", run_group=[], seed=1),
    )
    engine = _fit_engine(
        root, project_name="proj_drop", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    engine.claim_fit_identity()
    artifacts = engine.get_artifacts_directory().path
    (artifacts / "stale.txt").write_text("stale\n", encoding="utf-8")
    assert engine._claimed
    engine.cleanup_failed_claim()
    assert not engine._claimed
    assert not engine._identity_ready
    assert not artifacts.exists()
    assert _pair_state(engine) == HashManager.BOTH_FREE
    retry = _fit_engine(
        root, project_name="proj_drop", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    retry.claim_fit_identity()
    assert retry._claimed
    assert retry.get_artifacts_directory().path.is_dir()


def test_failed_train_cleans_occupancy_and_logger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = scaffold_project(tmp_path, name="proj_fail", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/fail.json",
        payload=cpu_fit_payload(name="fail", run_group=[], seed=1),
    )
    engine = _fit_engine(
        root, project_name="proj_fail", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )

    def boom(*, logger, **kwargs):
        del kwargs
        logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        logger.close()
        raise RuntimeError("fit failed")

    monkeypatch.setattr(
        "dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine.train_val_fit",
        boom,
    )
    layout = layout_from_root(root, name="proj_fail", backend="tensorboard")
    run_dir = layout.tensorboard_log_dir().path / "fail"
    with pytest.raises(RuntimeError, match="fit failed"):
        engine.train()
    assert not engine._claimed
    assert not engine._identity_ready
    assert not engine.get_artifacts_directory().path.exists()
    assert not run_dir.exists()
    assert _pair_state(engine) == HashManager.BOTH_FREE
    retry = _fit_engine(
        root, project_name="proj_fail", recipe=recipe, force=IdentityOverwritePolicy.OFF
    )
    retry.claim_fit_identity()
    assert retry._claimed


def test_failed_find_cleans_occupancy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = scaffold_project(tmp_path, name="proj_find_fail", backend="tensorboard")
    recipe = write_recipe(
        root,
        relative_path="run_recipes/find_fail.json",
        payload=find_lr_payload(name="find_fail", run_group=[], seed=1),
    )
    engine = find_engine(
        root,
        project_name="proj_find_fail",
        recipe=recipe,
        force=IdentityOverwritePolicy.OFF,
    )

    def boom(*, logger, **kwargs):
        del kwargs
        logger.setup(stream=LogStream.FIT, existence=LogExistence.CREATE)
        logger.close()
        raise RuntimeError("find_lr failed")

    monkeypatch.setattr(
        "dl_toolbox.dl_engine.exp_engine.find_lr_engine.find_lr_engine.find_lr",
        boom,
    )
    layout = layout_from_root(root, name="proj_find_fail", backend="tensorboard")
    run_dir = layout.tensorboard_log_dir().path / "find_fail"
    with pytest.raises(RuntimeError, match="find_lr failed"):
        engine.find()
    assert not engine._claimed
    assert not engine.get_artifacts_directory().path.exists()
    assert not run_dir.exists()
    assert _pair_state(engine) == HashManager.BOTH_FREE
