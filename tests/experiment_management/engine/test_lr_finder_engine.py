from __future__ import annotations

import random
from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.dl_engine.lr_finder import LrFinderError, LrFinderResult
from dl_toolbox.dl_engine.training.checkpointer.types import FIT_FINISHED
from dl_toolbox.experiment_management.default_layout.layout.names import (
    BACKEND_PLACEHOLDER,
)
from dl_toolbox.experiment_management.default_layout.layout.scaffold import setup_project
from dl_toolbox.experiment_management.engine.lr_finder.engine import (
    LrFinderExpEngine,
    LrFinderFullDsExpEngine,
)
from tests.experiment_management.em_support import (
    LrFinderTinyExperimentConfigManager,
    placeholder_recipe,
)
from tests.experiment_management.names import random_project_name


def _scaffold(tmp_path: Path, rng: random.Random) -> tuple[str, Path, Path]:
    parent = tmp_path / "parent"
    parent.mkdir()
    name = random_project_name(rng)
    root = setup_project(name, parent=parent, tracking_backend=BACKEND_PLACEHOLDER)
    start = root / "scripts" / "find_lr.py"
    start.write_text("x = 1\n", encoding="utf-8")
    return name, root, start


def _engine(name: str, start: Path) -> LrFinderExpEngine:
    return LrFinderExpEngine(
        name,
        start=start,
        experiment=LrFinderTinyExperimentConfigManager(),
    )


def _bowl_curve() -> tuple[tuple[float, ...], tuple[float, ...]]:
    n = 11
    lrs: list[float] = [0.0] * n
    losses: list[float] = [0.0] * n
    for i in range(n):
        lrs[i] = 10 ** (-6 + i * 0.5)
        losses[i] = float((i - 6) * (i - 6))
    return tuple(lrs), tuple(losses)


def test_train_writes_curve_and_marks_finished(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    lrs, losses = _bowl_curve()
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.lr_finder.find.run_range_test",
        lambda inputs: (lrs, losses),
    )
    recipe = placeholder_recipe(name="run_finder", seed=0, run_name="finder_run")
    result = engine.train(recipe)
    assert isinstance(result, LrFinderResult)
    assert result.curve_path.is_file()
    payload = read_json(result.curve_path)
    assert payload["suggested"]["min_loss_lr"] == lrs[6]
    session = engine.get_existing_run(recipe)
    state = engine.fit_state(session)
    assert state.status == FIT_FINISHED
    resolved = read_json(session.run_dir / "resolved_config.json")
    assert "lr_finder" in resolved


def test_full_ds_engine_writes_min_loss_lr(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    engine = LrFinderFullDsExpEngine(
        name,
        start=start,
        experiment=LrFinderTinyExperimentConfigManager(),
    )
    lrs, losses = _bowl_curve()
    monkeypatch.setattr(
        "dl_toolbox.dl_engine.lr_finder.find.run_full_ds_range_test",
        lambda inputs: (lrs, losses),
    )
    recipe = placeholder_recipe(name="run_finder_full_ds", seed=0, run_name="finder_ds")
    result = engine.train(recipe)
    assert isinstance(result, LrFinderResult)
    payload = read_json(result.curve_path)
    assert payload["suggested"] == {"min_loss_lr": lrs[6]}


def test_refuses_distributed_worker(
    tmp_path: Path, rng: random.Random, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    recipe = placeholder_recipe(name="run_finder", seed=0, run_name="finder_run")
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(LrFinderError, match="single-process"):
        engine.train(recipe)


def test_refuses_devices_not_one(
    tmp_path: Path, rng: random.Random
) -> None:
    name, _root, start = _scaffold(tmp_path, rng)
    engine = _engine(name, start)
    recipe = placeholder_recipe(name="run_finder", seed=0, run_name="finder_run")
    recipe["dl_toolbox_config"]["trainer_config"]["devices"] = 4
    with pytest.raises(LrFinderError, match="devices == 1"):
        engine.train(recipe)
