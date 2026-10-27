from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_logger import PlaceholderLogger
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.training.checkpoint import CheckpointLookup, CheckpointManager
from dl_toolbox.dl_engine.training.config import (
    EarlyStoppingConfig,
    FitConfig,
    MonitorConfig,
    SeedConfig,
    TrainerConfig,
)
from dl_toolbox.experiment_management.loops.inspect import inspect_experiment
from dl_toolbox.experiment_management.recipe.run_identity import RunIdentity
from dl_toolbox.experiment_management.signature.exp_signature_obj import ExpSignatureObj
from tests.dl_engine.training.test_checkpoint import _write_catalog
from tests.experiment_management.em_support import TinyExperimentConfigManager


def _signature(tmp_path: Path) -> ExpSignatureObj:
    run_dir = tmp_path / "payload" / "run_a"
    _write_catalog(run_dir)
    monitor = MonitorConfig(key="loss", mode="min")
    logging_manager = PlaceholderManager(log_dir=tmp_path / "logs")
    logger = PlaceholderLogger(manager=logging_manager, run_name="tiny_run")
    experiment = TinyExperimentConfigManager().build({}, project_root=tmp_path)
    return ExpSignatureObj(
        identity=RunIdentity(
            run_dir=run_dir,
            human_readable_name="run_a",
            configuration_hash="a" * 64,
        ),
        fit_config=FitConfig(
            trainer=TrainerConfig(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                precision=32,
                enable_progress_bar=False,
            ),
            monitor=monitor,
            early_stopping=EarlyStoppingConfig.none(),
        ),
        checkpoint_manager=CheckpointManager(
            save_top_k=1,
            save_last=True,
            run_dir=run_dir,
            monitor=monitor,
        ),
        logger=logger,
        seed=SeedConfig(seed=0),
        experiment=experiment,
    )


def _inspect_fn(
    signature: ExpSignatureObj,
    lookup: CheckpointLookup,
    staging: Path,
) -> None:
    del signature, lookup
    (staging / "note.txt").write_text("ok", encoding="utf-8")


def test_inspect_writes_exclusive_epoch_folder(tmp_path: Path) -> None:
    signature = _signature(tmp_path)
    dest = inspect_experiment(signature, selector="epoch=3", inspect_fn=_inspect_fn)
    assert dest == signature.run_dir / "inspection_runs" / "epoch=3"
    assert (dest / "note.txt").read_text(encoding="utf-8") == "ok"
    manifest = read_json(dest / "manifest.json")
    assert manifest["epoch"] == 3
    assert (dest / "epoch=003.ckpt").is_file()
    published = tmp_path / "logs" / "tiny_run" / "inspection" / "epoch=3" / "note.txt"
    assert published.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        inspect_experiment(signature, selector="rank=1", inspect_fn=_inspect_fn)
