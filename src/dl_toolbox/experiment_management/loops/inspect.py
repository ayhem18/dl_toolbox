"""Open a claimed run, look up one checkpoint, write exclusive epoch folder."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.dl_engine.loggers.mlflow.mlflow_manager import MlflowManager
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.loggers.tensorboard.tensorboard_manager import TensorBoardManager
from dl_toolbox.dl_engine.training.checkpoint import CheckpointLookup
from dl_toolbox.experiment_management.signature.exp_signature_obj import ExpSignatureObj

INSPECTION_RUNS_DIRNAME = "inspection_runs"
INSPECTION_STAGING_DIRNAME = "inspection_staging"


def inspect_experiment(
    signature: ExpSignatureObj,
    *,
    selector: str,
    inspect_fn: Callable[[ExpSignatureObj, CheckpointLookup, Path], None],
) -> Path:
    lookup = signature.checkpoint_manager.lookup(selector)
    folder = f"epoch={lookup.epoch}"
    dest = signature.run_dir / INSPECTION_RUNS_DIRNAME / folder
    staging = signature.run_dir / INSPECTION_STAGING_DIRNAME / folder
    if dest.exists():
        raise FileExistsError(f"inspection folder already exists: {dest}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(lookup.path, staging / lookup.path.name)
        write_json(
            staging / "manifest.json",
            {
                "selector": selector,
                "epoch": lookup.epoch,
                "checkpoint": lookup.path.name,
                "kind": lookup.kind,
                "rank": lookup.rank,
            },
        )
        inspect_fn(signature, lookup, staging)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(dest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _publish_inspection(signature, dest, folder=folder)
    return dest


def _publish_inspection(
    signature: ExpSignatureObj,
    dest: Path,
    *,
    folder: str,
) -> None:
    logger = signature.logger
    manager = getattr(logger, "_manager", None)
    if manager is None:
        return
    run_name = getattr(logger, "run_name", None)
    if isinstance(manager, (PlaceholderManager, TensorBoardManager)):
        if not run_name:
            raise ValueError("filesystem logger is missing run_name")
        target = manager.log_dir / run_name / "inspection" / folder
        if target.exists():
            raise FileExistsError(f"inspection vendor folder already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dest, target)
        return
    if isinstance(manager, MlflowManager):
        manager.log_directory_to_run(dest, artifact_path=f"inspection/{folder}")
