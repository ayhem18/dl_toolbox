"""Engine core: roles + orchestration. No project-tree discovery."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import lightning as L

from dl_toolbox.base.file_management.exclusive_lock import ExclusiveDirLock
from dl_toolbox.base.hashing.hash_manager import (
    BOTH_FREE,
    BOTH_MATCH,
    HASH_TAKEN,
    NAME_TAKEN,
)
from dl_toolbox.base.hashing.identity_manager import IdentityManager
from dl_toolbox.base.input_output.json_io import read_json, write_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager
from dl_toolbox.dl_engine.training.checkpointer.checkpointer import Checkpointer
from dl_toolbox.dl_engine.training.checkpointer.manager import CheckpointManager
from dl_toolbox.dl_engine.training.checkpointer.types import (
    CheckpointRequest,
    FitRunState,
)
from dl_toolbox.dl_engine.training.config import FitConfig
from dl_toolbox.dl_engine.training.fit import fit_classification
from dl_toolbox.experiment_management.engine.core.inspection import (
    InspectionManager,
    InspectFn,
)
from dl_toolbox.experiment_management.engine.core.paths import EnginePaths
from dl_toolbox.experiment_management.engine.core.session import RunSession
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager
from dl_toolbox.experiment_management.recipe.run_identity import RunIdentity
from dl_toolbox.experiment_management.recipe.run_recipe import RunRecipeParser

INSPECTION_RUNS_DIRNAME = "inspection_runs"
INSPECTION_STAGING_DIRNAME = "inspection_staging"
_CLAIM_LOCK_TIMEOUT = 30.0
_INSPECTION_KEYS = ("selector",)


class ExpEngine:
    def __init__(
        self,
        *,
        identity: IdentityManager,
        logging: LoggingManager,
        checkpoint: CheckpointManager,
        experiment: ExperimentConfigManager,
        paths: EnginePaths,
        compose_toolbox: Callable[..., Any],
        tracking_backend: str,
        inspection: InspectionManager | None = None,
    ) -> None:
        self._identity = identity
        self._logging = logging
        self._checkpoint = checkpoint
        self._experiment = experiment
        self._paths = paths
        self._compose_toolbox = compose_toolbox
        self._tracking_backend = tracking_backend
        self._inspection = inspection

    def reclaim_name(self, name: str) -> dict[str, Any] | None:
        released = self._identity.release_name(self._paths.records_dir, name=name)
        payload = self._paths.payload_dir(self._identity.assert_name(name))
        if payload.exists():
            shutil.rmtree(payload)
        return released

    def claim(
        self, recipe: Path | Mapping[str, Any], *, force: bool = False
    ) -> RunSession:
        return self._assemble(recipe, claim=True, force=force)

    def open(self, recipe: Path | Mapping[str, Any]) -> RunSession:
        return self._assemble(recipe, claim=False, force=False)

    def train(
        self, recipe: Path | Mapping[str, Any], *, force: bool = False
    ) -> L.Trainer:
        session = self.claim(recipe, force=force)
        try:
            return _fit(session)
        finally:
            session.checkpointer.mark_finished()

    def inspect(
        self,
        recipe: Path | Mapping[str, Any],
        inspection_recipe: Path | Mapping[str, Any],
        *,
        inspect_fn: InspectFn | None = None,
    ) -> Path:
        session = self.open(recipe)
        payload = self._inspection_mapping(inspection_recipe)
        require_exact_keys(payload, _INSPECTION_KEYS, label="inspection_recipe")
        request = CheckpointRequest.parse(str(payload["selector"]))
        fn = inspect_fn
        if fn is None:
            if self._inspection is None:
                raise ValueError("inspect requires InspectionManager or inspect_fn")
            fn = self._inspection.inspect_fn(payload)
        return _inspect_once(session, request=request, inspect_fn=fn)

    def fit_state(self, session: RunSession) -> FitRunState:
        return session.checkpointer.fit_state()

    def _recipe_mapping(self, recipe: Path | Mapping[str, Any]) -> dict:
        if isinstance(recipe, Path):
            return read_json(recipe)
        return parse_mapping(recipe, label="recipe")

    def _inspection_mapping(
        self, inspection_recipe: Path | Mapping[str, Any]
    ) -> dict:
        if isinstance(inspection_recipe, Path):
            return read_json(inspection_recipe)
        return parse_mapping(inspection_recipe, label="inspection_recipe")

    def _assemble(
        self,
        recipe: Path | Mapping[str, Any],
        *,
        claim: bool,
        force: bool,
    ) -> RunSession:
        parsed_recipe = RunRecipeParser(
            self._recipe_mapping(recipe),
            hash_manager=self._identity,
        )
        toolbox = self._compose_toolbox(
            parsed_recipe.dl_toolbox_config,
            backend=self._tracking_backend,
            logging_manager=self._logging,
        )
        toolbox.apply_seed()
        configuration_hash = self._identity.hash_payload(
            {
                "dl_toolbox_config": parsed_recipe.dl_toolbox_config,
                "experiment_config": parsed_recipe.experiment_config,
            }
        )
        name = parsed_recipe.human_readable_name
        records = self._paths.records_dir
        payload = self._paths.payload_dir(name)
        if claim:
            self._claim_payload(
                name=name,
                configuration_hash=configuration_hash,
                payload=payload,
                force=force,
            )
        else:
            self._open_payload(
                name=name,
                configuration_hash=configuration_hash,
                payload=payload,
            )
        experiment = self._experiment.build(
            parsed_recipe.experiment_config,
            project_root=self._paths.project_root,
        )
        checkpointer = Checkpointer(
            binding=self._checkpoint.bind(
                run_dir=payload,
                policy=toolbox.checkpoint_policy(),
                monitor=experiment.monitor,
                max_epochs=toolbox.trainer_config().max_epochs,
            )
        )
        logger = toolbox.build_logger()
        fit_config = FitConfig(
            trainer=toolbox.trainer_config(),
            monitor=experiment.monitor,
            early_stopping=toolbox.early_stopping_config(),
        )
        identity = RunIdentity(
            run_dir=payload,
            human_readable_name=name,
            configuration_hash=configuration_hash,
        )
        return RunSession(
            identity=identity,
            fit_config=fit_config,
            checkpointer=checkpointer,
            logger=logger,
            logging_manager=self._logging,
            seed=toolbox.seed_config(),
            experiment=experiment,
        )

    def _claim_payload(
        self,
        *,
        name: str,
        configuration_hash: str,
        payload: Path,
        force: bool,
    ) -> None:
        records = self._paths.records_dir
        lock = ExclusiveDirLock(
            records / "claim.lock",
            timeout_seconds=_CLAIM_LOCK_TIMEOUT,
        )
        with lock:
            lookup = self._identity.lookup(
                records, name=name, configuration_hash=configuration_hash
            )
            if lookup.state == HASH_TAKEN:
                occupant = read_json(lookup.hash_path)
                raise FileExistsError(
                    f"configuration hash already saved as name {occupant.get('name')!r}"
                )
            if force:
                self.reclaim_name(name)
            elif lookup.state == BOTH_FREE and payload.exists():
                raise FileExistsError(f"payload directory already exists: {payload}")
            self._identity.claim(
                records, name=name, configuration_hash=configuration_hash
            )
            payload.mkdir(parents=True, exist_ok=False)

    def _open_payload(
        self,
        *,
        name: str,
        configuration_hash: str,
        payload: Path,
    ) -> None:
        records = self._paths.records_dir
        lookup = self._identity.lookup(
            records, name=name, configuration_hash=configuration_hash
        )
        if lookup.state == BOTH_FREE:
            raise FileNotFoundError(f"run {name!r} has not been claimed")
        if lookup.state != BOTH_MATCH:
            if lookup.state == NAME_TAKEN:
                raise FileExistsError(
                    f"name {name!r} exists with a different configuration hash"
                )
            raise FileExistsError(
                "configuration hash already saved under a different name"
            )
        if not payload.is_dir():
            raise FileNotFoundError(f"payload directory does not exist: {payload}")


def _fit(session: RunSession) -> L.Trainer:
    experiment = session.experiment
    return fit_classification(
        experiment.model,
        experiment.train_loader,
        experiment.val_loader,
        experiment.optimizer,
        prepare_tensors_fn=experiment.prepare_tensors_fn,
        forward_fn=experiment.forward_fn,
        loss_fn=experiment.loss_fn,
        build_metric_inputs_fn=experiment.build_metric_inputs_fn,
        train_metrics=experiment.train_metrics,
        val_metrics=experiment.val_metrics,
        config=session.fit_config,
        checkpoint_manager=session.checkpointer,
        logger=session.logger,
    )


def _inspect_once(
    session: RunSession,
    *,
    request: CheckpointRequest,
    inspect_fn: InspectFn,
) -> Path:
    staging_root = session.run_dir / INSPECTION_STAGING_DIRNAME
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"pending-{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        record = session.checkpointer.fetch(request, dest=staging)
        folder = f"epoch={record.epoch}"
        dest = session.run_dir / INSPECTION_RUNS_DIRNAME / folder
        if dest.exists():
            raise FileExistsError(f"inspection folder already exists: {dest}")
        write_json(
            staging / "manifest.json",
            {
                "selector_kind": request.kind,
                "epoch": record.epoch,
                "checkpoint": record.path.name,
                "kind": record.kind,
                "rank": record.rank,
                "score": record.score,
            },
        )
        inspect_fn(session, record, staging)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(dest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    session.logging_manager.publish_directory(
        dest,
        dest=f"inspection/{folder}",
        logger=session.logger,
    )
    return dest
