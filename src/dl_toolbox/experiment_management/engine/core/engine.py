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
from dl_toolbox.dl_engine.loggers.base_logger import STREAM_INSPECT
from dl_toolbox.dl_engine.loggers.base_manager import LoggingManager
from dl_toolbox.dl_engine.training.checkpointer.checkpointer import Checkpointer
from dl_toolbox.dl_engine.training.checkpointer.manager import CheckpointManager
from dl_toolbox.dl_engine.training.checkpointer.types import (
    CheckpointRequest,
    FitRunState,
)
from dl_toolbox.dl_engine.training.config import FitConfig
from dl_toolbox.dl_engine.training.fit import fit_classification
from dl_toolbox.experiment_management.engine.core.inspection import InspectFn
from dl_toolbox.experiment_management.engine.core.paths import EnginePaths
from dl_toolbox.experiment_management.engine.core.session import RunSession
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager
from dl_toolbox.experiment_management.recipe.run_identity import RunIdentity
from dl_toolbox.experiment_management.recipe.run_recipe import RunRecipeParser

INSPECTION_RUNS_DIRNAME = "inspection_runs"
INSPECTION_STAGING_DIRNAME = "inspection_staging"
_IDENTITY_LOCK_TIMEOUT = 30.0
_INSPECTION_KEYS = ("selector",)


class ExpEngine:
    def __init__(
        self,
        *,
        identity_manager: IdentityManager,
        logging_manager: LoggingManager,
        checkpoint_manager: CheckpointManager,
        experiment_manager: ExperimentConfigManager,
        paths: EnginePaths,
        compose_toolbox: Callable[..., Any], # the exact type must be specified here ... 
        tracking_backend: str,
    ) -> None:

        """
        1. identity_manager: the abstraction that ensures every (name, hash) pair is unique
        2. logging_manager: the abstraction that ensures the logging backend is selected / configured properly
        3. checkpoint_manager: the abstraction that ensures the chechkpoint policy is implemented properly
        4. experiment_manager: the abstraction that extracts the project-specific configuration from the recipe: extracts the callables used in fit
        5. paths: A wrapper around the 3 type of paths an experiment touches / uses
        6. compose_toolbox: the abstraction that ensures the toolbox is composed properly
        7. tracking_backend: TODO: explain what this is: is the same as the logging backend ? 
        """

        self._identity_manager = identity_manager
        self._logging_manager = logging_manager
        self._checkpoint_manager = checkpoint_manager
        self._experiment_manager = experiment_manager
        self._paths = paths 
        self._compose_toolbox = compose_toolbox
        self._tracking_backend = tracking_backend

    ################################## get_new_run_or_overwrite / get_existing_run ##################################

    def get_new_run_or_overwrite(
        self, recipe: Path | Mapping[str, Any], *, force: bool = False
    ) -> RunSession:
        """Create a run for this recipe, or overwrite this name if force=True.

        CONTRACT. Source of truth:
        standards/exp_mng_engine/identity/get_new_run_or_overwrite.md
        Changing branches below changes that contract.

        Identity is (human_readable_name, hash({dl_toolbox_config,
        experiment_config})). Catalog status is not consulted. Evaluate in
        this order:

        1. hash_taken — FileExistsError. force ignored.
        2. both_match — FileExistsError. force ignored. Resume is a later path.
        3. both_free — if run_artifacts/{name} exists, FileExistsError
           (orphan folder, even with force). Else write the pair and mkdir.
        4. name_taken — force=False: FileExistsError. force=True: reclaim
           this name, then create. Logger stores are not touched.

        train() calls this. inspect() does not.
        """
        return self._assemble(recipe, create=True, force=force)

    def get_existing_run(self, recipe: Path | Mapping[str, Any]) -> RunSession:
        """Attach to a run that already exists. Creates nothing. force does not apply.

        CONTRACT. Requires both_match (same name AND same hash) and
        run_artifacts/{name} on disk.

        - both_free → FileNotFoundError (run does not exist)
        - name_taken → FileExistsError (same name, different hash)
        - hash_taken → FileExistsError (same hash, different name)
        - both_match but artifacts dir missing → FileNotFoundError

        inspect() calls this. train() does not.
        """
        return self._assemble(recipe, create=False, force=False)

    def reclaim_name(self, name: str) -> dict[str, Any] | None:
        """Overwrite helper: drop this name's identity pair and its artifacts folder.

        IdentityManager.release_name deletes the JSON pair under run_identities_root.
        Then rmtree run_artifacts/{name} if it exists. Logger stores are not touched.
        Missing name returns None. Used only by name_taken + force=True.
        """
        released = self._identity_manager.release_name(
            self._paths.run_identities_root, name=name
        )
        run_artifacts = self._paths.run_artifacts_dir(
            self._identity_manager.assert_name(name)
        )
        if run_artifacts.exists():
            shutil.rmtree(run_artifacts)
        return released

    def _create_or_overwrite_run(
        self,
        *,
        name: str,
        configuration_hash: str,
        run_artifacts: Path,
        force: bool,
    ) -> None:
        # Implements get_new_run_or_overwrite's decision tree. Keep in lockstep
        # with that docstring.
        run_identities_root = self._paths.run_identities_root
        lock = ExclusiveDirLock(
            run_identities_root / "identity.lock",
            timeout_seconds=_IDENTITY_LOCK_TIMEOUT,
        )
        with lock:
            lookup = self._identity_manager.lookup(
                run_identities_root, name=name, configuration_hash=configuration_hash
            )
            if lookup.state == HASH_TAKEN:
                occupant = read_json(lookup.hash_path)
                raise FileExistsError(
                    f"configuration hash already saved as name {occupant.get('name')!r}"
                )
            if lookup.state == BOTH_MATCH:
                raise FileExistsError(
                    f"run {name!r} already exists with this configuration hash"
                )
            if lookup.state == BOTH_FREE:
                if run_artifacts.exists():
                    raise FileExistsError(
                        f"run artifacts directory already exists: {run_artifacts}"
                    )
            elif lookup.state == NAME_TAKEN:
                if not force:
                    raise FileExistsError(
                        f"name {name!r} exists with a different configuration hash"
                    )
                self.reclaim_name(name)
            else:
                raise ValueError(
                    f"unexpected identity lookup state {lookup.state!r}"
                )
            self._identity_manager.claim(
                run_identities_root, name=name, configuration_hash=configuration_hash
            )
            run_artifacts.mkdir(parents=True, exist_ok=False)

    def _require_existing_run(
        self,
        *,
        name: str,
        configuration_hash: str,
        run_artifacts: Path,
    ) -> None:
        # Implements get_existing_run's contract. Keep in lockstep with that docstring.
        run_identities_root = self._paths.run_identities_root
        lookup = self._identity_manager.lookup(
            run_identities_root, name=name, configuration_hash=configuration_hash
        )
        if lookup.state == BOTH_FREE:
            raise FileNotFoundError(f"run {name!r} does not exist")
        if lookup.state != BOTH_MATCH:
            if lookup.state == NAME_TAKEN:
                raise FileExistsError(
                    f"name {name!r} exists with a different configuration hash"
                )
            raise FileExistsError(
                "configuration hash already saved under a different name"
            )
        if not run_artifacts.is_dir():
            raise FileNotFoundError(
                f"run artifacts directory does not exist: {run_artifacts}"
            )

    ################################## train / inspect ##################################

    def train(
        self, recipe: Path | Mapping[str, Any], *, force: bool = False
    ) -> L.Trainer:
        session = self.get_new_run_or_overwrite(recipe, force=force)
        try:
            return _fit(session)
        finally:
            session.checkpointer.mark_finished()

    def inspect(
        self,
        experiment_recipe: Path | Mapping[str, Any],
        inspection_recipe: Path | Mapping[str, Any],
        *,
        inspect_fn: InspectFn,
    ) -> Path:

        print(
            f"inspect: opening run and building experiment for {experiment_recipe}",
            flush=True,
        )
        run_session = self.get_existing_run(experiment_recipe)

        inspection_recipe_as_dict = self._inspection_mapping(inspection_recipe)

        require_exact_keys(inspection_recipe_as_dict, _INSPECTION_KEYS, label="inspection_recipe")

        request = CheckpointRequest.parse(str(inspection_recipe_as_dict["selector"]))

        return _inspect_once(run_session, request=request, inspect_fn=inspect_fn)

    def fit_state(self, session: RunSession) -> FitRunState:
        return session.checkpointer.fit_state()

    ################################## session assembly ##################################

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
        create: bool,
        force: bool,
    ) -> RunSession:
        """
        1. parse the recipe configuration
        2. compose tool box: this should not be the case, the return type is unspecified ... this can very easily break 
        3. toolbox.apply_seed(): we are just assuming apply_seed() exists 
        4. hash the experiment configuration: beautiful 
        5. Now, there are 2 things: _require_existing_run() and _create_or_overwrite_run()
        """
    
        parsed_recipe = RunRecipeParser(
            self._recipe_mapping(recipe),
            hash_manager=self._identity_manager,
        )

        toolbox = self._compose_toolbox(
            parsed_recipe.dl_toolbox_config,
            backend=self._tracking_backend,
            logging_manager=self._logging_manager,
        )

        toolbox.apply_seed() # we are just assuming apply_seed() exists

        configuration_hash = self._identity_manager.hash_payload(
            {
                "dl_toolbox_config": parsed_recipe.dl_toolbox_config,
                "experiment_config": parsed_recipe.experiment_config,
            }
        )

        name = parsed_recipe.human_readable_name
        run_artifacts = self._paths.run_artifacts_dir(name)
 
        if create:
            self._create_or_overwrite_run(
                name=name,
                configuration_hash=configuration_hash,
                run_artifacts=run_artifacts,
                force=force,
            )

        else:
            self._require_existing_run(
                name=name,
                configuration_hash=configuration_hash,
                run_artifacts=run_artifacts,
            )

        experiment = self._experiment_manager.build(
            parsed_recipe.experiment_config,
            project_root=self._paths.project_root,
        )

        checkpointer = Checkpointer(
            binding=self._checkpoint_manager.bind(
                run_dir=run_artifacts,
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
            run_dir=run_artifacts,
            human_readable_name=name,
            configuration_hash=configuration_hash,
        )

        return RunSession(
            identity=identity,
            fit_config=fit_config,
            checkpointer=checkpointer,
            logger=logger,
            logging_manager=self._logging_manager,
            seed=toolbox.seed_config(),
            experiment=experiment,
        )


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
        checkpointer=session.checkpointer,
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

    print(f"inspect: run {session.human_readable_name!r} fetch {request.kind}", flush=True)
    try:
        record = session.checkpointer.fetch(request, dest=staging)
        print(
            f"inspect: fetched epoch={record.epoch} rank={record.rank} "
            f"{record.path.name}",
            flush=True,
        )
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
        print(f"inspect: calling inspect_fn for epoch={record.epoch}", flush=True)
        session.logger.setup(stream=STREAM_INSPECT)
        try:
            inspect_fn(session, record, staging)
            print(f"inspect: inspect_fn finished epoch={record.epoch}", flush=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(dest)
            print(f"inspect: committed {dest}", flush=True)
        finally:
            session.logger.close()
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return dest
