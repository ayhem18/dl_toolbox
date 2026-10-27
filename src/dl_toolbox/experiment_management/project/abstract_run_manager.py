"""Project contract: manage_run / open_run assemble ExpSignatureObj.

This module has no filesystem layout. The opinionated tree lives on
``DefaultRunManager``.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dl_toolbox.base.hashing.hash_manager import (
    BOTH_FREE,
    BOTH_MATCH,
    HASH_TAKEN,
    NAME_TAKEN,
    HashManager,
)
from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.dl_engine.training.config import FitConfig
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager
from dl_toolbox.experiment_management.recipe.run_identity import RunIdentity
from dl_toolbox.experiment_management.recipe.run_recipe import RunRecipeParser
from dl_toolbox.experiment_management.signature.exp_signature_obj import ExpSignatureObj
from dl_toolbox.experiment_management.toolbox.manager import DlToolboxConfigManager


class AbstractRunManager(ABC):
    """What the rest of the toolbox needs from one project.

    Layout, logging JSON, and vendor store topology belong on a concrete
    class such as ``DefaultRunManager``.
    """

    def __init__(self, *, experiment_config_manager: ExperimentConfigManager) -> None:
        self._experiment_config_manager = experiment_config_manager

    @property
    @abstractmethod
    def project_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def project_root(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def project_runs_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def run_payload_dir(self, name: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def tracking_backend(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def make_logging_manager(self) -> object:
        """Return the store manager for ``tracking_backend()``."""
        raise NotImplementedError

    @abstractmethod
    def make_hash_manager(self) -> HashManager:
        raise NotImplementedError

    def reclaim_name(self, name: str) -> dict[str, Any] | None:
        """Drop the identity pair and the payload directory for ``name``.

        HashManager is the identity index. Payload is derived storage under
        ``run_payload_dir``. Logger stores are not identity and are left alone.
        Missing name with no payload returns ``None``.
        """
        hash_manager = self.make_hash_manager()
        released = hash_manager.release_name(self.project_runs_dir(), name=name)
        payload = self.run_payload_dir(hash_manager.assert_name(name))
        if payload.exists():
            shutil.rmtree(payload)
        return released

    def manage_run(
        self, recipe: Path | Mapping[str, Any], *, force: bool = False
    ) -> ExpSignatureObj:
        return self._assemble(recipe, claim=True, force=force)

    def open_run(self, recipe: Path | Mapping[str, Any]) -> ExpSignatureObj:
        return self._assemble(recipe, claim=False, force=False)

    def _recipe_mapping(self, recipe: Path | Mapping[str, Any]) -> dict:
        if isinstance(recipe, Path):
            return read_json(recipe)
        return parse_mapping(recipe, label="recipe")

    def _assemble(
        self,
        recipe: Path | Mapping[str, Any],
        *,
        claim: bool,
        force: bool,
    ) -> ExpSignatureObj:
        hash_manager = self.make_hash_manager()
        logging_manager = self.make_logging_manager()
        parsed_recipe = RunRecipeParser(
            self._recipe_mapping(recipe),
            hash_manager=hash_manager,
        )
        toolbox = DlToolboxConfigManager(
            parsed_recipe.dl_toolbox_config,
            backend=self.tracking_backend(),
            logging_manager=logging_manager,
        )
        toolbox.apply_seed()
        configuration_hash = hash_manager.hash_payload(
            {
                "dl_toolbox_config": parsed_recipe.dl_toolbox_config,
                "experiment_config": parsed_recipe.experiment_config,
            }
        )
        name = parsed_recipe.human_readable_name
        records = self.project_runs_dir()
        payload = self.run_payload_dir(name)
        lookup = hash_manager.lookup_name_hash_json(
            records, name=name, configuration_hash=configuration_hash
        )
        if claim:
            if lookup.state == HASH_TAKEN:
                occupant = read_json(lookup.hash_path)
                raise FileExistsError(
                    f"configuration hash already saved as name {occupant.get('name')!r}"
                )
            if force:
                self.reclaim_name(name)
            elif lookup.state == BOTH_FREE and payload.exists():
                raise FileExistsError(f"payload directory already exists: {payload}")
            hash_manager.write_name_hash_json_unique(
                records, name=name, configuration_hash=configuration_hash
            )
            payload.mkdir(parents=True, exist_ok=False)
        else:
            if lookup.state == BOTH_FREE:
                raise FileNotFoundError(f"run {name!r} has not been claimed")
            if lookup.state != BOTH_MATCH:
                if lookup.state == NAME_TAKEN:
                    raise FileExistsError(
                        f"name {name!r} exists with a different configuration hash"
                    )
                raise FileExistsError(
                    f"configuration hash already saved under a different name"
                )
            if not payload.is_dir():
                raise FileNotFoundError(f"payload directory does not exist: {payload}")
        experiment = self._experiment_config_manager.build(
            parsed_recipe.experiment_config,
            project_root=self.project_root(),
        )
        checkpoint_manager = toolbox.materialize_checkpoint(
            run_dir=payload,
            monitor=experiment.monitor,
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
        return ExpSignatureObj(
            identity=identity,
            fit_config=fit_config,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            seed=toolbox.seed_config(),
            experiment=experiment,
        )
