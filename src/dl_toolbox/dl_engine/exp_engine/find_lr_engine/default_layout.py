"""DefaultLayoutFindLrEngine fills FindLrEngine getters from a ProjectLayout."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.experiment import FindLrParts
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.find_lr_engine import FindLrEngine
from dl_toolbox.dl_engine.exp_engine.find_lr_engine.recipe import parse_find_lr_recipe
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    parse_force,
)
from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.loggers.mlflow import MlflowLogger
from dl_toolbox.dl_engine.loggers.tensorboard import TensorBoardLogger
from dl_toolbox.dl_engine.runs.common.run_config import RunLogConfig
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrRuntime,
    FindLrSpec,
    FindLrSteps,
)
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend


class DefaultLayoutFindLrEngine(FindLrEngine):
    """FindLrEngine filled from a ProjectLayout plus a find-lr recipe File."""

    RESOLVED_CONFIG_FILENAME = "resolved_config.json"
    CONSTRUCTOR_KEYS = ("force",)

    def __init__(
        self,
        *,
        layout: ProjectLayout,
        recipe: File,
        experiment: object,
        **fields: object,
    ) -> None:
        super().__init__()
        if not isinstance(layout, ProjectLayout):
            raise TypeError(f"layout must be ProjectLayout, got {type(layout)!r}")
        if not isinstance(recipe, File):
            raise TypeError(f"recipe must be File, got {type(recipe)!r}")
        self._layout = layout
        self._recipe_file = recipe
        self._recipe = parse_find_lr_recipe(
            read_object(file=recipe),
            tracking_backend=layout.tracking_backend,
            hash_manager=layout.hash_manager,
        )
        self._experiment = _bind_experiment(experiment)
        require_exact_keys(
            fields,
            self.CONSTRUCTOR_KEYS,
            label="DefaultLayoutFindLrEngine",
        )
        self._force = parse_force(fields["force"])
        self._parts: FindLrParts | None = None

    def get_run_name(self) -> str:
        return self._recipe.human_readable_name

    def get_identity_payload(self) -> dict:
        return self._recipe.identity_payload()

    def get_force(self) -> str:
        return self._force

    def get_hash_manager(self) -> HashManager:
        return self._layout.hash_manager

    def get_identity_index(self) -> Directory:
        return self._layout.identity_index(run_group=self._recipe.run_group)

    def get_identity_lock_timeout(self) -> float:
        return self._layout.lock_timeout_seconds

    def get_artifacts_directory(self) -> Directory:
        return self._layout.artifacts_directory(
            run_group=self._recipe.run_group,
            name=self.get_run_name(),
        )

    def apply_seed(self) -> None:
        seed = self._recipe.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_net(self) -> nn.Module:
        return self._require_parts().net

    def get_optimizer(self) -> Optimizer:
        return self._require_parts().optimizer

    def get_train_loader(self) -> DataLoader:
        return self._require_parts().train_loader

    def get_steps(self) -> FindLrSteps:
        return self._require_parts().steps

    def get_spec(self) -> FindLrSpec:
        return self._recipe.spec

    def get_runtime(self) -> FindLrRuntime:
        return self._recipe.runtime

    def get_logger(self) -> Logger:
        self.require_identity()
        name = self.get_run_name()
        backend = self._layout.tracking_backend
        if backend == TrackingBackend.TENSORBOARD:
            return TensorBoardLogger(
                log_dir=self._layout.tensorboard_log_dir().path,
                run_name=name,
            )
        experiment = self._recipe.mlflow_experiment
        if experiment is None:
            raise ValueError("mlflow logger.experiment is required")
        db_path = self._layout.mlflow_db_path()
        timeout = self._layout.mlflow_timeout_seconds()
        return MlflowLogger(
            backend_store_uri=(
                f"sqlite:///{db_path.as_posix()}?timeout={timeout}"
            ),
            artifact_store_uri=self._layout.mlflow_artifact_root().path.as_posix(),
            experiment_name=experiment,
            run_name=name,
            tags=self._recipe.tags,
        )

    def get_run_log(self) -> RunLogConfig:
        self.require_identity()
        payload: dict[str, Any] = dict(self.get_identity_payload())
        payload["human_readable_name"] = self.get_run_name()
        payload["tags"] = dict(self._recipe.tags)
        experiment = self._recipe.mlflow_experiment
        if experiment is not None:
            payload["experiment"] = experiment
        return RunLogConfig(
            payload=payload, filename=self.RESOLVED_CONFIG_FILENAME
        )

    def _require_parts(self) -> FindLrParts:
        self.require_identity()
        if self._parts is None:
            parts = self._experiment.build_find_lr(
                self._recipe.experiment_config,
                project_root=self._layout.root,
            )
            if not isinstance(parts, FindLrParts):
                raise TypeError(
                    "experiment.build_find_lr must return FindLrParts, "
                    f"got {type(parts)!r}"
                )
            self._parts = parts
        return self._parts


def _bind_experiment(experiment: object) -> object:
    if isinstance(experiment, type):
        bound = experiment()
    else:
        bound = experiment
    if not callable(getattr(bound, "build_find_lr", None)):
        raise TypeError("experiment must have build_find_lr")
    return bound
