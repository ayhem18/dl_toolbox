"""DefaultLayoutTrainOnlyEngine fills TrainOnlyEngine getters from a ProjectLayout.

Fit construction takes a train-recipe File, an experiment with build_train_only /
build_eval, and force. Inspect adds inspect_recipe File and eval_device.
GPU count picks the first N visible CUDA devices; a list is explicit indices.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CheckpointMonitor,
    CheckpointRequest,
)
from dl_toolbox.dl_engine.exp_engine.train_only_engine.experiment import (
    TrainOnlyEvalParts,
    TrainOnlyFitParts,
)
from dl_toolbox.dl_engine.exp_engine.train_only_engine.recipe import (
    parse_inspect_recipe,
    parse_train_only_recipe,
)
from dl_toolbox.dl_engine.exp_engine.train_only_engine.train_only_engine import (
    EnginePurpose,
    TrainOnlyEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    devices_and_strategy,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    parse_force,
)
from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.loggers.mlflow import MlflowLogger
from dl_toolbox.dl_engine.loggers.tensorboard import TensorBoardLogger
from dl_toolbox.dl_engine.runs.common.run_config import (
    FitSteps,
    OptimizationConfig,
    RunLogConfig,
    TrainerConfig,
)
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalContext,
    EvalLoaders,
    LogItem,
)
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import (
    TrainOnlyLoaders,
    TrainOnlyMetrics,
)
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend


class DefaultLayoutTrainOnlyEngine(TrainOnlyEngine):
    """TrainOnlyEngine filled from a ProjectLayout plus recipe Files."""

    RESOLVED_CONFIG_FILENAME = "resolved_config.json"
    INSPECTION_STAGING_DIRNAME = "inspection_staging"
    INSPECTION_RUNS_DIRNAME = "inspection_runs"
    FIT_CONSTRUCTOR_KEYS = ("force",)
    INSPECT_CONSTRUCTOR_KEYS = ("inspect_recipe", "eval_device")

    def __init__(
        self,
        *,
        purpose: str,
        layout: ProjectLayout,
        train_recipe: File,
        experiment: object,
        **fields: object,
    ) -> None:
        super().__init__(purpose=purpose)
        if not isinstance(layout, ProjectLayout):
            raise TypeError(f"layout must be ProjectLayout, got {type(layout)!r}")
        if not isinstance(train_recipe, File):
            raise TypeError(f"train_recipe must be File, got {type(train_recipe)!r}")
        self._layout = layout
        self._train_recipe_file = train_recipe
        self._train_recipe = parse_train_only_recipe(
            read_object(file=train_recipe),
            tracking_backend=layout.tracking_backend,
            hash_manager=layout.hash_manager,
        )
        self._experiment = _bind_experiment(experiment)
        self._fit_parts: TrainOnlyFitParts | None = None
        self._eval_parts: TrainOnlyEvalParts | None = None
        self._trainer_config: TrainerConfig | None = None
        self._payload_directory: Directory | None = None
        if self._purpose == EnginePurpose.FIT:
            require_exact_keys(
                fields,
                self.FIT_CONSTRUCTOR_KEYS,
                label="DefaultLayoutTrainOnlyEngine fit",
            )
            self._force = parse_force(fields["force"])
            return
        require_exact_keys(
            fields,
            self.INSPECT_CONSTRUCTOR_KEYS,
            label="DefaultLayoutTrainOnlyEngine inspect",
        )
        inspect_recipe = fields["inspect_recipe"]
        eval_device = fields["eval_device"]
        if not isinstance(inspect_recipe, File):
            raise TypeError(
                f"inspect_recipe must be File, got {type(inspect_recipe)!r}"
            )
        if not isinstance(eval_device, torch.device):
            raise TypeError(
                f"eval_device must be torch.device, got {type(eval_device)!r}"
            )
        self._inspect_recipe_file = inspect_recipe
        self._inspect_recipe = parse_inspect_recipe(read_object(file=inspect_recipe))
        self._eval_device = eval_device

    def get_run_name(self) -> str:
        return self._train_recipe.human_readable_name

    def get_identity_payload(self) -> dict:
        return self._train_recipe.identity_payload()

    def get_force(self) -> str:
        if self._purpose != EnginePurpose.FIT:
            raise ValueError("get_force is not a slot when purpose is inspect")
        return self._force

    def get_hash_manager(self) -> HashManager:
        return self._layout.hash_manager

    def get_identity_index(self) -> Directory:
        return self._layout.identity_index(run_group=self._train_recipe.run_group)

    def get_identity_lock_timeout(self) -> float:
        return self._layout.lock_timeout_seconds

    def get_artifacts_directory(self) -> Directory:
        return self._layout.artifacts_directory(
            run_group=self._train_recipe.run_group,
            name=self.get_run_name(),
        )

    def apply_seed(self) -> None:
        seed = self._train_recipe.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_net(self) -> nn.Module:
        return self._require_fit_parts().net

    def get_loaders(self) -> TrainOnlyLoaders:
        self._require_purpose(EnginePurpose.FIT)
        return self._require_fit_parts().loaders

    def get_steps(self) -> FitSteps:
        self._require_purpose(EnginePurpose.FIT)
        return self._require_fit_parts().steps

    def get_metrics(self) -> TrainOnlyMetrics:
        return self._require_fit_parts().metrics

    def get_optimization(self) -> OptimizationConfig:
        self._require_purpose(EnginePurpose.FIT)
        return self._require_fit_parts().optimization

    def get_trainer(self) -> TrainerConfig:
        if self._trainer_config is None:
            recipe = self._train_recipe
            devices, strategy = devices_and_strategy(
                accelerator=recipe.accelerator,
                devices_field=recipe.devices_field,
            )
            self._trainer_config = TrainerConfig(
                accelerator=recipe.accelerator,
                devices=devices,
                strategy=strategy,
                precision=recipe.precision,
                max_epochs=recipe.max_epochs,
                accumulate_grad_batches=recipe.accumulate_grad_batches,
                enable_progress_bar=recipe.enable_progress_bar,
                early_stopping=recipe.early_stopping,
            )
        return self._trainer_config

    def get_checkpointer(self) -> InspectableCheckpointer:
        self.require_identity()
        artifacts = self._existing_artifacts()
        metrics = self.get_metrics()
        trainer = self.get_trainer()
        return InspectableCheckpointer(
            run_artifacts_dir=artifacts.path,
            policy=self._train_recipe.checkpoint_policy,
            monitor=CheckpointMonitor(
                key=metrics.monitor.key, mode=metrics.monitor.mode
            ),
            max_epochs=trainer.max_epochs,
            lock_timeout_seconds=self.get_identity_lock_timeout(),
            lock_poll_seconds=self.get_hash_manager().lock_poll_seconds,
        )

    def get_logger(self) -> Logger:
        self.require_identity()
        name = self.get_run_name()
        backend = self._layout.tracking_backend
        if backend == TrackingBackend.TENSORBOARD:
            return TensorBoardLogger(
                log_dir=self._layout.tensorboard_log_dir().path,
                run_name=name,
            )
        experiment = self._train_recipe.mlflow_experiment
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
            tags=self._train_recipe.tags,
        )

    def get_run_log(self) -> RunLogConfig:
        self.require_identity()
        payload: dict[str, Any] = dict(self.get_identity_payload())
        payload["human_readable_name"] = self.get_run_name()
        payload["tags"] = dict(self._train_recipe.tags)
        experiment = self._train_recipe.mlflow_experiment
        if experiment is not None:
            payload["experiment"] = experiment
        return RunLogConfig(
            payload=payload, filename=self.RESOLVED_CONFIG_FILENAME
        )

    def get_checkpoint_request(self) -> CheckpointRequest:
        self._require_purpose(EnginePurpose.INSPECT)
        self.require_identity()
        return CheckpointRequest.parse(self._inspect_recipe.selector)

    def get_inspect_staging_directory(self) -> Directory:
        self._require_purpose(EnginePurpose.INSPECT)
        self.require_identity()
        token = uuid.uuid4().hex
        return Directory.from_under(
            root=self._existing_artifacts(),
            relative_path=f"{self.INSPECTION_STAGING_DIRNAME}/{token}",
            policy=DirectoryParsePolicy.CREATE,
        )

    def get_inspect_payload_directory(self, *, epoch: int) -> Directory:
        self._require_purpose(EnginePurpose.INSPECT)
        self.require_identity()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch must be an int >= 0, got {epoch!r}")
        directory = Directory.from_under(
            root=self._existing_artifacts(),
            relative_path=f"{self.INSPECTION_RUNS_DIRNAME}/epoch={epoch}",
            policy=DirectoryParsePolicy.CREATE,
        )
        self._payload_directory = directory
        return directory

    def get_eval_loaders(self) -> EvalLoaders:
        self._require_purpose(EnginePurpose.INSPECT)
        return self._require_eval_parts().loaders

    def get_evaluate_fn(
        self,
    ) -> Callable[[EvalContext], Sequence[LogItem]]:
        self._require_purpose(EnginePurpose.INSPECT)
        return self._require_eval_parts().evaluate_fn

    def get_eval_device(self) -> torch.device:
        self._require_purpose(EnginePurpose.INSPECT)
        return self._eval_device

    def _require_purpose(self, purpose: str) -> None:
        if self._purpose != purpose:
            raise ValueError(
                f"getter requires purpose {purpose!r}, got {self._purpose!r}"
            )

    def _existing_artifacts(self) -> Directory:
        artifacts = self.get_artifacts_directory()
        if artifacts.spec != DirectorySpec.EXISTS:
            raise FileNotFoundError(
                f"artifacts directory does not exist: {artifacts.path}"
            )
        return artifacts

    def _require_fit_parts(self) -> TrainOnlyFitParts:
        self.require_identity()
        if self._fit_parts is None:
            parts = self._experiment.build_train_only(
                self._train_recipe.experiment_config,
                project_root=self._layout.root,
            )
            if not isinstance(parts, TrainOnlyFitParts):
                raise TypeError(
                    "experiment.build_train_only must return TrainOnlyFitParts, "
                    f"got {type(parts)!r}"
                )
            self._fit_parts = parts
        return self._fit_parts

    def _require_eval_parts(self) -> TrainOnlyEvalParts:
        self.require_identity()
        payload_directory = self._payload_directory
        if payload_directory is None:
            raise RuntimeError(
                "get_inspect_payload_directory must run before build_eval"
            )
        if self._eval_parts is None:
            parts = self._experiment.build_eval(
                self._train_recipe.experiment_config,
                self._inspect_recipe.evaluate,
                project_root=self._layout.root,
                payload_dir=payload_directory,
            )
            if not isinstance(parts, TrainOnlyEvalParts):
                raise TypeError(
                    "experiment.build_eval must return TrainOnlyEvalParts, "
                    f"got {type(parts)!r}"
                )
            self._eval_parts = parts
        return self._eval_parts


def _bind_experiment(experiment: object) -> object:
    if isinstance(experiment, type):
        bound = experiment()
    else:
        bound = experiment
    if not callable(getattr(bound, "build_train_only", None)):
        raise TypeError("experiment must have build_train_only")
    if not callable(getattr(bound, "build_eval", None)):
        raise TypeError("experiment must have build_eval")
    return bound
