"""Train-only recipe document. Same eight keys as a train+val fit recipe."""

from __future__ import annotations

from typing import Any

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import CheckpointPolicy
from dl_toolbox.dl_engine.exp_engine.train_val_engine.recipe import (
    FitRecipe,
    parse_fit_recipe,
    parse_inspect_recipe,
)
from dl_toolbox.dl_engine.runs.common.run_config import EarlyStoppingConfig


class TrainOnlyRecipe:
    def __init__(
        self,
        *,
        human_readable_name: str,
        run_group: tuple[str, ...],
        seed: int,
        trainer_payload: dict[str, Any],
        accelerator: str,
        devices_field: int | list[int],
        precision: int | str,
        max_epochs: int,
        accumulate_grad_batches: int,
        enable_progress_bar: bool,
        checkpoint_policy: CheckpointPolicy,
        checkpoint_policy_payload: dict[str, Any],
        early_stopping: EarlyStoppingConfig,
        early_stopping_payload: dict[str, Any],
        tags: dict[str, str],
        mlflow_experiment: str | None,
        experiment_config: dict[str, Any],
    ) -> None:
        self.human_readable_name = human_readable_name
        self.run_group = run_group
        self.seed = seed
        self.trainer_payload = trainer_payload
        self.accelerator = accelerator
        self.devices_field = devices_field
        self.precision = precision
        self.max_epochs = max_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.enable_progress_bar = enable_progress_bar
        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_policy_payload = checkpoint_policy_payload
        self.early_stopping = early_stopping
        self.early_stopping_payload = early_stopping_payload
        self.tags = tags
        self.mlflow_experiment = mlflow_experiment
        self.experiment_config = experiment_config

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_group": list(self.run_group),
            "experiment_config": self.experiment_config,
            "seed": self.seed,
            "trainer": self.trainer_payload,
            "checkpoint_policy": self.checkpoint_policy_payload,
            "early_stopping": self.early_stopping_payload,
        }


def parse_train_only_recipe(
    payload: object, *, tracking_backend: str, hash_manager: HashManager
) -> TrainOnlyRecipe:
    fit = parse_fit_recipe(
        payload, tracking_backend=tracking_backend, hash_manager=hash_manager
    )
    return _from_fit_recipe(fit)


def _from_fit_recipe(fit: FitRecipe) -> TrainOnlyRecipe:
    return TrainOnlyRecipe(
        human_readable_name=fit.human_readable_name,
        run_group=fit.run_group,
        seed=fit.seed,
        trainer_payload=fit.trainer_payload,
        accelerator=fit.accelerator,
        devices_field=fit.devices_field,
        precision=fit.precision,
        max_epochs=fit.max_epochs,
        accumulate_grad_batches=fit.accumulate_grad_batches,
        enable_progress_bar=fit.enable_progress_bar,
        checkpoint_policy=fit.checkpoint_policy,
        checkpoint_policy_payload=fit.checkpoint_policy_payload,
        early_stopping=fit.early_stopping,
        early_stopping_payload=fit.early_stopping_payload,
        tags=fit.tags,
        mlflow_experiment=fit.mlflow_experiment,
        experiment_config=fit.experiment_config,
    )


__all__ = [
    "TrainOnlyRecipe",
    "parse_inspect_recipe",
    "parse_train_only_recipe",
]
