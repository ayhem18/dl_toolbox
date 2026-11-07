"""Fit and inspect recipe documents. Exact keys. No store URIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import JsonTextForm, dump
from dl_toolbox.base.parsing.parse_numerical import parse_non_negative_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    CheckpointPolicy,
    CheckpointRequest,
)
from dl_toolbox.dl_engine.runs.common.run_config import EarlyStoppingConfig
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)

FIT_RECIPE_KEYS = (
    "human_readable_name",
    "run_group",
    "seed",
    "trainer",
    "checkpoint_policy",
    "early_stopping",
    "logger",
    "experiment_config",
)
INSPECT_RECIPE_KEYS = ("selector", "evaluate")


class FitTrainerDocument(BaseModel):
    model_config = _STRICT

    max_epochs: int = Field(ge=1)
    accelerator: Literal["gpu", "cpu"]
    devices: int | list[int]
    precision: int | str
    enable_progress_bar: bool
    accumulate_grad_batches: int = Field(ge=1)

    @field_validator("devices")
    @classmethod
    def _devices(cls, value: object) -> int | list[int]:
        if isinstance(value, bool):
            raise ValueError("devices must be a GPU count or a list of indices")
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"devices count must be >= 1, got {value}")
            return value
        if not isinstance(value, list):
            raise TypeError(
                f"devices must be an int count or a list of indices, got {type(value)!r}"
            )
        seen: dict[int, None] = {}
        n_devices = len(value)
        parsed: list[int] = []
        for i in range(n_devices):
            index = value[i]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(
                    f"devices[{i}] must be an int GPU index >= 0, got {index!r}"
                )
            if index in seen:
                raise ValueError(f"devices has duplicate index {index}")
            seen[index] = None
            parsed.append(index)
        return parsed

    @field_validator("max_epochs", "accumulate_grad_batches")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"must be an int >= 1, got {value!r}")
        return value


class NoneEarlyStoppingDocument(BaseModel):
    model_config = _STRICT

    name: Literal["none"]


class PatienceEarlyStoppingDocument(BaseModel):
    model_config = _STRICT

    name: Literal["patience"]
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0.0)

    @field_validator("patience")
    @classmethod
    def _patience(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"patience must be an int >= 1, got {value!r}")
        return value

    @field_validator("min_delta")
    @classmethod
    def _min_delta(cls, value: float) -> float:
        if type(value) is not float or value < 0.0:
            raise ValueError(f"min_delta must be a float >= 0, got {value!r}")
        return value


class TensorBoardLoggerTagsDocument(BaseModel):
    model_config = _STRICT

    tags: dict[str, str]


class TensorBoardLoggerEmptyDocument(BaseModel):
    model_config = _STRICT


class MlflowLoggerTagsDocument(BaseModel):
    model_config = _STRICT

    experiment: str
    tags: dict[str, str]

    @field_validator("experiment")
    @classmethod
    def _experiment(cls, value: str) -> str:
        return parse_non_empty_str(value, label="logger.experiment")


class MlflowLoggerExperimentDocument(BaseModel):
    model_config = _STRICT

    experiment: str

    @field_validator("experiment")
    @classmethod
    def _experiment(cls, value: str) -> str:
        return parse_non_empty_str(value, label="logger.experiment")


class InspectRecipeDocument(BaseModel):
    model_config = _STRICT

    selector: str
    evaluate: dict[str, Any]

    @field_validator("selector")
    @classmethod
    def _selector(cls, value: str) -> str:
        return parse_non_empty_str(value, label="selector")


class FitRecipe:
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


class InspectRecipe:
    def __init__(self, *, selector: str, evaluate: dict[str, Any]) -> None:
        self.selector = selector
        self.evaluate = evaluate


def parse_fit_recipe(
    payload: object, *, tracking_backend: str, hash_manager: HashManager
) -> FitRecipe:
    if not isinstance(payload, dict):
        raise TypeError(f"fit recipe must be a JSON object, got {type(payload)!r}")
    require_exact_keys(payload, FIT_RECIPE_KEYS, label="fit recipe")
    try:
        trainer = FitTrainerDocument.model_validate(payload["trainer"])
    except ValidationError as exc:
        raise ValueError(f"fit recipe trainer: {exc}") from exc
    _require_trainer_devices(trainer)
    try:
        checkpoint_policy = CheckpointPolicy.model_validate(payload["checkpoint_policy"])
    except ValidationError as exc:
        raise ValueError(f"fit recipe checkpoint_policy: {exc}") from exc
    early_stopping, early_stopping_payload = _parse_early_stopping(
        payload["early_stopping"]
    )
    tags, mlflow_experiment = _parse_logger(
        payload["logger"], tracking_backend=tracking_backend
    )
    experiment_config = payload["experiment_config"]
    if not isinstance(experiment_config, dict):
        raise TypeError(
            "fit recipe experiment_config must be a JSON object, "
            f"got {type(experiment_config)!r}"
        )
    dump(experiment_config, form=JsonTextForm.CANONICAL)
    seed = parse_non_negative_int(payload["seed"], label="seed")
    name = hash_manager.assert_name(
        parse_non_empty_str(payload["human_readable_name"], label="human_readable_name")
    )
    run_group = ProjectLayout.parse_run_group(payload["run_group"])
    trainer_payload = trainer.model_dump(mode="json")
    return FitRecipe(
        human_readable_name=name,
        run_group=run_group,
        seed=seed,
        trainer_payload=trainer_payload,
        accelerator=trainer.accelerator,
        devices_field=trainer.devices,
        precision=trainer.precision,
        max_epochs=trainer.max_epochs,
        accumulate_grad_batches=trainer.accumulate_grad_batches,
        enable_progress_bar=trainer.enable_progress_bar,
        checkpoint_policy=checkpoint_policy,
        checkpoint_policy_payload=checkpoint_policy.model_dump(mode="json"),
        early_stopping=early_stopping,
        early_stopping_payload=early_stopping_payload,
        tags=tags,
        mlflow_experiment=mlflow_experiment,
        experiment_config=experiment_config,
    )


def parse_inspect_recipe(payload: object) -> InspectRecipe:
    if not isinstance(payload, dict):
        raise TypeError(f"inspect recipe must be a JSON object, got {type(payload)!r}")
    require_exact_keys(payload, INSPECT_RECIPE_KEYS, label="inspect recipe")
    try:
        document = InspectRecipeDocument.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"inspect recipe: {exc}") from exc
    dump(document.evaluate, form=JsonTextForm.CANONICAL)
    CheckpointRequest.parse(document.selector)
    return InspectRecipe(selector=document.selector, evaluate=document.evaluate)


def _require_trainer_devices(trainer: FitTrainerDocument) -> None:
    if trainer.accelerator == "cpu":
        if trainer.devices != []:
            raise ValueError(
                f"cpu trainer.devices must be [], got {trainer.devices!r}"
            )
        return
    if isinstance(trainer.devices, int):
        return
    if len(trainer.devices) < 1:
        raise ValueError("gpu trainer.devices must be a count or a non-empty list")


def _parse_early_stopping(
    payload: object,
) -> tuple[EarlyStoppingConfig, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError(
            f"early_stopping must be a JSON object, got {type(payload)!r}"
        )
    name = payload.get("name")
    try:
        if name == EarlyStoppingConfig.NONE:
            NoneEarlyStoppingDocument.model_validate(payload)
            return EarlyStoppingConfig.none(), {"name": EarlyStoppingConfig.NONE}
        if name == EarlyStoppingConfig.PATIENCE:
            document = PatienceEarlyStoppingDocument.model_validate(payload)
            return (
                EarlyStoppingConfig(
                    name=EarlyStoppingConfig.PATIENCE,
                    patience=document.patience,
                    min_delta=document.min_delta,
                ),
                {
                    "name": EarlyStoppingConfig.PATIENCE,
                    "patience": document.patience,
                    "min_delta": document.min_delta,
                },
            )
    except ValidationError as exc:
        raise ValueError(f"fit recipe early_stopping: {exc}") from exc
    raise ValueError(
        "early_stopping.name must be "
        f"{EarlyStoppingConfig.NONE!r} or {EarlyStoppingConfig.PATIENCE!r}, "
        f"got {name!r}"
    )


def _parse_logger(
    payload: object, *, tracking_backend: str
) -> tuple[dict[str, str], str | None]:
    if not isinstance(payload, dict):
        raise TypeError(f"logger must be a JSON object, got {type(payload)!r}")
    try:
        if tracking_backend == TrackingBackend.TENSORBOARD:
            if "tags" in payload:
                document = TensorBoardLoggerTagsDocument.model_validate(payload)
                return dict(document.tags), None
            TensorBoardLoggerEmptyDocument.model_validate(payload)
            return {}, None
        if tracking_backend == TrackingBackend.MLFLOW:
            if "tags" in payload:
                document_ml = MlflowLoggerTagsDocument.model_validate(payload)
                return dict(document_ml.tags), document_ml.experiment
            experiment_only = MlflowLoggerExperimentDocument.model_validate(payload)
            return {}, experiment_only.experiment
    except ValidationError as exc:
        raise ValueError(f"fit recipe logger: {exc}") from exc
    raise ValueError(f"unknown tracking_backend {tracking_backend!r}")
