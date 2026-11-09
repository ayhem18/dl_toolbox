"""Find-lr recipe document. Exact keys. No store URIs."""

from __future__ import annotations

from typing import Any

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.input_output.json_io import JsonTextForm, dump_object
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import parse_non_negative_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from dl_toolbox.dl_engine.exp_engine.train_val_engine.recipe import _parse_logger
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrRuntime,
    FindLrSpec,
    parse_find_lr_spec,
)
from dl_toolbox.project_layout.project_layout import ProjectLayout

FIND_LR_RECIPE_KEYS = (
    "human_readable_name",
    "run_group",
    "seed",
    "lr_finder",
    "runtime",
    "logger",
    "experiment_config",
)
RUNTIME_KEYS = ("accelerator", "precision", "enable_progress_bar")


class FindLrRecipe:
    def __init__(
        self,
        *,
        human_readable_name: str,
        run_group: tuple[str, ...],
        seed: int,
        spec: FindLrSpec,
        runtime: FindLrRuntime,
        tags: dict[str, str],
        mlflow_experiment: str | None,
        experiment_config: dict[str, Any],
    ) -> None:
        self.human_readable_name = human_readable_name
        self.run_group = run_group
        self.seed = seed
        self.spec = spec
        self.runtime = runtime
        self.tags = tags
        self.mlflow_experiment = mlflow_experiment
        self.experiment_config = experiment_config

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_group": list(self.run_group),
            "experiment_config": self.experiment_config,
            "seed": self.seed,
            "lr_finder": self.spec.to_dict(),
            "runtime": {
                "accelerator": self.runtime.accelerator,
                "precision": self.runtime.precision,
                "enable_progress_bar": self.runtime.enable_progress_bar,
            },
        }


def parse_find_lr_recipe(
    payload: object, *, tracking_backend: str, hash_manager: HashManager
) -> FindLrRecipe:
    if not isinstance(payload, dict):
        raise TypeError(f"find-lr recipe must be a JSON object, got {type(payload)!r}")
    require_exact_keys(payload, FIND_LR_RECIPE_KEYS, label="find-lr recipe")
    spec = parse_find_lr_spec(payload["lr_finder"])
    runtime = _parse_runtime(payload["runtime"])
    tags, mlflow_experiment = _parse_logger(
        payload["logger"], tracking_backend=tracking_backend
    )
    experiment_config = payload["experiment_config"]
    if not isinstance(experiment_config, dict):
        raise TypeError(
            "find-lr recipe experiment_config must be a JSON object, "
            f"got {type(experiment_config)!r}"
        )
    dump_object(experiment_config, form=JsonTextForm.CANONICAL)
    seed = parse_non_negative_int(payload["seed"], label="seed")
    name = hash_manager.validate_human_readable_name(
        parse_non_empty_str(payload["human_readable_name"], label="human_readable_name")
    )
    run_group = ProjectLayout.parse_run_group(payload["run_group"])
    return FindLrRecipe(
        human_readable_name=name,
        run_group=run_group,
        seed=seed,
        spec=spec,
        runtime=runtime,
        tags=tags,
        mlflow_experiment=mlflow_experiment,
        experiment_config=experiment_config,
    )


def _parse_runtime(payload: object) -> FindLrRuntime:
    obj = parse_mapping(payload, label="runtime")
    require_exact_keys(obj, RUNTIME_KEYS, label="runtime")
    return FindLrRuntime(
        accelerator=obj["accelerator"],
        precision=obj["precision"],
        enable_progress_bar=obj["enable_progress_bar"],
    )
