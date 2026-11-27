"""Write kept grid members. Invalids and uninteresting stay in index.json."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.input_output.json_io import JsonTextForm, write_object
from dl_toolbox.dl_engine.exp_engine.grid_search.grid import (
    GridSearchDocument,
    GridSearchPersist,
    GridSkip,
    KeepFn,
    ValidateFn,
    experiment_file_stem,
    select_grid_members,
)


class StampRecipeFn(Protocol):
    def __call__(
        self,
        starting: dict[str, Any],
        *,
        experiment_config: dict[str, Any],
        grid_search_exp_name: str,
        index: int,
        cartesian_index: int,
    ) -> dict[str, Any]:
        ...


def write_grid_search(
    *,
    starting_recipe: Mapping[str, Any],
    document: GridSearchDocument,
    family: Directory,
    validate: ValidateFn,
    keep: KeepFn | None,
    stamp_recipe: StampRecipeFn,
) -> tuple[Path, ...]:
    experiment_config = starting_recipe["experiment_config"]
    if not isinstance(experiment_config, dict):
        raise TypeError(
            f"experiment_config must be a dict, got {type(experiment_config)!r}"
        )
    starting = dict(starting_recipe)
    keeps, skips = select_grid_members(
        starting_experiment_config=experiment_config,
        axes=document.axes,
        validate=validate,
        keep=keep,
    )
    n_keep = len(keeps)
    paths: list[Path | None] = [None] * n_keep
    index_rows: list[dict[str, Any] | None] = [None] * n_keep
    for i in range(n_keep):
        stem = experiment_file_stem(index=i)
        recipe = stamp_recipe(
            starting,
            experiment_config=keeps[i].experiment_config,
            grid_search_exp_name=document.grid_search_exp_name,
            index=i,
            cartesian_index=keeps[i].cartesian_index,
        )
        written = write_object(
            directory=family,
            name=f"{stem}.json",
            payload=recipe,
            form=JsonTextForm.DOCUMENT,
        )
        paths[i] = written.path
        index_rows[i] = {
            "index": i,
            "cartesian_index": keeps[i].cartesian_index,
            "file": f"{stem}.json",
            "human_readable_name": recipe["human_readable_name"],
            "params": dict(keeps[i].assignment),
        }
    n_skips = len(skips)
    skip_rows: list[dict[str, Any] | None] = [None] * n_skips
    for i in range(n_skips):
        skip_rows[i] = _skip_as_json(skips[i])
    write_object(
        directory=family,
        name=GridSearchPersist.INDEX_FILENAME,
        payload={
            "starting_recipe": document.starting_recipe,
            "grid_search_exp_name": document.grid_search_exp_name,
            "grid_search_config": _axes_as_json(document),
            "experiments": [item for item in index_rows if item is not None],
            "skips": [item for item in skip_rows if item is not None],
        },
        form=JsonTextForm.DOCUMENT,
    )
    _remove_stale_experiments(family.path, n_keep=n_keep)
    return tuple(item for item in paths if item is not None)


def stamp_grid_recipe(
    starting: dict[str, Any],
    *,
    experiment_config: dict[str, Any],
    grid_search_exp_name: str,
    index: int,
    cartesian_index: int,
    run_group: tuple[str, ...],
) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be an int >= 0, got {index!r}")
    if (
        isinstance(cartesian_index, bool)
        or not isinstance(cartesian_index, int)
        or cartesian_index < 0
    ):
        raise ValueError(
            f"cartesian_index must be an int >= 0, got {cartesian_index!r}"
        )
    recipe = copy.deepcopy(starting)
    recipe["experiment_config"] = copy.deepcopy(experiment_config)
    recipe["human_readable_name"] = f"{grid_search_exp_name}_exp_{index}"
    recipe["run_group"] = list(run_group)
    logger = recipe.get("logger")
    if isinstance(logger, dict):
        recipe["logger"] = dict(logger)
        tags = logger.get("tags")
        updated_tags = dict(tags) if isinstance(tags, dict) else {}
        updated_tags["grid_search_exp_name"] = grid_search_exp_name
        updated_tags["grid_index"] = str(index)
        updated_tags["grid_cartesian_index"] = str(cartesian_index)
        recipe["logger"]["tags"] = updated_tags
    return recipe


def _axes_as_json(document: GridSearchDocument) -> list[dict[str, object]]:
    n = len(document.axes)
    rows: list[dict[str, object] | None] = [None] * n
    for i in range(n):
        axis = document.axes[i]
        rows[i] = {
            "param_name": axis.param_name,
            "param_values": axis.param_values,
        }
    return list(item for item in rows if item is not None)


def _skip_as_json(skip: GridSkip) -> dict[str, Any]:
    return {
        "cartesian_index": skip.cartesian_index,
        "reason": skip.reason,
        "error": skip.error,
        "params": dict(skip.assignment),
    }


def _remove_stale_experiments(family_dir: Path, *, n_keep: int) -> None:
    prefix = GridSearchPersist.RECIPE_STEM_PREFIX
    existing = tuple(sorted(family_dir.glob(f"{prefix}*.json")))
    n_existing = len(existing)
    for i in range(n_existing):
        path = existing[i]
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        try:
            index = int(suffix, 10)
        except ValueError:
            path.unlink()
            continue
        if index < 0 or index >= n_keep:
            path.unlink()
