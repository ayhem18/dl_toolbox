"""Enumerate, then validate, then filter a cartesian over experiment_config.

Generation does not parse. validate raises on an illegal candidate. keep
returns False on a legal but uninteresting candidate. Skips stay in the
index; they are not written as recipes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from itertools import product
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dl_toolbox.base.parsing.parse_iterable import parse_mapping, parse_sequence
from dl_toolbox.base.parsing.parse_numerical import parse_number, parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of
from dl_toolbox.dl_engine.exp_engine.grid_search.suggest import (
    suggest_lin_values,
    suggest_log_learning_rates,
)
from dl_toolbox.project_layout.project_layout import ProjectLayout

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)

ValidateFn = Callable[[object], None]
KeepFn = Callable[..., bool]


class GridValueStrategy:
    """Prefixes for a strategy string in param_values."""

    LOGSPACE = "logspace"
    LINSPACE = "linspace"
    STRATEGIES = (LOGSPACE, LINSPACE)


class GridSkipReason:
    """Why a cartesian cell is not written as a recipe."""

    INVALID = "invalid"
    UNINTERESTING = "uninteresting"
    REASONS = (INVALID, UNINTERESTING)


class GridSearchAxisDocument(BaseModel):
    model_config = _STRICT

    param_name: str
    param_values: object

    @field_validator("param_name")
    @classmethod
    def _param_name(cls, value: str) -> str:
        return parse_non_empty_str(value, label="param_name")


class GridSearchDocumentModel(BaseModel):
    model_config = _STRICT

    starting_recipe: str
    grid_search_exp_name: str
    grid_search_config: list[GridSearchAxisDocument] = Field(min_length=1)

    @field_validator("starting_recipe")
    @classmethod
    def _starting_recipe(cls, value: str) -> str:
        return parse_non_empty_str(value, label="starting_recipe")

    @field_validator("grid_search_exp_name")
    @classmethod
    def _grid_search_exp_name(cls, value: str) -> str:
        return parse_non_empty_str(value, label="grid_search_exp_name")


class GridSearchAxis:
    def __init__(self, *, param_name: str, param_values: object) -> None:
        self.param_name = param_name
        self.param_values = param_values
        self.path = parse_param_name(param_name)
        self.values = expand_param_values(param_values, label=param_name)


class GridSearchDocument:
    def __init__(
        self,
        *,
        starting_recipe: str,
        grid_search_exp_name: str,
        axes: tuple[GridSearchAxis, ...],
    ) -> None:
        self.starting_recipe = starting_recipe
        self.grid_search_exp_name = grid_search_exp_name
        self.axes = axes


class GridKeep:
    def __init__(
        self,
        *,
        cartesian_index: int,
        assignment: dict[str, object],
        experiment_config: dict[str, Any],
    ) -> None:
        self.cartesian_index = cartesian_index
        self.assignment = assignment
        self.experiment_config = experiment_config


class GridSkip:
    def __init__(
        self,
        *,
        cartesian_index: int,
        assignment: dict[str, object],
        reason: str,
        error: str | None,
    ) -> None:
        parsed = parse_one_of(
            reason,
            allowed=GridSkipReason.REASONS,
            label="reason",
            error_type=ValueError,
            error_message=one_of_message,
        )
        if not isinstance(parsed, str):
            raise TypeError(f"reason must be str, got {type(parsed)!r}")
        self.cartesian_index = cartesian_index
        self.assignment = assignment
        self.reason = parsed
        self.error = error


def parse_grid_search_document(payload: object) -> GridSearchDocument:
    if not isinstance(payload, dict):
        raise TypeError(
            f"grid search document must be a JSON object, got {type(payload)!r}"
        )
    try:
        document = GridSearchDocumentModel.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"grid search document: {exc}") from exc
    grid_search_exp_name = ProjectLayout.parse_run_group(
        [document.grid_search_exp_name]
    )[0]
    n_axes = len(document.grid_search_config)
    axes: list[GridSearchAxis | None] = [None] * n_axes
    seen: dict[str, None] = {}
    for i in range(n_axes):
        axis_document = document.grid_search_config[i]
        axis = GridSearchAxis(
            param_name=axis_document.param_name,
            param_values=axis_document.param_values,
        )
        if axis.param_name in seen:
            raise ValueError(
                f"grid_search_config has duplicate param_name {axis.param_name!r}"
            )
        seen[axis.param_name] = None
        axes[i] = axis
    return GridSearchDocument(
        starting_recipe=document.starting_recipe,
        grid_search_exp_name=grid_search_exp_name,
        axes=tuple(item for item in axes if item is not None),
    )


def parse_param_name(raw: object) -> tuple[str, ...]:
    text = parse_non_empty_str(raw, label="param_name")
    if text.startswith(".") or text.endswith("."):
        raise ValueError(f"param_name must not start or end with '.', got {text!r}")
    parts = text.split(".")
    n_parts = len(parts)
    parsed: list[str] = [""] * n_parts
    for i in range(n_parts):
        parsed[i] = parse_non_empty_str(parts[i], label=f"param_name[{i}]")
    return tuple(parsed)


def expand_param_values(raw: object, *, label: str) -> tuple[object, ...]:
    if isinstance(raw, str):
        return _expand_strategy(raw, label=label)
    values = parse_sequence(raw, label=f"{label} param_values")
    if len(values) < 1:
        raise ValueError(f"{label} param_values must not be empty")
    n_values = len(values)
    expanded: list[object] = [None] * n_values
    for i in range(n_values):
        expanded[i] = values[i]
    return tuple(expanded)


def cartesian_assignments(
    axes: tuple[GridSearchAxis, ...],
) -> tuple[dict[str, object], ...]:
    if len(axes) < 1:
        raise ValueError("grid_search_config must not be empty")
    n_axes = len(axes)
    n = 1
    for i in range(n_axes):
        n *= len(axes[i].values)
    rows: list[dict[str, object] | None] = [None] * n
    value_lists = tuple(axis.values for axis in axes)
    index = 0
    for combo in product(*value_lists):
        row: dict[str, object] = {}
        for i in range(n_axes):
            row[axes[i].param_name] = combo[i]
        rows[index] = row
        index += 1
    return tuple(item for item in rows if item is not None)


def apply_assignment(
    experiment_config: Mapping[str, Any],
    assignment: Mapping[str, object],
) -> dict[str, Any]:
    root = copy.deepcopy(
        dict(parse_mapping(experiment_config, label="experiment_config"))
    )
    names = list(assignment)
    n_names = len(names)
    for i in range(n_names):
        name = names[i]
        _set_path(root, param_name=name, value=assignment[name])
    return root


def set_experiment_param(
    experiment_config: Mapping[str, Any],
    param_name: str,
    value: object,
) -> dict[str, Any]:
    root = copy.deepcopy(
        dict(parse_mapping(experiment_config, label="experiment_config"))
    )
    _set_path(root, param_name=param_name, value=value)
    return root


def select_grid_members(
    *,
    starting_experiment_config: Mapping[str, Any],
    axes: tuple[GridSearchAxis, ...],
    validate: ValidateFn,
    keep: KeepFn | None,
) -> tuple[tuple[GridKeep, ...], tuple[GridSkip, ...]]:
    assignments = cartesian_assignments(axes)
    n_assignments = len(assignments)
    keep_slots: list[GridKeep | None] = [None] * n_assignments
    skip_slots: list[GridSkip | None] = [None] * n_assignments
    for i in range(n_assignments):
        assignment = assignments[i]
        candidate = apply_assignment(starting_experiment_config, assignment)
        try:
            validate(candidate)
        except Exception as exc:
            skip_slots[i] = GridSkip(
                cartesian_index=i,
                assignment=dict(assignment),
                reason=GridSkipReason.INVALID,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if keep is not None and not keep(
            assignment=assignment, experiment_config=candidate
        ):
            skip_slots[i] = GridSkip(
                cartesian_index=i,
                assignment=dict(assignment),
                reason=GridSkipReason.UNINTERESTING,
                error=None,
            )
            continue
        keep_slots[i] = GridKeep(
            cartesian_index=i,
            assignment=dict(assignment),
            experiment_config=candidate,
        )
    keeps = tuple(item for item in keep_slots if item is not None)
    skips = tuple(item for item in skip_slots if item is not None)
    return keeps, skips


class GridSearchPersist:
    """Filenames written by write_grid_search."""

    INDEX_FILENAME = "index.json"
    RECIPE_STEM_PREFIX = "exp_"


def experiment_file_stem(*, index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be an int >= 0, got {index!r}")
    return f"{GridSearchPersist.RECIPE_STEM_PREFIX}{index}"


def _set_path(
    root: dict[str, Any], *, param_name: str, value: object
) -> None:
    path = parse_param_name(param_name)
    cursor: dict[str, Any] = root
    n_path = len(path)
    for i in range(n_path - 1):
        key = path[i]
        if key not in cursor:
            raise ValueError(
                f"param_name {param_name!r} is missing key {key!r} "
                f"under experiment_config"
            )
        nested = cursor[key]
        if not isinstance(nested, dict):
            raise TypeError(
                f"param_name {param_name!r} expected an object at {key!r}, "
                f"got {type(nested)!r}"
            )
        cursor = nested
    leaf = path[n_path - 1]
    cursor[leaf] = value


def _expand_strategy(raw: str, *, label: str) -> tuple[float, ...]:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"{label} strategy must be 'logspace:START:END:NUM' or "
            f"'linspace:START:END:NUM', got {raw!r}"
        )
    kind = parse_one_of(
        parts[0],
        allowed=GridValueStrategy.STRATEGIES,
        label=f"{label} strategy",
        error_type=ValueError,
        error_message=one_of_message,
    )
    start = _strategy_number(parts[1], label=f"{label} start")
    end = _strategy_number(parts[2], label=f"{label} end")
    num = _strategy_count(parts[3], label=f"{label} num")
    if kind == GridValueStrategy.LOGSPACE:
        return suggest_log_learning_rates(start_lr=start, end_lr=end, num=num)
    return suggest_lin_values(start=start, end=end, num=num)


def _strategy_number(raw: str, *, label: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number, got {raw!r}") from exc
    return float(parse_number(value, label=label))


def _strategy_count(raw: str, *, label: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int, got {raw!r}") from exc
    return parse_positive_int(value, label=label)
