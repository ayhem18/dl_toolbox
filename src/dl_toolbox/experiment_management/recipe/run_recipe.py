"""Run recipe: label plus the two hashed configs. Name is not hashed."""

from __future__ import annotations

from dataclasses import dataclass

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

_KEYS = ("human_readable_name", "dl_toolbox_config", "experiment_config")


@dataclass(frozen=True)
class RunRecipe:
    human_readable_name: str
    dl_toolbox_config: dict
    experiment_config: dict


class RunRecipeParser:
    def __init__(self, payload: dict, *, hash_manager: HashManager) -> None:
        obj = parse_mapping(payload, label="recipe")
        require_exact_keys(obj, _KEYS, label="recipe")
        self._recipe = RunRecipe(
            human_readable_name=hash_manager.assert_name(obj["human_readable_name"]),
            dl_toolbox_config=parse_mapping(
                obj["dl_toolbox_config"], label="dl_toolbox_config"
            ),
            experiment_config=parse_mapping(
                obj["experiment_config"], label="experiment_config"
            ),
        )

    @property
    def recipe(self) -> RunRecipe:
        return self._recipe

    @property
    def human_readable_name(self) -> str:
        return self._recipe.human_readable_name

    @property
    def dl_toolbox_config(self) -> dict:
        return self._recipe.dl_toolbox_config

    @property
    def experiment_config(self) -> dict:
        return self._recipe.experiment_config
