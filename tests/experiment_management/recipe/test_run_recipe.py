from __future__ import annotations

import pytest

from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.experiment_management.recipe.run_recipe import RunRecipeParser
from tests.experiment_management.em_support import placeholder_recipe


def _hash_manager() -> HashManager:
    return HashManager(
        prefix_len=12,
        human_readable_dirname="human_readable",
        hashes_dirname="hashes",
    )


def test_recipe_parser_does_not_parse_nested_configs() -> None:
    payload = placeholder_recipe(name="run_alpha")
    parsed = RunRecipeParser(payload, hash_manager=_hash_manager())
    assert parsed.human_readable_name == "run_alpha"
    assert parsed.dl_toolbox_config is payload["dl_toolbox_config"]
    assert parsed.experiment_config == {"task": "tiny"}


def test_recipe_parser_rejects_hash_stem_name() -> None:
    manager = _hash_manager()
    digest = manager.hash_payload({"k": 1})
    payload = placeholder_recipe(name=manager.stem(digest))
    with pytest.raises(ValueError, match="hash stem"):
        RunRecipeParser(payload, hash_manager=manager)
