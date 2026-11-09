from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.dl_engine.exp_engine.find_lr_engine.recipe import (
    FindLrRecipe,
    parse_find_lr_recipe,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    IdentityOverwritePolicy,
)
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import FindLrSpec
from tests.dl_engine.exp_engine.find_lr_engine.support import (
    find_engine,
    find_lr_payload,
    scaffold_project,
    write_recipe,
)
from tests.dl_engine.exp_engine.support import layout_from_root


def test_parse_find_lr_recipe_exact_keys(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_parse", backend="tensorboard")
    layout = layout_from_root(root, name="proj_parse", backend="tensorboard")
    payload = find_lr_payload(name="parsed", run_group=["lr_finder"], seed=2)
    recipe = parse_find_lr_recipe(
        payload,
        tracking_backend=layout.tracking_backend,
        hash_manager=layout.hash_manager,
    )
    assert isinstance(recipe, FindLrRecipe)
    assert recipe.human_readable_name == "parsed"
    assert recipe.run_group == ("lr_finder",)
    assert recipe.seed == 2
    assert recipe.spec.kind == FindLrSpec.RANGE
    extra = dict(payload)
    extra["nope"] = 1
    with pytest.raises(ValueError, match="keys must be"):
        parse_find_lr_recipe(
            extra,
            tracking_backend=layout.tracking_backend,
            hash_manager=layout.hash_manager,
        )
    missing = dict(payload)
    del missing["runtime"]
    with pytest.raises(ValueError, match="keys must be"):
        parse_find_lr_recipe(
            missing,
            tracking_backend=layout.tracking_backend,
            hash_manager=layout.hash_manager,
        )


def test_identity_payload_excludes_logger_labels(tmp_path: Path) -> None:
    root = scaffold_project(tmp_path, name="proj_hash", backend="tensorboard")
    first = write_recipe(
        root,
        relative_path="run_recipes/a.json",
        payload=find_lr_payload(
            name="tagged", run_group=[], seed=4, tags={"task": "one"}
        ),
    )
    second = write_recipe(
        root,
        relative_path="run_recipes/b.json",
        payload=find_lr_payload(
            name="tagged", run_group=[], seed=4, tags={"task": "two"}
        ),
    )
    left = find_engine(
        root, project_name="proj_hash", recipe=first, force=IdentityOverwritePolicy.OFF
    )
    right = find_engine(
        root, project_name="proj_hash", recipe=second, force=IdentityOverwritePolicy.OFF
    )
    assert left.get_identity_payload() == right.get_identity_payload()
    payload = left.get_identity_payload()
    assert "tags" not in payload
    assert "human_readable_name" not in payload
    assert payload["lr_finder"]["kind"] == FindLrSpec.RANGE
    assert payload["runtime"]["accelerator"] == "cpu"
    left.claim_fit_identity()
    with pytest.raises(FileExistsError, match="already claimed"):
        right.claim_fit_identity()
