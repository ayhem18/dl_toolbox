from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.dl_engine.exp_engine.grid_search import (
    GridSearchPersist,
    GridSkipReason,
    parse_grid_search_document,
    stamp_grid_recipe,
    write_grid_search,
)

_STARTING = {
    "human_readable_name": "base",
    "run_group": ["overfit_sanity_check"],
    "seed": 0,
    "trainer": {
        "max_epochs": 2,
        "accelerator": "gpu",
        "devices": 1,
        "precision": 16,
        "enable_progress_bar": True,
        "accumulate_grad_batches": 1,
    },
    "checkpoint_policy": {"save_top_k": 1, "save_last": False},
    "early_stopping": {"name": "none"},
    "logger": {
        "experiment": "overfit_sanity_check",
        "tags": {"task": "probe"},
    },
    "experiment_config": {
        "data": {"batch_size": 4, "mask_ratio": 0.6},
        "lr_config": {"learning_rate": 0.0002},
        "loss_config": {"name": "mse"},
    },
}

_RUN_GROUP = ("overfit_sanity_check", "grid_search_overfit", "lr_band")


def _validate(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"experiment_config must be a dict, got {type(payload)!r}")
    loss = payload["loss_config"]
    if not isinstance(loss, dict):
        raise TypeError("loss_config must be a dict")
    if loss["name"] not in ("mse", "mae"):
        raise ValueError(f"loss_config.name must be mse or mae, got {loss['name']!r}")


def _stamp(
    starting: dict,
    *,
    experiment_config: dict,
    grid_search_exp_name: str,
    index: int,
    cartesian_index: int,
) -> dict:
    return stamp_grid_recipe(
        starting,
        experiment_config=experiment_config,
        grid_search_exp_name=grid_search_exp_name,
        index=index,
        cartesian_index=cartesian_index,
        run_group=(
            "overfit_sanity_check",
            "grid_search_overfit",
            grid_search_exp_name,
        ),
    )


def test_stamp_grid_recipe_sets_compact_and_cartesian_index() -> None:
    recipe = stamp_grid_recipe(
        _STARTING,
        experiment_config=_STARTING["experiment_config"],
        grid_search_exp_name="native_b4_lr_band",
        index=0,
        cartesian_index=3,
        run_group=_RUN_GROUP,
    )
    assert recipe["human_readable_name"] == "native_b4_lr_band_exp_0"
    assert recipe["run_group"] == [
        "overfit_sanity_check",
        "grid_search_overfit",
        "lr_band",
    ]
    assert recipe["logger"]["tags"]["grid_index"] == "0"
    assert recipe["logger"]["tags"]["grid_cartesian_index"] == "3"
    assert _STARTING["experiment_config"]["lr_config"]["learning_rate"] == 0.0002


def test_write_grid_search_uses_exp_i_and_index(tmp_path: Path) -> None:
    family = Directory.from_absolute(
        path=tmp_path / "lr_band",
        policy=DirectoryParsePolicy.CREATE,
    )
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "lr_band",
            "grid_search_config": [
                {
                    "param_name": "lr_config.learning_rate",
                    "param_values": [0.0001, 0.001],
                }
            ],
        }
    )
    paths = write_grid_search(
        starting_recipe=_STARTING,
        document=document,
        family=family,
        validate=_validate,
        keep=None,
        stamp_recipe=_stamp,
    )
    assert [path.name for path in paths] == ["exp_0.json", "exp_1.json"]
    first = read_object(file=File.from_absolute(path=paths[0]))
    assert first["experiment_config"]["lr_config"]["learning_rate"] == 0.0001
    assert first["human_readable_name"] == "lr_band_exp_0"
    assert first["logger"]["tags"]["grid_cartesian_index"] == "0"
    index = read_object(
        file=File.from_absolute(path=paths[0].parent / GridSearchPersist.INDEX_FILENAME)
    )
    assert index["experiments"][0]["file"] == "exp_0.json"
    assert index["experiments"][1]["params"]["lr_config.learning_rate"] == 0.001
    assert index["skips"] == []


def test_write_grid_search_skips_illegal_variants(tmp_path: Path) -> None:
    family = Directory.from_absolute(
        path=tmp_path / "loss_probe",
        policy=DirectoryParsePolicy.CREATE,
    )
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "loss_probe",
            "grid_search_config": [
                {
                    "param_name": "loss_config.name",
                    "param_values": ["mse", "unknown"],
                }
            ],
        }
    )
    paths = write_grid_search(
        starting_recipe=_STARTING,
        document=document,
        family=family,
        validate=_validate,
        keep=None,
        stamp_recipe=_stamp,
    )
    assert [path.name for path in paths] == ["exp_0.json"]
    index = read_object(
        file=File.from_absolute(path=paths[0].parent / GridSearchPersist.INDEX_FILENAME)
    )
    assert len(index["skips"]) == 1
    skip = index["skips"][0]
    assert skip["reason"] == GridSkipReason.INVALID
    assert skip["params"]["loss_config.name"] == "unknown"
    assert skip["cartesian_index"] == 1
