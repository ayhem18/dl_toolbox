from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.exp_engine.grid_search import (
    GridSkipReason,
    apply_assignment,
    cartesian_assignments,
    expand_param_values,
    parse_grid_search_document,
    select_grid_members,
    set_experiment_param,
)

_STARTING = {
    "data": {"batch_size": 4, "mask_ratio": 0.6},
    "lr_config": {"learning_rate": 0.001},
    "loss_config": {"name": "mse"},
}


def _validate(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"experiment_config must be a dict, got {type(payload)!r}")
    data = payload["data"]
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    batch_size = data["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"data.batch_size must be >= 1, got {batch_size!r}")
    loss = payload["loss_config"]
    if not isinstance(loss, dict):
        raise TypeError("loss_config must be a dict")
    if loss["name"] not in ("mse", "mae"):
        raise ValueError(f"loss_config.name must be mse or mae, got {loss['name']!r}")


def test_parse_grid_search_document() -> None:
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "lr_band",
            "grid_search_config": [
                {
                    "param_name": "lr_config.learning_rate",
                    "param_values": "logspace:1e-4:1e-3:3",
                }
            ],
        }
    )
    assert document.starting_recipe == "base.json"
    assert document.grid_search_exp_name == "lr_band"
    assert len(document.axes) == 1
    assert document.axes[0].param_name == "lr_config.learning_rate"
    assert document.axes[0].values[0] == pytest.approx(1e-4)
    assert document.axes[0].values[-1] == pytest.approx(1e-3)


def test_expand_logspace_and_linspace() -> None:
    log_values = expand_param_values(
        "logspace:1e-4:1e-2:3", label="lr_config.learning_rate"
    )
    assert log_values[0] == pytest.approx(1e-4)
    assert log_values[1] == pytest.approx(1e-3)
    assert log_values[2] == pytest.approx(1e-2)
    lin_values = expand_param_values("linspace:0.1:0.3:3", label="data.mask_ratio")
    assert lin_values[0] == pytest.approx(0.1)
    assert lin_values[1] == pytest.approx(0.2)
    assert lin_values[2] == pytest.approx(0.3)


def test_expand_explicit_list() -> None:
    values = expand_param_values([4, 8], label="data.batch_size")
    assert values == (4, 8)


def test_cartesian_product_last_axis_varies_fastest() -> None:
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "two_axes",
            "grid_search_config": [
                {"param_name": "data.batch_size", "param_values": [4, 8]},
                {"param_name": "data.mask_ratio", "param_values": [0.3, 0.6]},
            ],
        }
    )
    rows = cartesian_assignments(document.axes)
    assert len(rows) == 4
    assert rows[0] == {"data.batch_size": 4, "data.mask_ratio": 0.3}
    assert rows[1] == {"data.batch_size": 4, "data.mask_ratio": 0.6}
    assert rows[2] == {"data.batch_size": 8, "data.mask_ratio": 0.3}
    assert rows[3] == {"data.batch_size": 8, "data.mask_ratio": 0.6}


def test_set_experiment_param_allows_new_leaf() -> None:
    updated = set_experiment_param(_STARTING, "loss_config.args", {"coefficient": 0.5})
    assert updated["loss_config"]["args"]["coefficient"] == 0.5
    assert "args" not in _STARTING["loss_config"]


def test_set_experiment_param_rejects_missing_parent() -> None:
    with pytest.raises(ValueError, match="missing key"):
        set_experiment_param(_STARTING, "lr_config.nested.leaf", 0.1)


def test_apply_assignment_does_not_ask_whether_name_is_legal() -> None:
    candidate = apply_assignment(_STARTING, {"loss_config.name": "unknown"})
    assert candidate["loss_config"]["name"] == "unknown"
    with pytest.raises(ValueError, match="loss_config.name"):
        _validate(candidate)


def test_select_records_invalid_and_does_not_call_keep() -> None:
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "loss_probe",
            "grid_search_config": [
                {
                    "param_name": "loss_config.name",
                    "param_values": ["mse", "unknown"],
                },
                {
                    "param_name": "data.mask_ratio",
                    "param_values": [0.3, 0.6],
                },
            ],
        }
    )
    keep_calls: list[str] = []

    def keep(*, assignment, experiment_config) -> bool:
        del experiment_config
        keep_calls.append(str(assignment["loss_config.name"]))
        return True

    keeps, skips = select_grid_members(
        starting_experiment_config=_STARTING,
        axes=document.axes,
        validate=_validate,
        keep=keep,
    )
    assert len(keeps) == 2
    assert len(skips) == 2
    assert skips[0].reason == GridSkipReason.INVALID
    assert skips[0].assignment["loss_config.name"] == "unknown"
    assert skips[0].cartesian_index == 2
    assert skips[1].assignment["loss_config.name"] == "unknown"
    assert keep_calls == ["mse", "mse"]


def test_filter_drops_legal_members_as_uninteresting() -> None:
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "filter_mask",
            "grid_search_config": [
                {
                    "param_name": "data.mask_ratio",
                    "param_values": [0.3, 0.6],
                }
            ],
        }
    )

    def keep(*, assignment, experiment_config) -> bool:
        del experiment_config
        return assignment["data.mask_ratio"] == 0.6

    keeps, skips = select_grid_members(
        starting_experiment_config=_STARTING,
        axes=document.axes,
        validate=_validate,
        keep=keep,
    )
    assert len(keeps) == 1
    assert keeps[0].assignment["data.mask_ratio"] == 0.6
    assert len(skips) == 1
    assert skips[0].reason == GridSkipReason.UNINTERESTING
    assert skips[0].assignment["data.mask_ratio"] == 0.3
    assert skips[0].error is None


def test_rejects_duplicate_param_name() -> None:
    with pytest.raises(ValueError, match="duplicate param_name"):
        parse_grid_search_document(
            {
                "starting_recipe": "base.json",
                "grid_search_exp_name": "dup",
                "grid_search_config": [
                    {"param_name": "data.batch_size", "param_values": [4]},
                    {"param_name": "data.batch_size", "param_values": [8]},
                ],
            }
        )


def test_rejects_bad_strategy_string() -> None:
    with pytest.raises(ValueError, match="strategy"):
        expand_param_values("logspace:1e-4:1e-3", label="lr")


def test_rejects_extra_document_key() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_grid_search_document(
            {
                "starting_recipe": "base.json",
                "grid_search_exp_name": "extra",
                "grid_search_config": [
                    {"param_name": "data.batch_size", "param_values": [4]},
                ],
                "note": "no",
            }
        )
