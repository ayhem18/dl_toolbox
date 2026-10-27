from __future__ import annotations

import pytest

from dl_toolbox.experiment_management.logging.run_parser import (
    MlflowRunLoggerConfigParser,
    TensorBoardRunLoggerConfigParser,
    WandbRunLoggerConfigParser,
    parse_run_logger_config,
)


def test_tensorboard_run_parser_requires_run_name() -> None:
    parsed = TensorBoardRunLoggerConfigParser({"run_name": "exp_a"})
    assert parsed.run_name == "exp_a"
    with pytest.raises(ValueError, match="keys must be"):
        TensorBoardRunLoggerConfigParser({})


def test_mlflow_run_parser_does_not_read_store_paths() -> None:
    parsed = MlflowRunLoggerConfigParser(
        {
            "experiment": "occupancy",
            "run_name": "exp_a",
            "tags": {"task": "seg"},
        }
    )
    assert parsed.experiment == "occupancy"
    assert parsed.run_name == "exp_a"
    assert parsed.tags == {"task": "seg"}


def test_wandb_run_parser_requires_project() -> None:
    parsed = WandbRunLoggerConfigParser(
        {"project": "cloud-name", "run_name": "exp_a", "tags": {}}
    )
    assert parsed.project == "cloud-name"
    with pytest.raises(ValueError, match="keys must be"):
        WandbRunLoggerConfigParser({"run_name": "exp_a", "tags": {}})


def test_dispatch_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown tracking backend"):
        parse_run_logger_config("not-a-backend", {"run_name": "x"})
