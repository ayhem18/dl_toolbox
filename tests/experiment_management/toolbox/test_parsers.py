from __future__ import annotations

import pytest

from dl_toolbox.dl_engine.training.config import EarlyStoppingConfig
from dl_toolbox.experiment_management.toolbox.checkpoint.parser import CheckpointConfigParser
from dl_toolbox.experiment_management.toolbox.early_stopping.parser import (
    EarlyStoppingConfigParser,
)
from dl_toolbox.experiment_management.toolbox.seed.parser import SeedConfigParser
from dl_toolbox.experiment_management.toolbox.trainer.parser import TrainerConfigParser


def test_trainer_parser_exact_keys() -> None:
    parsed = TrainerConfigParser(
        {
            "max_epochs": 200,
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
            "enable_progress_bar": False,
        }
    )
    assert parsed.trainer_config.max_epochs == 200
    assert parsed.trainer_config.precision == 32
    with pytest.raises(ValueError, match="keys must be"):
        TrainerConfigParser(
            {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": 32,
                "enable_progress_bar": False,
                "batch_size": 8,
            }
        )


def test_seed_parser_rejects_negative() -> None:
    parsed = SeedConfigParser({"seed": 0})
    assert parsed.seed_config.seed == 0
    with pytest.raises(ValueError, match="non-negative"):
        SeedConfigParser({"seed": -1})


def test_early_stopping_none_and_patience() -> None:
    none = EarlyStoppingConfigParser({"name": "none", "params": {}})
    assert none.early_stopping_config.name == EarlyStoppingConfig.NONE
    patience = EarlyStoppingConfigParser(
        {"name": "patience", "params": {"patience": 3, "min_delta": 0.0}}
    )
    assert patience.early_stopping_config.patience == 3
    with pytest.raises(ValueError, match="keys must be"):
        EarlyStoppingConfigParser({"name": "none", "params": {"patience": 1}})


def test_checkpoint_parser_policy_only() -> None:
    parsed = CheckpointConfigParser({"save_top_k": -1, "save_last": True})
    assert parsed.save_top_k == -1
    assert parsed.save_last is True
    with pytest.raises(ValueError, match="keys must be"):
        CheckpointConfigParser(
            {"save_top_k": 1, "save_last": False, "dirpath": "x"}
        )
    with pytest.raises(ValueError, match="save_top_k"):
        CheckpointConfigParser({"save_top_k": -2, "save_last": False})
