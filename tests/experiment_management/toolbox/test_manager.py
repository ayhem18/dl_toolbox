from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.dl_engine._loggers.tensorboard.tensorboard_logger import TensorBoardLogger
from dl_toolbox.dl_engine._loggers.tensorboard.tensorboard_manager import TensorBoardManager
from dl_toolbox.experiment_management.toolbox.manager import DlToolboxConfigManager
from tests.experiment_management.em_support import tensorboard_toolbox_config


def test_toolbox_manager_exact_keys_and_logger(tmp_path: Path) -> None:
    manager = TensorBoardManager(log_dir=tmp_path / "logs")
    toolbox = DlToolboxConfigManager(
        tensorboard_toolbox_config(run_name="run_a"),
        backend="tensorboard",
        logging_manager=manager,
    )
    logger = toolbox.build_logger()
    assert isinstance(logger, TensorBoardLogger)
    assert logger.run_name == "run_a"
    policy = toolbox.checkpoint_policy()
    assert policy.save_top_k == -1
    assert policy.save_last is True


def test_toolbox_manager_rejects_monitor_config(tmp_path: Path) -> None:
    payload = tensorboard_toolbox_config()
    payload["monitor_config"] = {"key": "x", "mode": "min"}
    with pytest.raises(ValueError, match="keys must be"):
        DlToolboxConfigManager(
            payload,
            backend="tensorboard",
            logging_manager=TensorBoardManager(log_dir=tmp_path / "logs"),
        )
