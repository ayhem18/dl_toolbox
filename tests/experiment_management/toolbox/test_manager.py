from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.placeholder.placeholder_logger import PlaceholderLogger
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager
from dl_toolbox.dl_engine.training.config import MonitorConfig
from dl_toolbox.experiment_management.toolbox.manager import DlToolboxConfigManager
from tests.experiment_management.em_support import tensorboard_toolbox_config


def test_toolbox_manager_exact_keys_and_logger(tmp_path: Path) -> None:
    manager = PlaceholderManager(log_dir=tmp_path / "logs")
    toolbox = DlToolboxConfigManager(
        tensorboard_toolbox_config(run_name="run_a"),
        backend="placeholder",
        logging_manager=manager,
    )
    logger = toolbox.build_logger()
    assert isinstance(logger, PlaceholderLogger)
    assert logger.run_name == "run_a"
    checkpoint = toolbox.materialize_checkpoint(
        run_dir=tmp_path / "run",
        monitor=MonitorConfig(key="loss", mode="min"),
    )
    assert checkpoint.save_top_k == -1
    assert checkpoint.run_dir == tmp_path / "run"


def test_toolbox_manager_rejects_monitor_config(tmp_path: Path) -> None:
    payload = tensorboard_toolbox_config()
    payload["monitor_config"] = {"key": "x", "mode": "min"}
    with pytest.raises(ValueError, match="keys must be"):
        DlToolboxConfigManager(
            payload,
            backend="placeholder",
            logging_manager=PlaceholderManager(log_dir=tmp_path / "logs"),
        )
