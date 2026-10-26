from pathlib import Path

import pytest

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.loggers.base_logger import LoggerError, flatten_mapping
from dl_toolbox.loggers.factory import get_logger
from dl_toolbox.loggers.placeholder.default_configurations import (
    default_placeholder_setup,
)
from dl_toolbox.loggers.placeholder.placeholder_manager import PlaceholderManager


def _placeholder_logger(log_dir: Path):
    manager = PlaceholderManager(default_placeholder_setup(log_dir=log_dir))
    return get_logger("placeholder", manager=manager)


def test_placeholder_logger_copies_artifact_onto_itself(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    logger = _placeholder_logger(log_dir)
    logger.setup()
    source = log_dir / "train.json"
    write_json(source, {"lr": 1e-3})
    logger.log_artifact(source)
    logger.close()
    assert source.is_file()


def test_placeholder_logger_writes_artifact(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    logger = _placeholder_logger(log_dir)
    source = tmp_path / "train.json"
    write_json(source, {"lr": 1e-3})
    logger.setup()
    logger.log_scalar("loss", 1.0, step=0)
    logger.log_params(flatten_mapping({"lr": 1e-3}))
    logger.log_artifact(source)
    logger.close()
    assert (log_dir / "train.json").read_text(encoding="utf-8")


def test_placeholder_log_before_setup_raises(tmp_path: Path) -> None:
    logger = _placeholder_logger(tmp_path / "run")
    with pytest.raises(LoggerError, match="setup"):
        logger.log_scalars({"loss": 1.0}, step=0)


def test_placeholder_rejects_relative_log_dir() -> None:
    with pytest.raises(ValueError, match="absolute"):
        PlaceholderManager(default_placeholder_setup(log_dir="relative/run"))
