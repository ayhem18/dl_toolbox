from pathlib import Path

import pytest

from dl_toolbox.dl_engine.loggers.base_logger import LoggerError, flatten_mapping
from dl_toolbox.dl_engine.loggers.factory import get_logger
from dl_toolbox.dl_engine.loggers.placeholder.placeholder_manager import PlaceholderManager


def _placeholder_logger(log_dir: Path):
    manager = PlaceholderManager(log_dir=log_dir)
    return get_logger("placeholder", manager=manager, run_name="test")


def test_placeholder_logger_writes_text_and_table(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    logger = _placeholder_logger(log_dir)
    logger.setup()
    logger.log_scalar("loss", 1.0, step=0)
    logger.log_params(flatten_mapping({"lr": 1e-3}))
    logger.log_text("run_config", '{"lr": 0.001}', step=0)
    logger.log_table("inspection/rows", [{"loss": 0.2, "pred": 1}], step=3)
    logger.log_image("inspection/panel", object(), step=3)
    logger.close()
    run_dir = log_dir / "test"
    assert (run_dir / "run_config_step0.txt").read_text(encoding="utf-8") == '{"lr": 0.001}'
    table = (run_dir / "inspection" / "rows_step3.table.json").read_text(encoding="utf-8")
    assert "0.2" in table
    assert (run_dir / "inspection" / "panel_step3.image.txt").is_file()


def test_placeholder_log_before_setup_raises(tmp_path: Path) -> None:
    logger = _placeholder_logger(tmp_path / "run")
    with pytest.raises(LoggerError, match="setup"):
        logger.log_scalars({"loss": 1.0}, step=0)


def test_placeholder_rejects_relative_log_dir() -> None:
    with pytest.raises(ValueError, match="absolute"):
        PlaceholderManager(log_dir=Path("relative/run"))
