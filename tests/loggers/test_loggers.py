import inspect
from pathlib import Path

import pytest

from dl_toolbox.loggers.base import BaseLogger
from dl_toolbox.loggers.placeholder_logger import PlaceholderLogger
from dl_toolbox.loggers.tensorboard_logger import TensorBoardLogger
from dl_toolbox.loggers.wandb_logger import WandbLogger
from dl_toolbox.loggers import get_logger

_METHODS = (
    "__init__",
    "log_scalar",
    "log_dict",
    "log_image",
    "log_table",
    "log_histogram",
    "log_config",
    "close",
)


@pytest.mark.parametrize("cls", [PlaceholderLogger, TensorBoardLogger, WandbLogger])
@pytest.mark.parametrize("name", _METHODS)
def test_logger_signatures_match_base(cls, name: str) -> None:
    base = inspect.signature(getattr(BaseLogger, name))
    impl = inspect.signature(getattr(cls, name))
    assert list(impl.parameters) == list(base.parameters)


def test_time_series_methods_take_step_first() -> None:
    for name in ("log_scalar", "log_dict", "log_image", "log_table", "log_histogram"):
        params = list(inspect.signature(getattr(BaseLogger, name)).parameters)
        assert params[1] == "step"


def test_placeholder_logger_writes_config(tmp_path: Path) -> None:
    logger = get_logger("placeholder", log_dir=str(tmp_path / "run"))
    logger.log_scalar(0, "loss", 1.0)
    logger.log_config({"lr": 1e-3}, "train")
    logger.close()
    assert (tmp_path / "run" / "train.json").read_text(encoding="utf-8")


def test_missing_wandb_explains_extra() -> None:
    try:
        import wandb  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"dl-toolbox\[wandb\]"):
            WandbLogger._import_wandb()
    else:
        pytest.skip("wandb is installed in this environment")
