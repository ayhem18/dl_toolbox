import pytest

from dl_toolbox.loggers.wandb.wandb_manager import WandbManager


def test_missing_wandb_explains_extra() -> None:
    try:
        import wandb  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"dl-toolbox\[wandb\]"):
            WandbManager._import_wandb()
    else:
        pytest.skip("wandb is installed in this environment")
