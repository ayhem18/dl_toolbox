import pytest

from dl_toolbox.dl_engine.loggers.wandb.wandb_manager import WandbManager


def test_missing_wandb_explains_extra() -> None:
    try:
        import wandb  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"dl-toolbox\[wandb\]"):
            WandbManager.import_wandb()
    else:
        pytest.skip("wandb is installed in this environment")
