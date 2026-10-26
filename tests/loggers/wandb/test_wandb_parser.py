from pathlib import Path

import pytest

from dl_toolbox.loggers.wandb.default_configurations import default_wandb_setup
from dl_toolbox.loggers.wandb.wandb_parser import WandbParseError, WandbParser


def test_wandb_store_dict_is_dir_only(tmp_path: Path) -> None:
    parsed = WandbParser(default_wandb_setup(dir=tmp_path / "wandb"))
    assert parsed.dir == tmp_path / "wandb"
    with pytest.raises(WandbParseError):
        WandbParser({"dir": str(tmp_path / "wandb"), "project": "x"})
