from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.base.input_output.json_io import read_json
from dl_toolbox.dl_engine.lr_finder import (
    LrFinderError,
    LrFinderRequest,
    find_lr,
    find_lr_full_ds,
)
from tests.dl_engine.loggers.null_logger import NullLogger
from tests.dl_engine.lr_finder.support import make_inputs, make_spec


def test_find_lr_writes_curve_and_closes_logger(tmp_path: Path) -> None:
    torch.manual_seed(3)
    logger = NullLogger()
    spec = make_spec()
    run_config = {"task": "tiny", "lr_finder": spec.to_dict()}
    result = find_lr(
        LrFinderRequest(
            inputs=make_inputs(),
            logger=logger,
            run_dir=tmp_path,
            run_config=run_config,
        )
    )
    assert result.curve_path.is_file()
    payload = read_json(result.curve_path)
    assert len(payload["lrs"]) >= 2
    assert "steepest" in payload["suggested"]
    assert logger.setup_calls == 1
    assert logger.closed
    curve_steps = [
        mapping
        for mapping, _step in logger.scalars
        if "lr_finder/lr" in mapping and "lr_finder/loss" in mapping
    ]
    assert len(curve_steps) == len(payload["lrs"])
    suggested_keys = {
        key
        for mapping, _step in logger.scalars
        for key in mapping
        if key.startswith("suggested/")
    }
    assert "suggested/steepest" in suggested_keys
    assert "suggested/before_min" in suggested_keys
    assert "suggested/min_loss_lr" in suggested_keys


def test_find_lr_full_ds_writes_min_loss_lr(tmp_path: Path) -> None:
    torch.manual_seed(3)
    logger = NullLogger()
    spec = make_spec()
    result = find_lr_full_ds(
        LrFinderRequest(
            inputs=make_inputs(),
            logger=logger,
            run_dir=tmp_path,
            run_config={"task": "tiny", "lr_finder": spec.to_dict()},
        )
    )
    assert result.curve_path.is_file()
    payload = read_json(result.curve_path)
    assert payload["lrs"] == list(result.lrs)
    assert len(payload["lrs"]) == spec.num_iter
    assert payload["suggested"] == {"min_loss_lr": result.suggested["min_loss_lr"]}
    assert logger.setup_calls == 1
    assert logger.closed
    suggested_keys = {
        key
        for mapping, _step in logger.scalars
        for key in mapping
        if key.startswith("suggested/")
    }
    assert suggested_keys == {"suggested/min_loss_lr"}


def test_find_lr_closes_logger_on_failure(tmp_path: Path) -> None:
    x = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    logger = NullLogger()
    with pytest.raises(LrFinderError, match="empty"):
        find_lr(
            LrFinderRequest(
                inputs=make_inputs(loader=loader),
                logger=logger,
                run_dir=tmp_path,
                run_config={"lr_finder": make_spec().to_dict()},
            )
        )
    assert logger.setup_calls == 1
    assert logger.closed
