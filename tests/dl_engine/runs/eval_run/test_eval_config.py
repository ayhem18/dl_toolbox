from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    LogStream,
)
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalLoaders,
    EvalLoggerConfig,
    LogItem,
)
from tests.dl_engine.runs.support import binary_loader


def test_eval_logger_config_has_no_defaults() -> None:
    params = inspect.signature(EvalLoggerConfig).parameters
    for name, param in params.items():
        assert param.default is inspect.Parameter.empty, name


def test_eval_logger_config_accepts_inspect_require() -> None:
    config = EvalLoggerConfig(
        stream=LogStream.INSPECT, existence=LogExistence.REQUIRE, step=4
    )
    assert config.stream == LogStream.INSPECT
    assert config.existence == LogExistence.REQUIRE
    assert config.step == 4


def test_eval_logger_config_rejects_illegal_stream() -> None:
    with pytest.raises(ValidationError):
        EvalLoggerConfig(stream="eval", existence=LogExistence.REQUIRE, step=0)


def test_eval_logger_config_rejects_negative_step() -> None:
    with pytest.raises(ValidationError):
        EvalLoggerConfig(
            stream=LogStream.INSPECT, existence=LogExistence.REQUIRE, step=-1
        )


def test_eval_loaders_stores_named_map() -> None:
    loader = binary_loader(n=8, batch_size=4)
    loaders = EvalLoaders(named={"test": loader})
    assert loaders.named["test"] is loader


def test_eval_loaders_rejects_empty_map() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EvalLoaders(named={})


def test_log_item_constructors_are_keyword_only() -> None:
    for method in (LogItem.scalars, LogItem.image, LogItem.text, LogItem.table):
        params = inspect.signature(method).parameters
        for name, param in params.items():
            if name == "cls":
                continue
            assert param.default is inspect.Parameter.empty, name
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, name


def test_log_item_scalars_and_text() -> None:
    scalars = LogItem.scalars(step=3, values={"acc": 0.5})
    assert scalars.kind == LogItem.SCALARS
    assert scalars.step == 3
    assert scalars.values == {"acc": 0.5}
    text = LogItem.text(step=3, tag="note", text="ok")
    assert text.kind == LogItem.TEXT
    assert text.tag == "note"
    assert text.text == "ok"


def test_log_item_rejects_direct_init() -> None:
    with pytest.raises(TypeError, match="LogItem.scalars"):
        LogItem()


def test_log_item_scalars_rejects_non_float() -> None:
    with pytest.raises(TypeError, match="float"):
        LogItem.scalars(step=0, values={"acc": 1})


def test_log_item_rejects_empty_tag() -> None:
    with pytest.raises(ValueError, match="tag"):
        LogItem.image(step=0, tag="", image=object())
