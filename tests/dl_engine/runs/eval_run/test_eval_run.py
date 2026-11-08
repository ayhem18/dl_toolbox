from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

from dl_toolbox.base.input_output.json_io import dump_object, flatten_object, JsonTextForm
from dl_toolbox.dl_engine.loggers.logger import (
    LogExistence,
    LogStream,
)
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalContext,
    EvalLoaders,
    EvalLoggerConfig,
    LogItem,
)
from dl_toolbox.dl_engine.runs.eval_run.eval_run import EvalError, emit, evaluate
from tests.dl_engine.runs.support import LinearBinary, RecordingLogger, binary_loader, run_log


class _EvalLogger(RecordingLogger):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str, object, int]] = []
        self.tables: list[tuple[str, list[Mapping[str, Any]], int]] = []

    def log_image(self, tag: str, image: object, *, step: int) -> None:
        self.images.append((tag, image, int(step)))

    def log_table(
        self, tag: str, rows: Sequence[Mapping[str, Any]], *, step: int
    ) -> None:
        self.tables.append((tag, [dict(row) for row in rows], int(step)))


def _logger_config(*, step: int) -> EvalLoggerConfig:
    return EvalLoggerConfig(
        stream=LogStream.INSPECT, existence=LogExistence.REQUIRE, step=step
    )


def _loaders() -> EvalLoaders:
    return EvalLoaders(named={"test": binary_loader(n=8, batch_size=4)})


def test_evaluate_has_no_defaults() -> None:
    params = inspect.signature(evaluate).parameters
    for name, param in params.items():
        assert param.default is inspect.Parameter.empty, name
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_evaluate_setup_emits_and_closes() -> None:
    net = LinearBinary()
    logger = _EvalLogger()
    image = object()

    def evaluate_fn(context: EvalContext) -> list[LogItem]:
        assert context.step == 7
        assert context.device.type == "cpu"
        assert not context.net.training
        assert "test" in context.loaders.named
        return [
            LogItem.scalars(step=context.step, values={"acc": 0.25}),
            LogItem.image(step=context.step, tag="sheet", image=image),
            LogItem.text(step=context.step, tag="note", text="ok"),
            LogItem.table(
                step=context.step,
                tag="rows",
                rows=[{"id": "a", "y": 1}],
            ),
        ]

    evaluate(
        net=net,
        loaders=_loaders(),
        evaluate_fn=evaluate_fn,
        device=torch.device("cpu"),
        logger=logger,
        logger_config=_logger_config(step=7),
        run_log=run_log(),
    )
    assert logger.setup_calls == [(LogStream.INSPECT, LogExistence.REQUIRE)]
    assert logger.close_calls == 1
    assert logger.params == [flatten_object({"task": "unit"})]
    assert logger.texts[0][0] == "run_config"
    assert logger.texts[0][1] == dump_object(
        {"task": "unit"}, form=JsonTextForm.ASCII_PRETTY
    )
    assert logger.texts[0][2] == 0
    assert logger.scalars == [({"acc": 0.25}, 7)]
    assert logger.images == [("sheet", image, 7)]
    assert logger.texts[1] == ("note", "ok", 7)
    assert logger.tables == [("rows", [{"id": "a", "y": 1}], 7)]
    assert net.training is False


def test_evaluate_closes_when_evaluate_fn_raises() -> None:
    logger = _EvalLogger()

    def evaluate_fn(context: EvalContext) -> list[LogItem]:
        del context
        raise RuntimeError("callable failed")

    with pytest.raises(RuntimeError, match="callable failed"):
        evaluate(
            net=LinearBinary(),
            loaders=_loaders(),
            evaluate_fn=evaluate_fn,
            device=torch.device("cpu"),
            logger=logger,
            logger_config=_logger_config(step=0),
            run_log=run_log(),
        )
    assert logger.setup_calls == [(LogStream.INSPECT, LogExistence.REQUIRE)]
    assert logger.close_calls == 1
    assert logger.scalars == []


def test_evaluate_rejects_non_log_item_return() -> None:
    logger = _EvalLogger()

    def evaluate_fn(context: EvalContext) -> list[LogItem]:
        del context
        return [{"acc": 1.0}]  # type: ignore[return-value]

    with pytest.raises(EvalError, match="LogItem"):
        evaluate(
            net=LinearBinary(),
            loaders=_loaders(),
            evaluate_fn=evaluate_fn,
            device=torch.device("cpu"),
            logger=logger,
            logger_config=_logger_config(step=0),
            run_log=run_log(),
        )
    assert logger.close_calls == 1


def test_emit_dispatches_in_list_order() -> None:
    logger = _EvalLogger()
    emit(
        logger,
        [
            LogItem.text(step=1, tag="a", text="one"),
            LogItem.scalars(step=2, values={"x": 1.0}),
            LogItem.text(step=3, tag="b", text="two"),
        ],
    )
    assert logger.texts == [("a", "one", 1), ("b", "two", 3)]
    assert logger.scalars == [({"x": 1.0}, 2)]
