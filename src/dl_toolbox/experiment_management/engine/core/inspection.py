"""Project inspect callable. Engine owns fetch / exclusive folders / publish."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dl_toolbox.dl_engine.training.checkpointer.types import CheckpointRecord
from dl_toolbox.experiment_management.engine.core.session import RunSession

InspectFn = Callable[[RunSession, CheckpointRecord, Path], None]
