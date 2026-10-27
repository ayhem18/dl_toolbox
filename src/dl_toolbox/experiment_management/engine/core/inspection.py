"""Project-owned inspect callable. Engine owns fetch / exclusive folders / publish."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dl_toolbox.dl_engine.training.checkpointer.types import CheckpointRecord
from dl_toolbox.experiment_management.engine.core.session import RunSession

InspectFn = Callable[[RunSession, CheckpointRecord, Path], None]


class InspectionManager(ABC):
    @abstractmethod
    def inspect_fn(self, inspection_recipe: Mapping[str, Any]) -> InspectFn:
        raise NotImplementedError
