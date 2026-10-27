"""Project-owned experiment construction. Subclass this, not DefaultRunManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dl_toolbox.experiment_management.experiment.build import ExperimentBuild


class ExperimentConfigManager(ABC):
    @abstractmethod
    def build(
        self,
        experiment_config: Mapping[str, Any],
        *,
        project_root: Path,
    ) -> ExperimentBuild:
        raise NotImplementedError
