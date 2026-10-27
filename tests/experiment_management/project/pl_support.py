"""Shared generators for experiment_management.project tests. Not collected by pytest."""

from __future__ import annotations

import random
import string
from pathlib import Path

from dl_toolbox.experiment_management.experiment.build import ExperimentBuild
from dl_toolbox.experiment_management.experiment.manager import ExperimentConfigManager
from dl_toolbox.experiment_management.project.default_run_manager import DefaultRunManager

N_TRIALS = 20


class UnusedExperimentConfigManager(ExperimentConfigManager):
    def build(self, experiment_config, *, project_root: Path) -> ExperimentBuild:
        del experiment_config, project_root
        raise NotImplementedError("this test does not build experiments")


def make_layout_run_manager(name: str, *, start: Path) -> DefaultRunManager:
    return DefaultRunManager(
        name,
        start=start,
        experiment_config_manager=UnusedExperimentConfigManager(),
    )


def random_project_name(rng: random.Random) -> str:
    alphabet = string.ascii_lowercase + string.digits
    n = rng.randint(6, 14)
    return "proj_" + "".join(rng.choice(alphabet) for _ in range(n))


def random_package_name(rng: random.Random) -> str:
    alphabet = string.ascii_lowercase
    n = rng.randint(4, 10)
    return "pkg_" + "".join(rng.choice(alphabet) for _ in range(n))
