"""Finder run configuration: sweep spec plus accelerator/precision."""

from __future__ import annotations

from dataclasses import dataclass

from dl_toolbox.dl_engine.lr_finder.runtime import LrFinderRuntime
from dl_toolbox.dl_engine.lr_finder.spec import LrFinderSpec


@dataclass(frozen=True)
class LrFinderConfig:
    spec: LrFinderSpec
    runtime: LrFinderRuntime
    num_iterations_per_lr: int = 1
