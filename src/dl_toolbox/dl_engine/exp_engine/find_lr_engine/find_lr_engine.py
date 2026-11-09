"""FindLrEngine is one process's finder engine: bound getters plus occupancy.

find() is zero-arg. Rank 0 claims HashManager occupancy. A worker rank errors.
Occupancy is RunOccupancy. Failed-claim cleanup is the retry path, not resume.
"""

from __future__ import annotations

from abc import abstractmethod

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.dl_engine.exp_engine.process_rank import is_ddp_worker
from dl_toolbox.dl_engine.exp_engine.run_occupancy import (
    IdentityOverwritePolicy,
    RunOccupancy,
    parse_force,
)
from dl_toolbox.dl_engine.runs.common.run_config import RunLogConfig
from dl_toolbox.dl_engine.runs.lr_finder_run.find_lr_run import FindLrResult, find_lr
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrRuntime,
    FindLrSpec,
    FindLrSteps,
)


class FindLrEngineError(ValueError):
    """Worker rank called find()."""


class FindLrEngine(RunOccupancy):
    """One process's finder engine: bound getters plus occupancy."""

    def __init__(self) -> None:
        super().__init__()

    def find(self) -> FindLrResult:
        if is_ddp_worker():
            raise FindLrEngineError(
                "FindLrEngine is single-process; this process is a DDP worker"
            )
        self.claim_fit_identity()
        self.apply_seed()
        try:
            return find_lr(
                net=self.get_net(),
                optimizer=self.get_optimizer(),
                train_loader=self.get_train_loader(),
                steps=self.get_steps(),
                spec=self.get_spec(),
                runtime=self.get_runtime(),
                logger=self.get_logger(),
                artifacts=self.get_artifacts_directory(),
                run_log=self.get_run_log(),
            )
        except BaseException:
            if self._claimed:
                self.cleanup_failed_claim()
            raise

    @abstractmethod
    def apply_seed(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_net(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self) -> Optimizer:
        raise NotImplementedError

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def get_steps(self) -> FindLrSteps:
        raise NotImplementedError

    @abstractmethod
    def get_spec(self) -> FindLrSpec:
        raise NotImplementedError

    @abstractmethod
    def get_runtime(self) -> FindLrRuntime:
        raise NotImplementedError

    @abstractmethod
    def get_run_log(self) -> RunLogConfig:
        raise NotImplementedError


__all__ = [
    "FindLrEngine",
    "FindLrEngineError",
    "IdentityOverwritePolicy",
    "is_ddp_worker",
    "parse_force",
]
