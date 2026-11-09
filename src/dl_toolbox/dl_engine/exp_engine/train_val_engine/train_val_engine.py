"""TrainValEngine is one process's train+val engine: purpose plus bound getters.

EnginePurpose is fit or inspect. train() and inspect() take no arguments.
Occupancy is RunOccupancy. Rank 0 claims; other ranks attach.
Failed-claim cleanup is the retry path, not resume.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import CheckpointRequest
from dl_toolbox.dl_engine.exp_engine.process_rank import is_ddp_worker
from dl_toolbox.dl_engine.exp_engine.run_occupancy import (
    IdentityOverwritePolicy,
    RunOccupancy,
    parse_force,
)
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream
from dl_toolbox.dl_engine.runs.common.run_config import (
    FitSteps,
    OptimizationConfig,
    RunLogConfig,
    TrainerConfig,
)
from dl_toolbox.dl_engine.runs.eval_run.eval_config import EvalLoaders, EvalLoggerConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_run import evaluate as eval_evaluate
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import (
    TrainValLoaders,
    TrainValMetrics,
)
from dl_toolbox.dl_engine.runs.train_val_run.train_val_fit import fit as train_val_fit
from dl_toolbox.dl_engine.runs.train_val_run.train_val_module import (
    TrainValLightningModule,
)


class EnginePurpose:
    FIT = "fit"
    INSPECT = "inspect"


class TrainValEngine(RunOccupancy):
    """One process's train+val engine: EnginePurpose plus bound getters.

    train() / inspect() are zero-arg. Rank 0 claims occupancy; other ranks
    attach. Getters are the recipe surface a concrete engine fills.
    """

    def __init__(self, *, purpose: str) -> None:
        super().__init__()
        self._purpose = parse_purpose(purpose)

    def train(self) -> None:
        if self._purpose != EnginePurpose.FIT:
            raise ValueError(
                f"train requires purpose {EnginePurpose.FIT!r}, got {self._purpose!r}"
            )
        self.process_run_identity()
        self.apply_seed()
        try:
            checkpointer = self.get_checkpointer()
            train_val_fit(
                net=self.get_net(),
                loaders=self.get_loaders(),
                steps=self.get_steps(),
                metrics=self.get_metrics(),
                optimization=self.get_optimization(),
                trainer=self.get_trainer(),
                checkpointer=checkpointer,
                logger=self.get_logger(),
                run_log=self.get_run_log(),
            )
        except BaseException:
            if self._claimed:
                self.cleanup_failed_claim()
            raise
        else:
            if self._claimed:
                checkpointer.mark_finished()

    def inspect(self) -> None:
        if self._purpose != EnginePurpose.INSPECT:
            raise ValueError(
                f"inspect requires purpose {EnginePurpose.INSPECT!r}, got {self._purpose!r}"
            )
        self.attach_fit_identity()
        checkpointer = self.get_checkpointer()
        request = self.get_checkpoint_request()
        staging = self.get_inspect_staging_directory()
        record = checkpointer.fetch(request, dest=staging.path)
        self.get_inspect_payload_directory(epoch=record.epoch)
        net = self.get_net()
        TrainValLightningModule.load_net(
            net=net,
            file=record.path,
            map_location=self.get_eval_device(),
        )
        eval_evaluate(
            net=net,
            loaders=self.get_eval_loaders(),
            evaluate_fn=self.get_evaluate_fn(),
            device=self.get_eval_device(),
            logger=self.get_logger(),
            logger_config=EvalLoggerConfig(
                stream=LogStream.INSPECT,
                existence=LogExistence.REQUIRE,
                step=record.epoch,
            ),
            run_log=self.get_run_log(),
        )

    def process_run_identity(self) -> None:
        """Rank 0 claims; a DDP worker attaches. Purpose must be FIT."""
        if self._purpose != EnginePurpose.FIT:
            raise ValueError(
                "process_run_identity requires purpose "
                f"{EnginePurpose.FIT!r}, got {self._purpose!r}"
            )
        super().process_run_identity()

    @abstractmethod
    def apply_seed(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_net(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_loaders(self) -> TrainValLoaders:
        raise NotImplementedError

    @abstractmethod
    def get_steps(self) -> FitSteps:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> TrainValMetrics:
        raise NotImplementedError

    @abstractmethod
    def get_optimization(self) -> OptimizationConfig:
        raise NotImplementedError

    @abstractmethod
    def get_trainer(self) -> TrainerConfig:
        raise NotImplementedError

    @abstractmethod
    def get_checkpointer(self) -> InspectableCheckpointer:
        raise NotImplementedError

    @abstractmethod
    def get_run_log(self) -> RunLogConfig:
        raise NotImplementedError

    @abstractmethod
    def get_checkpoint_request(self) -> CheckpointRequest:
        raise NotImplementedError

    @abstractmethod
    def get_inspect_staging_directory(self) -> Directory:
        raise NotImplementedError

    @abstractmethod
    def get_inspect_payload_directory(self, *, epoch: int) -> Directory:
        raise NotImplementedError

    @abstractmethod
    def get_eval_loaders(self) -> EvalLoaders:
        raise NotImplementedError

    @abstractmethod
    def get_evaluate_fn(self):
        raise NotImplementedError

    @abstractmethod
    def get_eval_device(self) -> torch.device:
        raise NotImplementedError


def parse_purpose(value: object) -> str:
    return parse_one_of(
        value,
        allowed=(EnginePurpose.FIT, EnginePurpose.INSPECT),
        label="purpose",
        error_type=ValueError,
        error_message=one_of_message,
    )


__all__ = [
    "EnginePurpose",
    "IdentityOverwritePolicy",
    "TrainValEngine",
    "is_ddp_worker",
    "parse_force",
    "parse_purpose",
]
