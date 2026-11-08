"""
- implementation of the TrainValEngine

1. TrainValEngine: accepts - purpose (fit or inspect)

2. the fit logic in a nutshell is 2 steps

- 
- 
- 
- 

"""

from __future__ import annotations

import os
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.file_management.exclusive_lock import ExclusiveDirLock
from dl_toolbox.base.hashing.hash_manager import HashManager, PairState
from dl_toolbox.base.parsing.parse_utils import parse_one_of
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.checkpointer import (
    InspectableCheckpointer,
)
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import CheckpointRequest
from dl_toolbox.dl_engine.loggers.logger import LogExistence, LogStream, Logger
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

IDENTITY_LOCK_NAME = "identity.lock"
IDENTITY_LOCK_POLL_SECONDS = 0.05
ATTACH_POLL_SECONDS = 0.05
RANK_ENV = "RANK"
GLOBAL_RANK_ENV = "GLOBAL_RANK"
NODE_RANK_ENV = "NODE_RANK"
GROUP_RANK_ENV = "GROUP_RANK"
LOCAL_RANK_ENV = "LOCAL_RANK"


class EnginePurpose:
    FIT = "fit"
    INSPECT = "inspect"


class IdentityOverwritePolicy:
    OFF = "off"
    OVERWRITE_NAME = "overwrite_name"

class TrainValEngine(ABC):
    """
    The experiment engine for runs/ experiments with both Train and Validation split. 

    This engine has 3 main functionalities: 

    1. Manage run identities: a run is defined be the hash of its configuration and its human readable name
    The engine uses a HashManager to enforce a clear protocol when working with multiple runs. 

    2. Define the logical steps for parsing and validating a run recipe compatible with the fit and inspect logic of the train_val_run: 

    dl_toolbox/src/dl_toolbox/dl_engine/runs/train_val_run/train_val_fit.py: fit()

    3. Define the inspect logic.  
    """

    def __init__(self, *, purpose: str) -> None:
        self._purpose = parse_purpose(purpose) 
        self._identity_ready = False
        self._claimed_fit = False

    def train(self) -> None:
        if self._purpose != EnginePurpose.FIT:
            raise ValueError(
                f"train requires purpose {EnginePurpose.FIT!r}, got {self._purpose!r}"
            )

        # the process_run_identity logic should probably be revisited         
        self.process_run_identity()
        self.apply_seed()
         
        net = self.get_net()
        loaders = self.get_loaders()
        steps = self.get_steps()
        metrics = self.get_metrics()
        optimization = self.get_optimization()
        trainer = self.get_trainer()
        checkpointer = self.get_checkpointer()
        logger = self.get_logger()
        run_log = self.get_run_log()

        try:
            train_val_fit(
                net=net,
                loaders=loaders,
                steps=steps,
                metrics=metrics,
                optimization=optimization,
                trainer=trainer,
                checkpointer=checkpointer,
                logger=logger,
                run_log=run_log,
            )
        finally:
            if self._claimed_fit:
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
        """
        This is a very important method. 

        - The identity of a run defined by 2 things: 

        1. the hash of its json representation
        2. the human readnable name it was assigned



        """
        if self._purpose != EnginePurpose.FIT:
            raise ValueError(
                "process_run_identity requires purpose "
                f"{EnginePurpose.FIT!r}, got {self._purpose!r}"
            )



        if is_ddp_worker():
            # TODO: this behavior must documented properly
            self.attach_fit_identity()
            return

        self.claim_fit_identity()

    def claim_fit_identity(self) -> None:
        """
        1. extract the run's identity
        2. 
        """


        name = self.get_run_name()
        payload = self.get_identity_payload()
        manager = self.get_hash_manager()
        configuration_hash = manager.hash_payload(payload)
        
        # self.get_force() 
        force = self.get_force()
        listed = self.get_identity_index()
        index = Directory.from_absolute(
            path=listed.path, policy=DirectoryParsePolicy.CREATE
        )
        lock_path = index.path / IDENTITY_LOCK_NAME

        # it is a bit funny that the engine itself must open a lock ... 

        with ExclusiveDirLock(
            lock_path,
            timeout_seconds=self.get_identity_lock_timeout(),
            poll_seconds=IDENTITY_LOCK_POLL_SECONDS,
        ):
            lookup = manager.lookup(
                index, human_readable_name=name, hash=configuration_hash
            )

            if lookup.state == PairState.HASH_TAKEN:
                raise FileExistsError(
                    "configuration hash already saved as another name in this group"
                )

            if lookup.state == PairState.BOTH_MATCH:
                raise FileExistsError(
                    f"already claimed name {name!r} with this configuration hash"
                )

            if lookup.state == PairState.BOTH_FREE:
                self._claim_free_pair(
                    manager=manager,
                    index=index,
                    name=name,
                    configuration_hash=configuration_hash,
                    payload=payload,
                )
                return

            # at this point, the only valid value for lookup.state is PairState.NAME_TAKEN

            # the logic below is for the case of lookup.state = PairState.NAME_TAKEN            
            if force != IdentityOverwritePolicy.OVERWRITE_NAME:
                raise FileExistsError(
                    f"name {name!r} already exists with a different "
                    "configuration hash"
                )
            manager.drop_pair(index, human_readable_name=name)
            artifacts = self.get_artifacts_directory()
            if artifacts.path.exists():
                shutil.rmtree(artifacts.path)
            manager.create_pair(
                index,
                human_readable_name=name,
                hash=configuration_hash,
                extra=payload,
            )
            Directory.from_absolute(
                path=artifacts.path, policy=DirectoryParsePolicy.CREATE
            )
            self._claimed_fit = True
            self._identity_ready = True
            return


    def attach_fit_identity(self) -> None:
        retry = is_ddp_worker()
        name = self.get_run_name()
        payload = self.get_identity_payload()
        manager = self.get_hash_manager()
        configuration_hash = manager.hash_payload(payload)
        deadline = time.monotonic() + self.get_identity_lock_timeout()
        while True:
            index = self.get_identity_index()
            if index.spec != DirectorySpec.EXISTS:
                _raise_or_wait_missing(
                    retry=retry,
                    deadline=deadline,
                    message=f"fit identity index does not exist: {index.path}",
                )
                continue
            lookup = manager.lookup(
                index, human_readable_name=name, hash=configuration_hash
            )
            if lookup.state == PairState.BOTH_MATCH:
                artifacts = self.get_artifacts_directory()
                if artifacts.spec == DirectorySpec.EXISTS:
                    self._identity_ready = True
                    return
                _raise_or_wait_missing(
                    retry=retry,
                    deadline=deadline,
                    message=f"artifacts directory does not exist: {artifacts.path}",
                )
                continue
            if lookup.state == PairState.BOTH_FREE:
                _raise_or_wait_missing(
                    retry=retry,
                    deadline=deadline,
                    message=f"fit identity pair is not claimed: {name!r}",
                )
                continue
            if lookup.state == PairState.NAME_TAKEN:
                raise FileExistsError(
                    f"name {name!r} already exists with a different "
                    "configuration hash"
                )
            if lookup.state == PairState.HASH_TAKEN:
                raise FileExistsError(
                    "configuration hash already saved under a different name"
                )
            raise ValueError(f"unknown pair state {lookup.state!r}")

    def _claim_free_pair(
        self,
        *,
        manager: HashManager,
        index: Directory,
        name: str,
        configuration_hash: str,
        payload: Mapping[str, object],) -> None:
        artifacts = self.get_artifacts_directory()
        if artifacts.spec == DirectorySpec.EXISTS:
            raise FileExistsError(
                f"orphan artifacts directory exists: {artifacts.path}"
            )
        manager.create_pair(
            index,
            human_readable_name=name,
            hash=configuration_hash,
            extra=payload,
        )
        Directory.from_absolute(
            path=artifacts.path, policy=DirectoryParsePolicy.CREATE
        )
        self._claimed_fit = True
        self._identity_ready = True

    def require_identity(self) -> None:
        if not self._identity_ready:
            raise RuntimeError(
                "process_run_identity / attach_fit_identity has not succeeded"
            )

    @abstractmethod
    def get_run_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_identity_payload(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_force(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_hash_manager(self) -> HashManager:
        raise NotImplementedError

    @abstractmethod
    def get_identity_index(self) -> Directory:
        raise NotImplementedError

    @abstractmethod
    def get_identity_lock_timeout(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_artifacts_directory(self) -> Directory:
        raise NotImplementedError

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
    def get_logger(self) -> Logger:
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
        error_message="{label} must be {allowed}, got {value!r}",
    )


def parse_force(value: object) -> str:
    return parse_one_of(
        value,
        allowed=(IdentityOverwritePolicy.OFF, IdentityOverwritePolicy.OVERWRITE_NAME),
        label="overwrite policy",
        error_type=ValueError,
        error_message="{label} must be {allowed}, got {value!r}",
    )


def is_ddp_worker() -> bool:
    # this must definitely move into a common package instead of being burried in here 
    rank = _env_int(RANK_ENV)
    if rank is None:
        rank = _env_int(GLOBAL_RANK_ENV)
    if rank is not None and rank > 0:
        return True
    node = _env_int(NODE_RANK_ENV)
    if node is None:
        node = _env_int(GROUP_RANK_ENV)
    if node is not None and node > 0:
        return True
    local = _env_int(LOCAL_RANK_ENV)
    if local is not None and local > 0:
        return True
    return False


def _env_int(name: str) -> int | None:
    # TODO: add a file to work on the environment variables 
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _raise_or_wait_missing(*, retry: bool, deadline: float, message: str) -> None:
    if not retry or time.monotonic() >= deadline:
        raise FileNotFoundError(message)
    time.sleep(ATTACH_POLL_SECONDS)
