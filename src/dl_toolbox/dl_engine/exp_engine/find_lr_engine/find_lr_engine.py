"""FindLrEngine is one process's finder engine: bound getters plus occupancy.

find() is zero-arg. Rank 0 claims HashManager occupancy. A worker rank errors.
IdentityOverwritePolicy is off or overwrite_name (NAME_TAKEN only).
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    IdentityOverwritePolicy,
)
from dl_toolbox.dl_engine.loggers.logger import Logger
from dl_toolbox.dl_engine.runs.common.run_config import RunLogConfig
from dl_toolbox.dl_engine.runs.lr_finder_run.find_lr_run import FindLrResult, find_lr
from dl_toolbox.dl_engine.runs.lr_finder_run.spec import (
    FindLrRuntime,
    FindLrSpec,
    FindLrSteps,
)


class FindLrEngineError(ValueError):
    """Worker rank called find()."""


class FindLrEngine(ABC):
    """One process's finder engine: bound getters plus occupancy."""

    RANK_ENV = "RANK"
    GLOBAL_RANK_ENV = "GLOBAL_RANK"
    NODE_RANK_ENV = "NODE_RANK"
    GROUP_RANK_ENV = "GROUP_RANK"
    LOCAL_RANK_ENV = "LOCAL_RANK"

    def __init__(self) -> None:
        self._identity_ready = False
        self._claimed = False

    def find(self) -> FindLrResult:
        if is_ddp_worker():
            raise FindLrEngineError(
                "FindLrEngine is single-process; this process is a DDP worker"
            )
        self.claim_identity()
        self.apply_seed()
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

    def claim_identity(self) -> None:
        """HashManager lookup, then create or overwrite. HashManager owns identity.lock."""
        name = self.get_run_name()
        payload = self.get_identity_payload()
        manager = self.get_hash_manager()
        configuration_hash = manager.hash_payload(payload)
        force = self.get_force()
        index = manager.create_identity_index(self.get_identity_index())
        lookup = manager.lookup(
            index, human_readable_name=name, hash=configuration_hash
        )
        if lookup.state == HashManager.HASH_TAKEN:
            raise FileExistsError(
                "configuration hash already saved as another name in this group"
            )
        if lookup.state == HashManager.BOTH_MATCH:
            raise FileExistsError(
                f"already claimed name {name!r} with this configuration hash"
            )
        if lookup.state == HashManager.BOTH_FREE:
            self._claim_free_pair(
                manager=manager,
                index=index,
                name=name,
                configuration_hash=configuration_hash,
                payload=payload,
            )
            return
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
        self._claimed = True
        self._identity_ready = True

    def _claim_free_pair(
        self,
        *,
        manager: HashManager,
        index: Directory,
        name: str,
        configuration_hash: str,
        payload: Mapping[str, object],
    ) -> None:
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
        self._claimed = True
        self._identity_ready = True

    def require_identity(self) -> None:
        if not self._identity_ready:
            raise RuntimeError("claim_identity has not succeeded")

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
    def get_logger(self) -> Logger:
        raise NotImplementedError

    @abstractmethod
    def get_run_log(self) -> RunLogConfig:
        raise NotImplementedError


def is_ddp_worker() -> bool:
    rank = _env_int(FindLrEngine.RANK_ENV)
    if rank is None:
        rank = _env_int(FindLrEngine.GLOBAL_RANK_ENV)
    if rank is not None and rank > 0:
        return True
    node = _env_int(FindLrEngine.NODE_RANK_ENV)
    if node is None:
        node = _env_int(FindLrEngine.GROUP_RANK_ENV)
    if node is not None and node > 0:
        return True
    local = _env_int(FindLrEngine.LOCAL_RANK_ENV)
    if local is not None and local > 0:
        return True
    return False


def _env_int(name: str) -> int | None:
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


__all__ = [
    "FindLrEngine",
    "FindLrEngineError",
    "IdentityOverwritePolicy",
    "is_ddp_worker",
]
