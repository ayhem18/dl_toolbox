"""RunOccupancy is one run's HashManager pair plus artifacts Directory.

TrainValEngine, TrainOnlyEngine, and FindLrEngine inherit this mixin.
Failed-claim cleanup is the retry path. It is not resume: drop_pair,
delete the artifacts Directory completely, Logger.cleanup.
"""

from __future__ import annotations

import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
    DirectorySpec,
)
from dl_toolbox.base.hashing.hash_manager import HashManager
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of
from dl_toolbox.dl_engine.exp_engine.process_rank import is_ddp_worker
from dl_toolbox.dl_engine.loggers.logger import Logger


class IdentityOverwritePolicy:
    OFF = "off"
    OVERWRITE_NAME = "overwrite_name"


class RunOccupancy(ABC):
    """Mixin: occupancy of one run (pair + artifacts Directory)."""

    ATTACH_POLL_SECONDS = 0.05

    def __init__(self) -> None:
        super().__init__()
        self._identity_ready = False
        self._claimed = False

    def process_run_identity(self) -> None:
        """Worker attaches; rank 0 claims."""
        if is_ddp_worker():
            self.attach_fit_identity()
            return
        self.claim_fit_identity()

    def claim_fit_identity(self) -> None:
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
            if lookup.state == HashManager.BOTH_MATCH:
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
            if lookup.state == HashManager.BOTH_FREE:
                _raise_or_wait_missing(
                    retry=retry,
                    deadline=deadline,
                    message=f"fit identity pair is not claimed: {name!r}",
                )
                continue
            if lookup.state == HashManager.NAME_TAKEN:
                raise FileExistsError(
                    f"name {name!r} already exists with a different "
                    "configuration hash"
                )
            if lookup.state == HashManager.HASH_TAKEN:
                raise FileExistsError(
                    "configuration hash already saved under a different name"
                )
            raise ValueError(f"unknown pair state {lookup.state!r}")

    def cleanup_failed_claim(self) -> None:
        """Undo occupancy this process created. Not resume: pair, artifacts, logger go."""
        if not self._claimed:
            raise RuntimeError(
                "cleanup_failed_claim requires this process claimed the pair"
            )
        self.get_logger().cleanup()
        manager = self.get_hash_manager()
        index = self.get_identity_index()
        manager.drop_pair(index, human_readable_name=self.get_run_name())
        artifacts = self.get_artifacts_directory()
        if artifacts.path.exists():
            shutil.rmtree(artifacts.path)
        self._claimed = False
        self._identity_ready = False

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
            raise RuntimeError(
                "claim_fit_identity / attach_fit_identity has not succeeded"
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
    def get_logger(self) -> Logger:
        raise NotImplementedError


def parse_force(value: object) -> str:
    return parse_one_of(
        value,
        allowed=(IdentityOverwritePolicy.OFF, IdentityOverwritePolicy.OVERWRITE_NAME),
        label="overwrite policy",
        error_type=ValueError,
        error_message=one_of_message,
    )


def _raise_or_wait_missing(*, retry: bool, deadline: float, message: str) -> None:
    if not retry or time.monotonic() >= deadline:
        raise FileNotFoundError(message)
    time.sleep(RunOccupancy.ATTACH_POLL_SECONDS)


__all__ = [
    "IdentityOverwritePolicy",
    "RunOccupancy",
    "parse_force",
]
