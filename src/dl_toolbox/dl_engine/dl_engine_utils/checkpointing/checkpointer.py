"""Inspectable catalog checkpointer. Lock + policy + store. No Lightning."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.exclusive_lock import ExclusiveDirLock
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.file_management.paths import abs_path
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.policy import decide_keep
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.store import CheckpointStore
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    FIT_CLAIMED,
    FIT_FINISHED,
    LIVE_DIRNAME,
    CheckpointMonitor,
    CheckpointPolicy,
    CheckpointRecord,
    CheckpointRequest,
    CheckpointSaveError,
    FitRunState,
)

if TYPE_CHECKING:
    from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.callback import (
        InspectableCheckpointCallback,
    )

_ID_LEN = 12


class InspectableCheckpointer:
    """Locked accessor for the Checkpoint Directory.

    Rank-0 of a DDP fit and an inspect process take ``ExclusiveDirLock``.
    ``Trainer.save_checkpoint`` / DDP barriers live in
    ``InspectableCheckpointCallback``.
    """

    def __init__(
        self,
        *,
        run_artifacts_dir: Path,
        policy: CheckpointPolicy,
        monitor: CheckpointMonitor,
        max_epochs: int,
        lock_timeout_seconds: float,
    ) -> None:
        timeout = float(lock_timeout_seconds)
        if timeout <= 0:
            raise ValueError(
                f"lock_timeout_seconds must be > 0, got {lock_timeout_seconds}"
            )
        artifacts = Directory.from_absolute(
            path=abs_path(run_artifacts_dir),
            policy=DirectoryParsePolicy.MUST_EXIST,
        )
        self._store = CheckpointStore(
            run_artifacts_dir=artifacts,
            policy=policy,
            monitor=monitor,
            max_epochs=max_epochs,
        )
        self._policy = policy
        self._monitor = monitor
        self._max_epochs = max_epochs
        self._lock = ExclusiveDirLock(
            self._store.lock_dir,
            timeout_seconds=timeout,
        )

    @property
    def monitor_key(self) -> str:
        return self._monitor.key

    def as_callback(self) -> InspectableCheckpointCallback:
        from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.callback import (
            InspectableCheckpointCallback,
        )

        return InspectableCheckpointCallback(checkpointer=self)

    def ddp_join_path(self) -> Path:
        """Path non-zero ranks pass to ``Trainer.save_checkpoint`` for the barrier."""
        return self._store.ddp_join_path()

    def save(
        self,
        *,
        epoch: int,
        score: float,
        write_function: Callable[[Path], None],
    ) -> bool:
        """Decide keep, then write. Returns whether a file was written."""
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self._lock.acquire()
        try:
            catalog = self._store.read_or_empty()
            current_files = self._store.current_scored(catalog)
            if not decide_keep(
                current_files=current_files,
                score=score,
                policy=self._policy,
                mode=self._monitor.mode,
            ):
                catalog = self._store.stamp_fit(catalog, epoch=epoch)
                self._store.write_catalog(catalog)
                return False
            self._store.ensure_live()
            checkpoint_id = _new_id()
            path = self._store.live_file_path(checkpoint_id)
            committed = False
            try:
                write_function(path)
                try:
                    written = File.from_absolute(path=path)
                except FileNotFoundError as exc:
                    raise CheckpointSaveError(
                        f"write_function did not create checkpoint file {path}"
                    ) from exc
                catalog = self._store.commit_new(
                    catalog,
                    epoch=epoch,
                    score=score,
                    checkpoint_id=checkpoint_id,
                    rel_path=f"{LIVE_DIRNAME}/{checkpoint_id}.ckpt",
                    written=written,
                )
                self._store.write_catalog(catalog)
                committed = True
                return True
            finally:
                if not committed and path.exists():
                    path.unlink()
        finally:
            self._lock.release()

    def fetch(
        self, request: CheckpointRequest, *, dest: Path
    ) -> CheckpointRecord:
        dest_dir = Directory.from_absolute(
            path=abs_path(dest),
            policy=DirectoryParsePolicy.CREATE,
        )
        self._lock.acquire()
        try:
            return self._store.fetch(request, dest=dest_dir)
        finally:
            self._lock.release()

    def fit_state(self) -> FitRunState:
        if not self._store.catalog_exists():
            return FitRunState(
                status=FIT_CLAIMED,
                latest_epoch=None,
                current_best_epoch=None,
                max_epochs=self._max_epochs,
            )
        catalog = self._store.read_catalog()
        return FitRunState(
            status=catalog.fit.status,
            latest_epoch=catalog.fit.latest_epoch,
            current_best_epoch=self._store.best_epoch(catalog),
            max_epochs=catalog.fit.max_epochs,
        )

    def mark_finished(self) -> None:
        self._lock.acquire()
        try:
            catalog = self._store.read_or_empty()
            catalog = catalog.model_copy(
                update={
                    "fit": catalog.fit.model_copy(update={"status": FIT_FINISHED})
                }
            )
            self._store.write_catalog(catalog)
        finally:
            self._lock.release()


def _new_id() -> str:
    return uuid.uuid4().hex[:_ID_LEN]
