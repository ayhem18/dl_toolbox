# PARKED: no contract yet. Not collected by pytest until this package is rewritten.
"""DDP + inspectability for InspectableCheckpointCallback.

Runs a real Lightning ``Trainer.fit`` with world_size 2. Prefers two CUDA
devices; otherwise two CPU processes. Never falls back to a single process.
"""

from __future__ import annotations

import pytest

pytest.skip("Parked: this package has no contract yet.", allow_module_level=True)

import random
import threading
import uuid
from pathlib import Path

import pytest
import torch

from dl_toolbox.base.file_management.exclusive_lock import ExclusiveLockTimeoutError
from dl_toolbox.dl_engine.dl_engine_utils.checkpointing.types import (
    FIT_FINISHED,
    FIT_RUNNING,
    CheckpointLookupError,
    CheckpointRequest,
)
from tests.dl_engine.dl_engine_utils.checkpointing.ddp_support import (
    LOCK_TIMEOUT_SECONDS,
    N_DDP_TRIALS,
    DdpFitConfig,
    catalog_path,
    cuda_device_count,
    fetch_epoch,
    fetch_last,
    fetch_rank1,
    live_ckpt_paths,
    lock_dir,
    best_epoch,
    make_checkpointer,
    run_ddp_fit,
    unique_scores,
)

_SNAP = "snap"

pytestmark = pytest.mark.skipif(
    cuda_device_count() < 2,
    reason="DDP checkpoint tests require at least 2 CUDA devices",
)


def _artifacts(tmp_path: Path, *, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _config(
    run_artifacts_dir: Path,
    *,
    scores: tuple[float, ...],
    save_top_k: int,
    save_last: bool,
    mode: str,
) -> DdpFitConfig:
    return DdpFitConfig(
        run_artifacts_dir=run_artifacts_dir,
        scores=scores,
        save_top_k=save_top_k,
        save_last=save_last,
        mode=mode,
        lock_timeout_seconds=LOCK_TIMEOUT_SECONDS,
    )


def _assert_lock_released(run_artifacts_dir: Path) -> None:
    assert not lock_dir(run_artifacts_dir).exists()


def _assert_single_catalog(run_artifacts_dir: Path) -> None:
    path = catalog_path(run_artifacts_dir)
    assert path.is_file()
    parent = path.parent
    catalogs = list(parent.glob("catalog.json"))
    assert len(catalogs) == 1


def test_ddp_keep_all_every_epoch_is_fetchable(tmp_path: Path) -> None:
    scores = (0.4, 0.1, 0.9, 0.3)
    run_artifacts_dir = _artifacts(tmp_path, name="keep_all")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=-1,
            save_last=True,
            mode="min",
        )
    )
    _assert_single_catalog(run_artifacts_dir)
    _assert_lock_released(run_artifacts_dir)
    state = checkpointer.fit_state()
    assert state.status == FIT_RUNNING
    assert state.latest_epoch == len(scores) - 1
    assert len(live_ckpt_paths(run_artifacts_dir)) == len(scores)
    n = len(scores)
    for i in range(n):
        record = fetch_epoch(
            checkpointer, epoch=i, dest=tmp_path / _SNAP / f"e{i}"
        )
        assert record.epoch == i
        assert record.score == pytest.approx(scores[i])
    last = fetch_last(checkpointer, dest=tmp_path / _SNAP / "last")
    assert last.epoch == n - 1
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    assert rank1.epoch == best_epoch(scores=scores, mode="min")


def test_ddp_min_top1_improving_keeps_only_last(tmp_path: Path) -> None:
    scores = (5.0, 4.0, 3.0, 2.0, 1.0)
    run_artifacts_dir = _artifacts(tmp_path, name="improve")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=1,
            save_last=False,
            mode="min",
        )
    )
    last_epoch = len(scores) - 1
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    assert rank1.epoch == last_epoch
    assert rank1.score == pytest.approx(1.0)
    assert len(live_ckpt_paths(run_artifacts_dir)) == 1
    with pytest.raises(CheckpointLookupError, match="epoch=0"):
        fetch_epoch(checkpointer, epoch=0, dest=tmp_path / _SNAP / "gone")
    assert checkpointer.fit_state().latest_epoch == last_epoch
    _assert_lock_released(run_artifacts_dir)


def test_ddp_min_top1_worsening_skips_after_first(tmp_path: Path) -> None:
    """Skip path under DDP: rank 0 must barrier instead of save_checkpoint."""
    scores = (1.0, 2.0, 3.0, 4.0)
    run_artifacts_dir = _artifacts(tmp_path, name="worsen")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=1,
            save_last=False,
            mode="min",
        )
    )
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    assert rank1.epoch == 0
    assert rank1.score == pytest.approx(1.0)
    assert len(live_ckpt_paths(run_artifacts_dir)) == 1
    with pytest.raises(CheckpointLookupError, match="epoch=3"):
        fetch_epoch(checkpointer, epoch=3, dest=tmp_path / _SNAP / "gone")
    assert checkpointer.fit_state().latest_epoch == 3
    assert checkpointer.fit_state().current_best_epoch == 0
    _assert_lock_released(run_artifacts_dir)


def test_ddp_min_top1_worsening_save_last_keeps_first_and_last(
    tmp_path: Path,
) -> None:
    scores = (1.0, 2.0, 3.0, 4.0)
    run_artifacts_dir = _artifacts(tmp_path, name="worsen_last")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=1,
            save_last=True,
            mode="min",
        )
    )
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    assert rank1.epoch == 0
    last = fetch_last(checkpointer, dest=tmp_path / _SNAP / "last")
    assert last.epoch == 3
    assert last.score == pytest.approx(4.0)
    with pytest.raises(CheckpointLookupError, match="epoch=3"):
        fetch_epoch(checkpointer, epoch=3, dest=tmp_path / _SNAP / "e3")
    assert len(live_ckpt_paths(run_artifacts_dir)) == 2
    _assert_lock_released(run_artifacts_dir)


def test_ddp_max_top1_improving_keeps_only_last(tmp_path: Path) -> None:
    scores = (0.1, 0.2, 0.3, 0.9)
    run_artifacts_dir = _artifacts(tmp_path, name="max_improve")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=1,
            save_last=False,
            mode="max",
        )
    )
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    assert rank1.epoch == 3
    assert rank1.score == pytest.approx(0.9)
    with pytest.raises(CheckpointLookupError, match="epoch=0"):
        fetch_epoch(checkpointer, epoch=0, dest=tmp_path / _SNAP / "gone")
    _assert_lock_released(run_artifacts_dir)


def test_ddp_equal_scores_min_top1_keeps_first_epoch(tmp_path: Path) -> None:
    scores = (0.5, 0.5, 0.5)
    run_artifacts_dir = _artifacts(tmp_path, name="ties")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=1,
            save_last=False,
            mode="min",
        )
    )
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    assert rank1.epoch == 0
    assert len(live_ckpt_paths(run_artifacts_dir)) == 1
    with pytest.raises(CheckpointLookupError, match="epoch=2"):
        fetch_epoch(checkpointer, epoch=2, dest=tmp_path / _SNAP / "gone")
    _assert_lock_released(run_artifacts_dir)


def test_ddp_min_top2_two_clear_winners(tmp_path: Path) -> None:
    scores = (5.0, 0.1, 0.2, 5.0)
    run_artifacts_dir = _artifacts(tmp_path, name="top2")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=2,
            save_last=False,
            mode="min",
        )
    )
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "r1")
    rank2 = checkpointer.fetch(
        CheckpointRequest.parse("rank=2"), dest=tmp_path / _SNAP / "r2"
    )
    assert rank1.epoch == 1
    assert rank2.epoch == 2
    assert len(live_ckpt_paths(run_artifacts_dir)) == 2
    with pytest.raises(CheckpointLookupError, match="epoch=0"):
        fetch_epoch(checkpointer, epoch=0, dest=tmp_path / _SNAP / "e0")
    with pytest.raises(CheckpointLookupError, match="epoch=3"):
        fetch_epoch(checkpointer, epoch=3, dest=tmp_path / _SNAP / "e3")
    _assert_lock_released(run_artifacts_dir)


def test_ddp_fetched_file_is_lightning_checkpoint(tmp_path: Path) -> None:
    scores = (0.2, 0.1)
    run_artifacts_dir = _artifacts(tmp_path, name="payload")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=-1,
            save_last=True,
            mode="min",
        )
    )
    record = fetch_rank1(checkpointer, dest=tmp_path / _SNAP)
    payload = torch.load(record.path, map_location="cpu", weights_only=False)
    assert "state_dict" in payload
    keys = list(payload["state_dict"])
    n = len(keys)
    has_net = False
    for i in range(n):
        if str(keys[i]).startswith("net."):
            has_net = True
            break
    assert has_net


def test_ddp_mark_finished_after_fit(tmp_path: Path) -> None:
    scores = (0.3, 0.2)
    run_artifacts_dir = _artifacts(tmp_path, name="finished")
    checkpointer = run_ddp_fit(
        _config(
            run_artifacts_dir,
            scores=scores,
            save_top_k=-1,
            save_last=True,
            mode="min",
        )
    )
    checkpointer.mark_finished()
    assert checkpointer.fit_state().status == FIT_FINISHED
    _assert_lock_released(run_artifacts_dir)


def test_ddp_randomized_keep_all(tmp_path: Path, rng: random.Random) -> None:
    for trial in range(N_DDP_TRIALS):
        n_epochs = rng.randint(3, 5)
        scores = unique_scores(rng=rng, n_epochs=n_epochs)
        run_artifacts_dir = _artifacts(tmp_path, name=f"keep_all_{trial}")
        checkpointer = run_ddp_fit(
            _config(
                run_artifacts_dir,
                scores=scores,
                save_top_k=-1,
                save_last=True,
                mode=rng.choice(("min", "max")),
            )
        )
        assert len(live_ckpt_paths(run_artifacts_dir)) == n_epochs
        assert checkpointer.fit_state().latest_epoch == n_epochs - 1
        for epoch in range(n_epochs):
            record = fetch_epoch(
                checkpointer,
                epoch=epoch,
                dest=tmp_path / _SNAP / f"t{trial}_e{epoch}",
            )
            assert record.score == pytest.approx(scores[epoch])
        last = fetch_last(
            checkpointer, dest=tmp_path / _SNAP / f"t{trial}_last"
        )
        assert last.epoch == n_epochs - 1
        _assert_lock_released(run_artifacts_dir)


def test_ddp_randomized_top1_unique(tmp_path: Path, rng: random.Random) -> None:
    for trial in range(N_DDP_TRIALS):
        n_epochs = rng.randint(3, 5)
        scores = unique_scores(rng=rng, n_epochs=n_epochs)
        mode = rng.choice(("min", "max"))
        save_last = rng.choice((True, False))
        run_artifacts_dir = _artifacts(tmp_path, name=f"top1_{trial}")
        checkpointer = run_ddp_fit(
            _config(
                run_artifacts_dir,
                scores=scores,
                save_top_k=1,
                save_last=save_last,
                mode=mode,
            )
        )
        expected_best = best_epoch(scores=scores, mode=mode)
        last_epoch = n_epochs - 1
        rank1 = fetch_rank1(
            checkpointer, dest=tmp_path / _SNAP / f"t{trial}_r1"
        )
        assert rank1.epoch == expected_best
        assert rank1.score == pytest.approx(scores[expected_best])
        if save_last:
            last = fetch_last(
                checkpointer, dest=tmp_path / _SNAP / f"t{trial}_last"
            )
            assert last.epoch == last_epoch
            if last_epoch != expected_best:
                with pytest.raises(CheckpointLookupError):
                    fetch_epoch(
                        checkpointer,
                        epoch=last_epoch,
                        dest=tmp_path / _SNAP / f"t{trial}_el",
                    )
                assert len(live_ckpt_paths(run_artifacts_dir)) == 2
            else:
                assert len(live_ckpt_paths(run_artifacts_dir)) == 1
        else:
            assert len(live_ckpt_paths(run_artifacts_dir)) == 1
            if last_epoch != expected_best:
                with pytest.raises(CheckpointLookupError):
                    fetch_epoch(
                        checkpointer,
                        epoch=last_epoch,
                        dest=tmp_path / _SNAP / f"t{trial}_el",
                    )
        assert checkpointer.fit_state().latest_epoch == last_epoch
        assert checkpointer.fit_state().current_best_epoch == expected_best
        _assert_lock_released(run_artifacts_dir)


def test_ddp_inspect_during_fit_does_not_timeout(tmp_path: Path) -> None:
    scores = (0.4, 0.3, 0.2, 0.1)
    run_artifacts_dir = _artifacts(tmp_path, name="inspect")
    config = _config(
        run_artifacts_dir,
        scores=scores,
        save_top_k=-1,
        save_last=True,
        mode="min",
    )
    checkpointer = make_checkpointer(config)
    errors: list[BaseException] = []
    stop = threading.Event()

    def inspect_loop() -> None:
        while not stop.is_set():
            try:
                state = checkpointer.fit_state()
                if state.latest_epoch is None:
                    continue
                dest = tmp_path / "inspect" / uuid.uuid4().hex
                fetch_rank1(checkpointer, dest=dest)
            except CheckpointLookupError:
                continue
            except ExclusiveLockTimeoutError as exc:
                errors.append(exc)
                return
            except BaseException as exc:
                errors.append(exc)
                return

    worker = threading.Thread(target=inspect_loop, daemon=True)
    worker.start()
    try:
        run_ddp_fit(config)
    finally:
        stop.set()
        worker.join(timeout=LOCK_TIMEOUT_SECONDS)
    assert errors == []
    assert worker.is_alive() is False
    rank1 = fetch_rank1(checkpointer, dest=tmp_path / _SNAP / "final")
    assert rank1.epoch == 3
    _assert_lock_released(run_artifacts_dir)


def test_ddp_concurrent_fetches_after_fit(tmp_path: Path) -> None:
    scores = (0.9, 0.2, 0.4)
    run_artifacts_dir = _artifacts(tmp_path, name="concurrent")
    config = _config(
        run_artifacts_dir,
        scores=scores,
        save_top_k=-1,
        save_last=True,
        mode="min",
    )
    run_ddp_fit(config)
    n_workers = 8
    errors: list[BaseException] = []
    epochs: list[int] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            reader = make_checkpointer(config)
            record = fetch_rank1(
                reader, dest=tmp_path / _SNAP / f"w{index}"
            )
            with lock:
                epochs.append(record.epoch)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads: list[threading.Thread] = []
    for i in range(n_workers):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    for i in range(n_workers):
        threads[i].join()
    assert errors == []
    assert len(epochs) == n_workers
    expected = best_epoch(scores=scores, mode="min")
    for i in range(n_workers):
        assert epochs[i] == expected
    _assert_lock_released(run_artifacts_dir)


def test_ddp_claimed_before_fit(tmp_path: Path) -> None:
    run_artifacts_dir = _artifacts(tmp_path, name="claimed")
    checkpointer = make_checkpointer(
        _config(
            run_artifacts_dir,
            scores=(0.1, 0.2),
            save_top_k=-1,
            save_last=True,
            mode="min",
        )
    )
    state = checkpointer.fit_state()
    assert state.status == "claimed"
    assert state.latest_epoch is None
    with pytest.raises(CheckpointLookupError, match="catalog does not exist"):
        fetch_rank1(checkpointer, dest=tmp_path / _SNAP)
