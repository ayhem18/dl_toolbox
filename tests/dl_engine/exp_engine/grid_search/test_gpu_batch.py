from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.dl_engine.dl_engine_utils.visible_gpus import VisibleGpus
from dl_toolbox.dl_engine.exp_engine.grid_search import run_recipes_on_free_gpus


class _FakeProc:
    def __init__(self, code: int) -> None:
        self._code = code

    def wait(self) -> int:
        return self._code


def test_batches_recipes_onto_unique_free_gpus(tmp_path: Path) -> None:
    recipes = tuple(tmp_path / f"r{i}.json" for i in range(5))
    queries = [(1, 3), (1, 3), (2,)]
    launched: list[tuple[str, str]] = []

    def query_free() -> tuple[int, ...]:
        return queries.pop(0)

    def popen(cmd: list[str], env: dict[str, str], **kwargs) -> _FakeProc:
        del kwargs
        launched.append((cmd[2], env[VisibleGpus.CUDA_VISIBLE_DEVICES]))
        return _FakeProc(0)

    run_recipes_on_free_gpus(
        recipes=recipes,
        train_script=tmp_path / "overfit_train.py",
        force=False,
        tracking_backend=None,
        python="python",
        query_free=query_free,
        popen=popen,
    )
    assert len(launched) == 5
    assert launched[0][1] == "1"
    assert launched[1][1] == "3"
    assert launched[2][1] == "1"
    assert launched[3][1] == "3"
    assert launched[4][1] == "2"
    assert queries == []


def test_passes_force_and_tracking_backend(tmp_path: Path) -> None:
    recipe = tmp_path / "r0.json"
    captured: list[list[str]] = []

    def popen(cmd: list[str], env: dict[str, str], **kwargs) -> _FakeProc:
        del env, kwargs
        captured.append(cmd)
        return _FakeProc(0)

    run_recipes_on_free_gpus(
        recipes=(recipe,),
        train_script=tmp_path / "train.py",
        force=True,
        tracking_backend="mlflow",
        python="python",
        query_free=lambda: (0,),
        popen=popen,
    )
    cmd = captured[0]
    assert "--force" in cmd
    assert cmd[cmd.index("--tracking-backend") + 1] == "mlflow"


def test_raises_when_no_free_gpu(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no GPU"):
        run_recipes_on_free_gpus(
            recipes=(tmp_path / "r0.json",),
            train_script=tmp_path / "train.py",
            force=False,
            tracking_backend=None,
            python="python",
            query_free=lambda: (),
            popen=lambda cmd, env, **kwargs: _FakeProc(0),
        )


def test_raises_when_a_child_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit 1"):
        run_recipes_on_free_gpus(
            recipes=(tmp_path / "r0.json",),
            train_script=tmp_path / "train.py",
            force=False,
            tracking_backend=None,
            python="python",
            query_free=lambda: (0,),
            popen=lambda cmd, env, **kwargs: _FakeProc(1),
        )


def test_later_batches_still_run_after_a_child_fails(tmp_path: Path) -> None:
    recipes = tuple(tmp_path / f"r{i}.json" for i in range(3))
    codes = {str(recipes[0]): 1, str(recipes[1]): 0, str(recipes[2]): 0}
    launched: list[str] = []
    queries = [(0,), (1,), (2,)]

    def query_free() -> tuple[int, ...]:
        return queries.pop(0)

    def popen(cmd: list[str], env: dict[str, str], **kwargs) -> _FakeProc:
        del env, kwargs
        launched.append(cmd[2])
        return _FakeProc(codes[cmd[2]])

    with pytest.raises(RuntimeError, match="r0.json"):
        run_recipes_on_free_gpus(
            recipes=recipes,
            train_script=tmp_path / "train.py",
            force=False,
            tracking_backend=None,
            python="python",
            query_free=query_free,
            popen=popen,
        )
    assert launched == [str(recipes[0]), str(recipes[1]), str(recipes[2])]
    assert queries == []


def test_failed_child_error_includes_log_path_and_tail(tmp_path: Path) -> None:
    recipe = tmp_path / "r0.json"

    def popen(cmd: list[str], env: dict[str, str], **kwargs) -> _FakeProc:
        del cmd, env
        stdout = kwargs["stdout"]
        stdout.write("identity index is not an identity index\n")
        stdout.flush()
        return _FakeProc(1)

    with pytest.raises(RuntimeError, match="r0.log") as raised:
        run_recipes_on_free_gpus(
            recipes=(recipe,),
            train_script=tmp_path / "train.py",
            force=False,
            tracking_backend=None,
            python="python",
            query_free=lambda: (0,),
            popen=popen,
        )
    assert "not an identity index" in str(raised.value)
